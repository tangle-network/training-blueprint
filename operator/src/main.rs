use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use blueprint_crypto::KeyType;
use blueprint_crypto::k256::K256Ecdsa;
use blueprint_networking::service::{AllowedKeys, NetworkCommandMessage, NetworkConfig as NetConfig};

use distributed_training::config::OperatorConfig;
use distributed_training::network::{self, TrainingNetwork, MOMENTUM_TOPIC, COORDINATION_TOPIC};
use distributed_training::TrainingServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let bandwidth = config.training.network_bandwidth_mbps;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (gpu_count, total_vram, bandwidth, gpu_model, endpoint).abi_encode()
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit payload and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            blueprint_sdk::std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        blueprint_sdk::std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            gpu_count = config.gpu.expected_gpu_count,
            vram_mib = config.gpu.min_vram_mib,
            bandwidth_mbps = config.training.network_bandwidth_mbps,
            "Registration payload saved"
        );
        return Ok(());
    }

    // Detect GPUs
    match tangle_inference_core::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed — running in CPU mode");
        }
    }

    // --- Start real libp2p networking ---
    let listen_addr = config.network.listen_addr.clone();
    let bootstrap_peers = config.network.bootstrap_peers.clone();

    let instance_key_pair = K256Ecdsa::generate_with_seed(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("key generation failed: {e}")))?;
    let local_key = libp2p::identity::Keypair::generate_ed25519();

    let net_config = NetConfig::<K256Ecdsa> {
        network_name: "distributed-training".to_string(),
        instance_id: "1.0.0".to_string(),
        instance_key_pair,
        local_key: local_key.clone(),
        listen_addr: listen_addr.parse().unwrap_or_else(|_| "/ip4/0.0.0.0/tcp/0".parse().unwrap()),
        target_peer_count: 10,
        bootstrap_peers: bootstrap_peers
            .iter()
            .filter_map(|p| p.parse().ok())
            .collect(),
        enable_mdns: true,
        enable_kademlia: true,
        using_evm_address_for_handshake_verification: false,
    };

    let (_allowed_keys_tx, allowed_keys_rx) = crossbeam_channel::unbounded();
    let net_service = blueprint_networking::service::NetworkService::new(
        net_config,
        AllowedKeys::default(), // empty set — will be updated as peers register
        allowed_keys_rx,
    ).map_err(|e| blueprint_sdk::Error::Other(format!("networking init failed: {e}")))?;

    let peer_id = net_service.get_listen_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let mut handle = net_service.start();

    // Subscribe to training gossip topics
    let _ = handle.send_network_message(NetworkCommandMessage::SubscribeToTopic(
        MOMENTUM_TOPIC.to_string(),
    ));
    let _ = handle.send_network_message(NetworkCommandMessage::SubscribeToTopic(
        COORDINATION_TOPIC.to_string(),
    ));

    tracing::info!(peer_id = %peer_id, "libp2p networking started");

    // Create training network with real peer ID
    let training_network = Arc::new(TrainingNetwork::new(&config.network, peer_id));

    // Spawn gossip event loop — routes incoming messages to TrainingNetwork
    let net_clone = training_network.clone();
    tokio::spawn(async move {
        network::run_gossip_event_loop(
            move || handle.next_protocol_message(),
            net_clone,
        ).await;
    });

    tracing::info!("gossip event loop started on topics: {}, {}", MOMENTUM_TOPIC, COORDINATION_TOPIC);

    // --- Tangle protocol setup ---
    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        match distributed_training::qos::start_heartbeat(config.clone()).await {
            Ok(_handle) => {
                let interval = config.qos.as_ref().unwrap().heartbeat_interval_secs;
                tracing::info!(interval_secs = interval, "QoS heartbeat started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS heartbeat failed to start");
            }
        }
    }

    let training_server = TrainingServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(distributed_training::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(training_server)
        .run()
        .await?;

    Ok(())
}
