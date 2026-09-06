//! Standalone HTTP-only operator for local testing.
//!
//! This binary starts the training job HTTP API without the Tangle chain
//! listener, libp2p gossip, or BlueprintRunner. It is intended for local
//! integration testing (e.g. the autoresearch adapter) and development only.

use std::sync::Arc;

use tangle_inference_core::{
    config::{BillingConfig, PaymentRails, TangleConfig},
    AppState,
};

use blueprint_crypto::k256::K256Ecdsa;
use blueprint_crypto::KeyType;
use blueprint_networking::service::{AllowedKeys, NetworkConfig as NetConfig, NetworkService};
use distributed_training::config::OperatorConfig;
use distributed_training::coordinator::TrainingCoordinator;
use distributed_training::server::TrainingAppBackend;
use distributed_training::{register_coordinator, server};

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

fn dummy_tangle_config() -> TangleConfig {
    TangleConfig {
        rpc_url: "http://localhost:8545".to_string(),
        chain_id: 31337,
        operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            .to_string(),
        shielded_credits: "0x0000000000000000000000000000000000000002".to_string(),
        blueprint_id: 1,
        service_id: None,
    }
}

fn dummy_billing_config() -> BillingConfig {
    BillingConfig {
        payment_rails: PaymentRails::NONE,
        billing_required: false,
        max_spend_per_request: 0,
        min_credit_balance: 0,
        min_charge_amount: 0,
        claim_max_retries: 0,
        clock_skew_tolerance_secs: 30,
        max_gas_price_gwei: 0,
        nonce_store_path: None,
        direct_replay_store_path: None,
        payment_token_address: None,
    }
}

#[tokio::main]
#[expect(
    clippy::result_large_err,
    reason = "This process entrypoint returns the SDK error once at shutdown; boxing adds no useful boundary."
)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    // Use the configured server address, but dummy Tangle/billing so the HTTP
    // gates are open for local tests.
    let tangle = dummy_tangle_config();
    let billing = dummy_billing_config();

    // Start a minimal libp2p service with no peers. Broadcasts are no-ops, so
    // this behaves like the single-operator path while still using the canonical
    // `NetworkServiceHandle` API.
    let instance_key_pair = K256Ecdsa::generate_with_seed(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("key generation failed: {e}")))?;
    let local_key = libp2p::identity::Keypair::generate_ed25519();

    let net_config = NetConfig::<K256Ecdsa> {
        network_name: "distributed-training-http".to_string(),
        instance_id: "1.0.0".to_string(),
        instance_key_pair,
        local_key,
        listen_addr: "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        target_peer_count: 0,
        bootstrap_peers: vec![],
        enable_mdns: false,
        enable_kademlia: false,
        using_evm_address_for_handshake_verification: false,
    };

    let (_allowed_keys_tx, allowed_keys_rx) = crossbeam_channel::unbounded();
    let net_service = NetworkService::new(net_config, AllowedKeys::default(), allowed_keys_rx)
        .map_err(|e| blueprint_sdk::Error::Other(format!("networking init failed: {e}")))?;
    let handle = net_service.start();

    let coord = Arc::new(TrainingCoordinator::new(config.clone(), handle));
    register_coordinator(coord.clone());

    let notifier = Arc::new(blueprint_webhooks::notifier::JobNotifier::new(
        blueprint_webhooks::notifier::NotifierConfig {
            signing_secret: std::env::var("WEBHOOK_SIGNING_SECRET").unwrap_or_default(),
            ..Default::default()
        },
    ));

    let backend = TrainingAppBackend {
        config: config.clone(),
        coordinator: coord,
        notifier,
    };

    let state = AppState::from_config(
        &tangle,
        &config.server,
        &billing,
        config.server.max_concurrent_requests,
        backend,
    )
    .map_err(|e| blueprint_sdk::Error::Other(format!("failed to build AppState: {e}")))?;

    let _handle = server::start(state)
        .await
        .map_err(|e| blueprint_sdk::Error::Other(format!("server start failed: {e}")))?;

    tracing::info!("HTTP-only training operator running; press Ctrl-C to stop");
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");

    Ok(())
}
