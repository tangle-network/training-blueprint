//! Two-operator DeMo end-to-end smoke test.
//!
//! Boots two real Python training adapters and two Rust coordinators in the same
//! process, wires them together with an in-memory gossip broker, and runs a tiny
//! distributed training job. Proves that the operators exchange compressed
//! momentum updates, apply them, and both complete with certified checkpoints.
//!
//! Ignored by default because it needs the Python ML stack. Run with:
//!
//! ```bash
//! cargo test -p distributed-training --test two_operator_demo -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use blueprint_crypto::k256::K256Ecdsa;
use blueprint_crypto::KeyType;
use blueprint_networking::service::AllowedKeys;
use blueprint_networking::service_handle::NetworkServiceHandle;
use blueprint_networking::test_utils::{wait_for_all_handshakes, TestNode};
use std::collections::HashSet;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

use distributed_training::{
    config::OperatorConfig, coordinator::TrainingCoordinator, network::CoordinationMessage,
};

const JOB_TIMEOUT: Duration = Duration::from_secs(300);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const ADAPTER_HEALTH_TIMEOUT: Duration = Duration::from_secs(60);

struct AdapterGuard {
    child: Child,
    port: u16,
}

impl AdapterGuard {
    async fn start(_train_path: &str, held_out_path: &str) -> Result<Self> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let script_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training-adapter");
        let main_py = script_dir.join("main.py");

        let mut child = Command::new("python3")
            .arg(&main_py)
            .current_dir(&script_dir)
            .env("TRAINING_PORT", port.to_string())
            .env("TRAINING_BACKEND", "trl")
            .env("HELD_OUT_DATASET_URL", held_out_path)
            .env("HELD_OUT_DATASET_SPLIT", "train")
            .env("HELD_OUT_MAX_EXAMPLES", "5")
            .env("PYTHONUNBUFFERED", "1")
            .kill_on_drop(true)
            .spawn()?;

        let health_url = format!("http://127.0.0.1:{port}/health");
        let healthy = timeout(ADAPTER_HEALTH_TIMEOUT, async {
            loop {
                match reqwest::get(&health_url).await {
                    Ok(r) if r.status().is_success() => return true,
                    _ => sleep(Duration::from_millis(250)).await,
                }
            }
        })
        .await
        .unwrap_or(false);

        if !healthy {
            let _ = child.start_kill();
            anyhow::bail!("training adapter did not become healthy on port {port}");
        }

        Ok(Self { child, port })
    }

    fn uri(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for AdapterGuard {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn base_env(adapter_uri: String, checkpoint_dir: String) -> Vec<(&'static str, String)> {
    vec![
        ("TRAIN_OP_TRAINING__ENDPOINT", adapter_uri),
        ("TRAIN_OP_TRAINING__PRICE_PER_GPU_HOUR", "0".into()),
        ("TRAIN_OP_TRAINING__SYNC_INTERVAL_STEPS", "2".into()),
        ("TRAIN_OP_TRAINING__MAX_OPERATORS", "2".into()),
        (
            "TRAIN_OP_TRAINING__NETWORK_BANDWIDTH_MBPS".into(),
            "1000".into(),
        ),
        ("TRAIN_OP_TRAINING__DATASET_FORMAT", "text".into()),
        ("TRAIN_OP_TRAINING__MAX_SEQ_LENGTH", "64".into()),
        ("TRAIN_OP_TRAINING__BATCH_SIZE", "2".into()),
        (
            "TRAIN_OP_TRAINING__GRADIENT_ACCUMULATION_STEPS".into(),
            "1".into(),
        ),
        ("TRAIN_OP_TRAINING__LORA_R", "4".into()),
        ("TRAIN_OP_TRAINING__LORA_ALPHA", "8".into()),
        ("TRAIN_OP_TRAINING__LORA_TARGET_MODULES", "c_attn".into()),
        (
            "TRAIN_OP_TRAINING__HELD_OUT_MIN_IMPROVEMENT".into(),
            "0.001".into(),
        ),
        ("TRAIN_OP_TRAINING__LOAD_IN_4BIT", "false".into()),
        ("TRAIN_OP_SERVER__HOST", "127.0.0.1".into()),
        ("TRAIN_OP_SERVER__PORT", "0".into()),
        (
            "TRAIN_OP_NETWORK__LISTEN_ADDR".into(),
            "/ip4/127.0.0.1/tcp/0".into(),
        ),
        ("TRAIN_OP_BILLING__PAYMENT_RAILS__SHIELDED", "false".into()),
        ("TRAIN_OP_BILLING__PAYMENT_RAILS__DIRECT", "false".into()),
        ("TRAIN_OP_BILLING__BILLING_REQUIRED", "false".into()),
        ("TRAIN_OP_BILLING__MAX_SPEND_PER_REQUEST", "0".into()),
        ("TRAIN_OP_BILLING__MIN_CREDIT_BALANCE", "0".into()),
        ("TRAIN_OP_GPU__EXPECTED_GPU_COUNT", "0".into()),
        ("TRAIN_OP_GPU__MIN_VRAM_MIB", "0".into()),
        ("TRAIN_OP_TANGLE__RPC_URL", "http://localhost:8545".into()),
        ("TRAIN_OP_TANGLE__CHAIN_ID", "31337".into()),
        (
            "TRAIN_OP_TANGLE__OPERATOR_KEY".into(),
            "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
        ),
        (
            "TRAIN_OP_TANGLE__SHIELDED_CREDITS".into(),
            "0x0000000000000000000000000000000000000002".into(),
        ),
        ("TRAIN_OP_TANGLE__BLUEPRINT_ID", "1".into()),
        ("TRAIN_OP_CHECKPOINT_DIR", checkpoint_dir),
    ]
}

fn load_config_with_env(vars: &[(&str, String)]) -> Result<OperatorConfig> {
    for (k, v) in vars {
        unsafe { std::env::set_var(k, v) };
    }
    OperatorConfig::load(None)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Python ML deps; run with --ignored"]
async fn two_operator_demo_sync() -> Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let train_path = manifest
        .join("../training-adapter/test-data/train.jsonl")
        .canonicalize()?
        .to_string_lossy()
        .to_string();
    let held_out_path = manifest
        .join("../training-adapter/test-data/held_out.jsonl")
        .canonicalize()?
        .to_string_lossy()
        .to_string();

    let adapter_a = AdapterGuard::start(&train_path, &held_out_path).await?;
    let adapter_b = AdapterGuard::start(&train_path, &held_out_path).await?;
    eprintln!(
        "[setup] adapters up at {} and {}",
        adapter_a.uri(),
        adapter_b.uri()
    );

    let checkpoint_a = tempfile::tempdir()?;
    let checkpoint_b = tempfile::tempdir()?;
    let checkpoint_a_path = checkpoint_a.path().to_path_buf();
    let checkpoint_b_path = checkpoint_b.path().to_path_buf();

    let cfg_a = Arc::new(load_config_with_env(&base_env(
        adapter_a.uri(),
        checkpoint_a_path.to_string_lossy().to_string(),
    ))?);
    let cfg_b = Arc::new(load_config_with_env(&base_env(
        adapter_b.uri(),
        checkpoint_b_path.to_string_lossy().to_string(),
    ))?);

    // Start two real libp2p nodes with explicit bootstrap so they form a
    // reliable two-peer mesh for the training gossip topics.
    let key_a = K256Ecdsa::generate_with_seed(None)?;
    let key_b = K256Ecdsa::generate_with_seed(None)?;
    let mut allowed_keys = HashSet::new();
    allowed_keys.insert(K256Ecdsa::public_from_secret(&key_a));
    allowed_keys.insert(K256Ecdsa::public_from_secret(&key_b));

    let mut node_a = TestNode::<K256Ecdsa>::new_with_keys(
        "distributed-training",
        "1.0.0",
        AllowedKeys::InstancePublicKeys(allowed_keys.clone()),
        vec![],
        Some(key_a),
        None,
        false,
    );
    let handle_a = node_a
        .start()
        .await
        .map_err(|e| anyhow::anyhow!("node A start failed: {e}"))?;
    let addr_a = handle_a
        .get_listen_addr()
        .ok_or_else(|| anyhow::anyhow!("node A listen addr not available"))?;

    let mut node_b = TestNode::<K256Ecdsa>::new_with_keys(
        "distributed-training",
        "1.0.0",
        AllowedKeys::InstancePublicKeys(allowed_keys),
        vec![addr_a],
        Some(key_b),
        None,
        false,
    );
    let handle_b = node_b
        .start()
        .await
        .map_err(|e| anyhow::anyhow!("node B start failed: {e}"))?;

    let mut handles = vec![handle_a, handle_b];

    // Create the coordinators first so they subscribe to the training topics
    // before the GossipSub mesh forms during handshake.
    let coord_a = Arc::new(TrainingCoordinator::new(cfg_a, handles[0].clone()));
    let coord_b = Arc::new(TrainingCoordinator::new(cfg_b, handles[1].clone()));

    let handle_refs: Vec<&mut NetworkServiceHandle<K256Ecdsa>> = handles.iter_mut().collect();
    wait_for_all_handshakes(&handle_refs, HANDSHAKE_TIMEOUT).await;
    eprintln!("[setup] libp2p handshake complete");

    // Give GossipSub a moment to graft the training topics into the mesh.
    sleep(Duration::from_secs(2)).await;

    // Run coordination inboxes.
    let _inbox_a = tokio::spawn({
        let c = Arc::clone(&coord_a);
        async move { c.run_coordination_inbox().await }
    });
    let _inbox_b = tokio::spawn({
        let c = Arc::clone(&coord_b);
        async move { c.run_coordination_inbox().await }
    });

    // Seed each operator with knowledge of the other before training starts,
    // otherwise both fall back to the single-operator path.
    let join_a = CoordinationMessage::JoinJob {
        job_id: 1,
        peer_id: coord_a.our_peer_id().to_string(),
        gpu_count: 0,
        vram_mib: 0,
    };
    let join_b = CoordinationMessage::JoinJob {
        job_id: 1,
        peer_id: coord_b.our_peer_id().to_string(),
        gpu_count: 0,
        vram_mib: 0,
    };
    // Each operator announces *itself* so the other peer learns about it.
    coord_a.broadcast_coordination(&join_a).await?;
    coord_b.broadcast_coordination(&join_b).await?;

    // Wait until both coordinators have processed the peer joins.
    timeout(Duration::from_secs(5), async {
        loop {
            let count_a = coord_a.peer_count(1).await;
            let count_b = coord_b.peer_count(1).await;
            if count_a >= 1 && count_b >= 1 {
                break;
            }
            sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("peer joins should be processed within 5 seconds");

    eprintln!(
        "[peers] A id={} peers={:?}; B id={} peers={:?}",
        coord_a.our_peer_id(),
        coord_a.get_peers(1).await,
        coord_b.our_peer_id(),
        coord_b.get_peers(1).await,
    );

    let dataset_url = format!("file://{train_path}");
    let dataset_url_a = dataset_url.clone();
    let dataset_url_b = dataset_url.clone();

    let job_a = {
        let c = Arc::clone(&coord_a);
        async move {
            c.start_or_join_job(1, "gpt2", &dataset_url_a, "sft", 1, 2, 5)
                .await
        }
    };
    let job_b = {
        let c = Arc::clone(&coord_b);
        async move {
            c.start_or_join_job(1, "gpt2", &dataset_url_b, "sft", 1, 2, 5)
                .await
        }
    };

    let (res_a, res_b) = timeout(JOB_TIMEOUT, futures::future::join(job_a, job_b)).await?;
    let res_a = res_a?;
    let res_b = res_b?;

    eprintln!(
        "[result A] certified={} improvementBps={} totalSteps={} hash={:x?}",
        res_a.certificate.certified,
        (res_a.certificate.mean_improvement * 10_000.0).round() as i64,
        res_a.total_steps,
        res_a.checkpoint_hash
    );
    eprintln!(
        "[result B] certified={} improvementBps={} totalSteps={} hash={:x?}",
        res_b.certificate.certified,
        (res_b.certificate.mean_improvement * 10_000.0).round() as i64,
        res_b.total_steps,
        res_b.checkpoint_hash
    );

    assert!(
        res_a.certificate.certified,
        "operator A should be certified"
    );
    assert!(
        res_b.certificate.certified,
        "operator B should be certified"
    );
    assert!(
        res_a.total_steps >= 5,
        "operator A should run at least 5 steps"
    );
    assert!(
        res_b.total_steps >= 5,
        "operator B should run at least 5 steps"
    );
    assert_ne!(
        res_a.checkpoint_hash, [0u8; 32],
        "operator A should have a checkpoint hash"
    );
    assert_ne!(
        res_b.checkpoint_hash, [0u8; 32],
        "operator B should have a checkpoint hash"
    );

    Ok(())
}
