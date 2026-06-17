//! Full-lifecycle devnet E2E for the training blueprint.
//!
//! This test drives the real Rust operator + real Python training adapter through
//! a local anvil devnet using `blueprint-anvil-testing-utils`. It submits a real
//! on-chain training job, waits for the operator to train `gpt2` for a few steps
//! on CPU, save a real checkpoint, run held-out eval, and return a certified
//! `TrainingJobResult`.
//!
//! ## Running it
//!
//! Ignored by default because it needs Docker + the seeded foundry anvil image
//! and the Python ML stack (torch, transformers, datasets, trl). Run explicitly:
//!
//! ```bash
//! cargo test -p distributed-training --test lifecycle -- --ignored --nocapture
//! ```

use alloy_primitives::Bytes;
use alloy_sol_types::SolValue;
use anyhow::Result;
use blueprint_anvil_testing_utils::{missing_tnt_core_artifacts, BlueprintHarness};
use blueprint_crypto::k256::K256Ecdsa;
use blueprint_crypto::KeyType;
use blueprint_networking::service::{AllowedKeys, NetworkConfig as NetConfig, NetworkService};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

use distributed_training::{
    config::OperatorConfig, coordinator::TrainingCoordinator, register_coordinator, router,
    TrainingJobRequest, TrainingJobResult, TRAINING_JOB,
};

const JOB_RESULT_TIMEOUT: Duration = Duration::from_secs(300);
const ADAPTER_HEALTH_TIMEOUT: Duration = Duration::from_secs(60);

struct AdapterGuard {
    child: Child,
    port: u16,
}

impl AdapterGuard {
    async fn start(held_out_path: &str) -> Result<Self> {
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

        // Wait for the adapter to be healthy.
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

async fn spawn_harness(adapter_uri: String) -> Result<Option<BlueprintHarness>> {
    let vars: Vec<(String, String)> = vec![
        ("TRAIN_OP_TRAINING__ENDPOINT".into(), adapter_uri),
        ("TRAIN_OP_TRAINING__PRICE_PER_GPU_HOUR".into(), "0".into()),
        ("TRAIN_OP_TRAINING__SYNC_INTERVAL_STEPS".into(), "2".into()),
        ("TRAIN_OP_TRAINING__MAX_OPERATORS".into(), "1".into()),
        (
            "TRAIN_OP_TRAINING__NETWORK_BANDWIDTH_MBPS".into(),
            "1000".into(),
        ),
        ("TRAIN_OP_TRAINING__DATASET_FORMAT".into(), "text".into()),
        ("TRAIN_OP_TRAINING__MAX_SEQ_LENGTH".into(), "64".into()),
        ("TRAIN_OP_TRAINING__BATCH_SIZE".into(), "2".into()),
        (
            "TRAIN_OP_TRAINING__GRADIENT_ACCUMULATION_STEPS".into(),
            "1".into(),
        ),
        ("TRAIN_OP_TRAINING__LORA_R".into(), "4".into()),
        ("TRAIN_OP_TRAINING__LORA_ALPHA".into(), "8".into()),
        (
            "TRAIN_OP_TRAINING__LORA_TARGET_MODULES".into(),
            "c_attn".into(),
        ),
        (
            "TRAIN_OP_TRAINING__HELD_OUT_MIN_IMPROVEMENT".into(),
            "0.001".into(),
        ),
        ("TRAIN_OP_TRAINING__LOAD_IN_4BIT".into(), "false".into()),
        ("TRAIN_OP_SERVER__HOST".into(), "127.0.0.1".into()),
        ("TRAIN_OP_SERVER__PORT".into(), "0".into()),
        (
            "TRAIN_OP_NETWORK__LISTEN_ADDR".into(),
            "/ip4/127.0.0.1/tcp/0".into(),
        ),
        (
            "TRAIN_OP_BILLING__PAYMENT_RAILS__SHIELDED".into(),
            "false".into(),
        ),
        (
            "TRAIN_OP_BILLING__PAYMENT_RAILS__DIRECT".into(),
            "false".into(),
        ),
        ("TRAIN_OP_BILLING__BILLING_REQUIRED".into(), "false".into()),
        ("TRAIN_OP_BILLING__MAX_SPEND_PER_REQUEST".into(), "0".into()),
        ("TRAIN_OP_BILLING__MIN_CREDIT_BALANCE".into(), "0".into()),
        ("TRAIN_OP_GPU__EXPECTED_GPU_COUNT".into(), "0".into()),
        ("TRAIN_OP_GPU__MIN_VRAM_MIB".into(), "0".into()),
        (
            "TRAIN_OP_TANGLE__RPC_URL".into(),
            "http://localhost:8545".into(),
        ),
        ("TRAIN_OP_TANGLE__CHAIN_ID".into(), "31337".into()),
        (
            "TRAIN_OP_TANGLE__OPERATOR_KEY".into(),
            "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
        ),
        (
            "TRAIN_OP_TANGLE__SHIELDED_CREDITS".into(),
            "0x0000000000000000000000000000000000000002".into(),
        ),
        ("TRAIN_OP_TANGLE__BLUEPRINT_ID".into(), "1".into()),
    ];

    match BlueprintHarness::builder(router())
        .poll_interval(Duration::from_millis(200))
        .with_env_vars(vars)
        .with_pre_spawn_hook(|_| async {
            let config = Arc::new(OperatorConfig::load(None)?);

            // Minimal local libp2p service: no bootstrap, no discovery. Broadcasts
            // are no-ops, so this exercises the single-operator lifecycle path.
            let instance_key_pair = K256Ecdsa::generate_with_seed(None)?;
            let local_key = libp2p::identity::Keypair::generate_ed25519();
            let net_config = NetConfig::<K256Ecdsa> {
                network_name: "distributed-training-lifecycle".to_string(),
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
            let net_service =
                NetworkService::new(net_config, AllowedKeys::default(), allowed_keys_rx)?;
            let handle = net_service.start();

            let coord = Arc::new(TrainingCoordinator::new(config, handle));
            register_coordinator(coord);
            Ok(())
        })
        .spawn()
        .await
    {
        Ok(h) => Ok(Some(h)),
        Err(e) if missing_tnt_core_artifacts(&e) => {
            eprintln!("[skip] seeded tnt-core artifacts unavailable: {e}");
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Docker + seeded anvil image + Python ML deps; run with --ignored"]
async fn training_job_lifecycle_with_real_adapter() -> Result<()> {
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

    let adapter = AdapterGuard::start(&held_out_path).await?;
    eprintln!("[setup] training adapter up at {}", adapter.uri());

    let Some(harness) = spawn_harness(adapter.uri()).await? else {
        return Ok(());
    };

    eprintln!(
        "[setup] harness up: service_id={} blueprint_id={}",
        harness.service_id(),
        harness.blueprint_id(),
    );

    let request = TrainingJobRequest {
        jobId: 1,
        baseModel: "gpt2".into(),
        datasetUrl: format!("file://{train_path}"),
        method: "sft".into(),
        totalEpochs: 1,
        syncIntervalSteps: 2,
        maxSteps: 5,
    };

    let sub = harness
        .submit_job(TRAINING_JOB, Bytes::from(request.abi_encode()))
        .await?;

    let output = harness
        .wait_for_job_result_with_deadline(sub, JOB_RESULT_TIMEOUT)
        .await?;

    let result = TrainingJobResult::abi_decode(&output)?;

    eprintln!(
        "[result] certified={} improvementBps={} totalSteps={} finalEpoch={}",
        result.heldOutCertified, result.improvementBps, result.totalSteps, result.finalEpoch
    );

    assert!(result.heldOutCertified, "expected held-out certification");
    assert!(result.improvementBps > 0, "expected positive improvement");
    assert!(
        result.finalCheckpointHash.as_slice() != [0u8; 32],
        "expected non-zero checkpoint hash"
    );
    assert!(result.totalSteps >= 5, "expected at least 5 training steps");

    harness.shutdown().await;
    Ok(())
}
