pub mod checkpoint;
pub mod config;
pub mod coordinator;
pub mod demo;
pub mod eval_gate;
pub mod network;
pub mod qos;
pub mod server;
pub mod training;
pub mod verification;

use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use tangle_inference_core::AppState;

use crate::config::OperatorConfig;
use crate::coordinator::TrainingCoordinator;
use crate::server::TrainingAppBackend;
use blueprint_crypto::k256::K256Ecdsa;
use blueprint_networking::service_handle::NetworkServiceHandle;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct TrainingJobRequest {
        uint64 jobId;
        string baseModel;
        string datasetUrl;
        string method;
        uint32 totalEpochs;
        uint64 syncIntervalSteps;
        uint64 maxSteps;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct TrainingJobResult {
        uint64 jobId;
        bytes32 finalCheckpointHash;
        uint64 totalSteps;
        uint32 finalEpoch;
        // Held-out evaluation gate. Improvement and its CI lower bound are
        // fixed-point scaled by 1e4 (basis points) and signed (a regression is
        // negative). `heldOutCertified` is the off-chain gate's verdict: true only
        // when the bootstrap lower bound cleared the protocol margin. The legitimate
        // result submitter records this per operator on-chain via the authenticated
        // `updateContribution`/`recordCertification` path, and `distributePayment`
        // pays ZERO to any operator whose recorded contribution is not certified.
        bool heldOutCertified;
        int64 improvementBps;
        int64 ciLowerBoundBps;
        uint32 heldOutExamples;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct CheckpointRequest {
        uint64 jobId;
        bytes32 checkpointHash;
        uint32 epoch;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct CheckpointResult {
        bool accepted;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeaveRequest {
        uint64 jobId;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeaveResult {
        bool acknowledged;
    }
}

// --- Job IDs ---

pub const TRAINING_JOB: u8 = 0;
pub const CHECKPOINT_JOB: u8 = 1;
pub const LEAVE_JOB: u8 = 2;

// --- Shared coordinator ---

static COORDINATOR: blueprint_sdk::std::sync::OnceLock<Arc<TrainingCoordinator>> =
    blueprint_sdk::std::sync::OnceLock::new();

// `RunnerError` is the SDK's job-handler error type and is intentionally large;
// boxing it here would diverge from every other handler signature in the crate.
#[allow(clippy::result_large_err)]
fn get_coordinator() -> Result<&'static Arc<TrainingCoordinator>, RunnerError> {
    COORDINATOR
        .get()
        .ok_or_else(|| RunnerError::Other("TrainingCoordinator not initialized".into()))
}

pub fn register_coordinator(coord: Arc<TrainingCoordinator>) {
    let _ = COORDINATOR.set(coord);
}

// --- Router ---

pub fn router() -> Router {
    Router::new()
        .route(
            TRAINING_JOB,
            handle_training_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            CHECKPOINT_JOB,
            handle_checkpoint_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            LEAVE_JOB,
            handle_leave_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
}

// --- Job handlers ---

#[debug_job]
pub async fn handle_training_job(
    TangleArg(request): TangleArg<TrainingJobRequest>,
) -> Result<TangleResult<TrainingJobResult>, RunnerError> {
    let coord = get_coordinator()?;

    let result = coord
        .start_or_join_job(
            request.jobId,
            &request.baseModel,
            &request.datasetUrl,
            &request.method,
            request.totalEpochs,
            request.syncIntervalSteps,
            request.maxSteps,
        )
        .await
        .map_err(|e| RunnerError::Other(format!("training job failed: {e}").into()))?;

    // Scale the held-out improvement and its CI lower bound to signed basis points
    // (1e4) for the integer-only on-chain ABI. Saturating cast keeps absurd values
    // from wrapping; certification itself is driven by the bool, not the magnitude.
    let to_bps = |v: f64| -> i64 { (v * 10_000.0).round() as i64 };

    Ok(TangleResult(TrainingJobResult {
        jobId: request.jobId,
        finalCheckpointHash: alloy::primitives::FixedBytes(result.checkpoint_hash),
        totalSteps: result.total_steps,
        finalEpoch: result.final_epoch,
        heldOutCertified: result.certificate.certified,
        improvementBps: to_bps(result.certificate.mean_improvement),
        ciLowerBoundBps: to_bps(result.certificate.ci_lower_bound),
        heldOutExamples: result.certificate.n_examples as u32,
    }))
}

#[debug_job]
pub async fn handle_checkpoint_job(
    TangleArg(request): TangleArg<CheckpointRequest>,
) -> Result<TangleResult<CheckpointResult>, RunnerError> {
    let coord = get_coordinator()?;

    coord
        .submit_checkpoint(request.jobId, request.checkpointHash.into(), request.epoch)
        .await
        .map_err(|e| RunnerError::Other(format!("checkpoint submit failed: {e}").into()))?;

    Ok(TangleResult(CheckpointResult { accepted: true }))
}

#[debug_job]
pub async fn handle_leave_job(
    TangleArg(request): TangleArg<LeaveRequest>,
) -> Result<TangleResult<LeaveResult>, RunnerError> {
    let coord = get_coordinator()?;

    coord
        .handle_leave(request.jobId)
        .await
        .map_err(|e| RunnerError::Other(format!("leave failed: {e}").into()))?;

    Ok(TangleResult(LeaveResult { acknowledged: true }))
}

// --- Background service: training coordinator + HTTP server ---

#[derive(Clone)]
pub struct TrainingServer {
    pub config: Arc<OperatorConfig>,
    pub network: NetworkServiceHandle<K256Ecdsa>,
}

impl BackgroundService for TrainingServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();
        let network = self.network.clone();

        tokio::spawn(async move {
            // Initialize training coordinator
            let coord = Arc::new(TrainingCoordinator::new(config.clone(), network));

            register_coordinator(coord.clone());

            // Process coordination messages (JoinJob / LeaveJob) from peers.
            let coord_inbox = coord.clone();
            tokio::spawn(async move {
                coord_inbox.run_coordination_inbox().await;
            });

            // Build core AppState with billing support
            let notifier = Arc::new(blueprint_webhooks::notifier::JobNotifier::new(
                blueprint_webhooks::notifier::NotifierConfig {
                    signing_secret: std::env::var("WEBHOOK_SIGNING_SECRET")
                        .unwrap_or_else(|_| String::new()),
                    ..Default::default()
                },
            ));

            let backend = TrainingAppBackend {
                config: config.clone(),
                coordinator: coord.clone(),
                notifier,
            };

            let state = match AppState::from_config(
                &config.tangle,
                &config.server,
                &config.billing,
                config.server.max_concurrent_requests,
                backend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(format!("{e}").into())));
                    return;
                }
            };

            match server::start(state).await {
                Ok(_handle) => {
                    tracing::info!("Training HTTP server started");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // Keep alive until shutdown
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("received shutdown signal");
        });

        Ok(rx)
    }
}
