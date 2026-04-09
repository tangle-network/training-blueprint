//! HTTP API server for the distributed training operator.
//!
//! Endpoints:
//! - POST /v1/training/jobs          -- create or join a training job
//! - GET  /v1/training/jobs/:id      -- get job status
//! - POST /v1/training/jobs/:id/leave -- gracefully leave a job
//! - GET  /v1/training/jobs/:id/checkpoint -- download latest checkpoint
//! - GET  /health                    -- health check
//! - GET  /health/gpu                -- GPU health (from core)
//! - GET  /metrics                   -- Prometheus metrics (from core)

use blueprint_sdk::std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router as HttpRouter,
};
use blueprint_webhooks::notifier::{JobEvent, JobNotifier, JobStatus as WebhookJobStatus};
use tokio_stream::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, gpu_health_handler, metrics_handler,
    settle_billing,
};
use tangle_inference_core::AppState;

use crate::config::OperatorConfig;
use crate::coordinator::TrainingCoordinator;

/// Minimum billing cost for read-only / admin endpoints (status, checkpoint).
/// Requires valid SpendAuth but is essentially free.
const MIN_ADMIN_COST: u64 = 1000;

/// Backend state attached to AppState.
pub struct TrainingAppBackend {
    pub config: Arc<OperatorConfig>,
    pub coordinator: Arc<TrainingCoordinator>,
    pub notifier: Arc<JobNotifier>,
}

/// Start the HTTP server. Returns a JoinHandle.
pub async fn start(state: AppState) -> anyhow::Result<JoinHandle<()>> {
    let bind_addr = format!("{}:{}", state.server_config.host, state.server_config.port);

    let app = HttpRouter::new()
        .route("/v1/training/jobs", post(create_job))
        .route("/v1/training/jobs/{id}", get(get_job))
        .route("/v1/training/jobs/{id}/leave", post(leave_job))
        .route("/v1/training/jobs/{id}/checkpoint", get(get_checkpoint))
        .route("/v1/jobs/{job_id}/events", get(sse_handler))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    tracing::info!(addr = %bind_addr, "HTTP server listening");

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request/Response types ---

#[derive(Debug, Deserialize)]
struct CreateJobRequest {
    job_id: u64,
    base_model: String,
    dataset_url: String,
    method: String,
    total_epochs: u32,
    #[serde(default = "default_sync_interval")]
    sync_interval_steps: u64,
    /// Optional webhook URL for job status notifications.
    webhook_url: Option<String>,
}

fn default_sync_interval() -> u64 {
    500
}

#[derive(Debug, Serialize)]
struct CreateJobResponse {
    job_id: u64,
    status: String,
    message: String,
    /// Bearer token for connecting to the SSE events endpoint.
    sse_token: String,
}

// --- Helpers ---

fn backend_from(state: &AppState) -> &TrainingAppBackend {
    state
        .backend::<TrainingAppBackend>()
        .expect("TrainingAppBackend (checked in lib.rs)")
}

// --- Handlers ---

async fn create_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    let backend = backend_from(&state);

    // Concurrency gate
    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp.into_response(),
    };

    // Billing gate — estimate cost from total_epochs as a rough proxy for GPU-hours.
    // Each epoch is ~1 GPU-hour on average. Actual cost settled on completion.
    let estimated_gpu_hours = req.total_epochs as u64;
    let estimated_cost = estimated_gpu_hours * backend.config.training.price_per_gpu_hour;
    let (spend_auth, preauth) =
        match billing_gate(&state, &headers, None, estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp.into_response(),
        };

    let start_time = std::time::Instant::now();
    let notifier = backend.notifier.clone();
    let job_id_str = req.job_id.to_string();
    let webhook_url = req.webhook_url.clone();

    // Register job with notifier to get SSE auth token
    let sse_token = notifier.register_job(&job_id_str).await;

    // Notify: job is now processing
    let _ = notifier
        .notify(
            &job_id_str,
            JobEvent {
                status: WebhookJobStatus::Processing,
                ..Default::default()
            },
            webhook_url.as_deref(),
        )
        .await;

    match backend
        .coordinator
        .start_or_join_job(
            req.job_id,
            &req.base_model,
            &req.dataset_url,
            &req.method,
            req.total_epochs,
            req.sync_interval_steps,
        )
        .await
    {
        Ok(result) => {
            // Settle billing based on actual compute time
            let compute_secs = start_time.elapsed().as_secs();
            let actual_gpu_hours = (compute_secs + 3599) / 3600; // round up
            let actual_cost = actual_gpu_hours * backend.config.training.price_per_gpu_hour;
            if let Some(ref auth) = spend_auth {
                if let Err(e) = settle_billing(
                    &state.billing,
                    auth,
                    preauth.unwrap_or(0),
                    actual_cost,
                )
                .await
                {
                    tracing::error!(error = %e, "billing settlement failed");
                }
            }

            // Notify: job completed
            let _ = notifier
                .notify(
                    &job_id_str,
                    JobEvent {
                        status: WebhookJobStatus::Completed,
                        result: Some(serde_json::json!({
                            "checkpoint_hash": hex::encode(result.checkpoint_hash),
                            "total_steps": result.total_steps,
                            "final_epoch": result.final_epoch,
                        })),
                        ..Default::default()
                    },
                    webhook_url.as_deref(),
                )
                .await;

            (
                StatusCode::OK,
                Json(CreateJobResponse {
                    job_id: req.job_id,
                    status: "running".to_string(),
                    message: format!(
                        "Training job started. Checkpoint: {}",
                        hex::encode(result.checkpoint_hash)
                    ),
                    sse_token,
                }),
            )
                .into_response()
        }
        Err(e) => {
            // Notify: job failed
            let _ = notifier
                .notify(
                    &job_id_str,
                    JobEvent {
                        status: WebhookJobStatus::Failed,
                        error: Some(e.to_string()),
                        ..Default::default()
                    },
                    webhook_url.as_deref(),
                )
                .await;

            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "internal_error",
                "job_failed",
            )
            .into_response()
        }
    }
}

async fn get_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    let backend = backend_from(&state);

    // Billing gate — minimal cost for status checks
    if let Err(resp) = billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        return resp.into_response();
    }

    match backend.coordinator.get_job_status(id).await {
        Some(status) => (StatusCode::OK, Json(status)).into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("job {id} not found"),
            "not_found",
            "job_not_found",
        )
        .into_response(),
    }
}

async fn leave_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    let backend = backend_from(&state);

    // Billing gate — minimal cost for leave requests
    if let Err(resp) = billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        return resp.into_response();
    }

    match backend.coordinator.handle_leave(id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "acknowledged": true })),
        )
            .into_response(),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "internal_error",
            "leave_failed",
        )
        .into_response(),
    }
}

async fn get_checkpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    // Billing gate — minimal cost for checkpoint downloads
    if let Err(resp) = billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        return resp.into_response();
    }

    match crate::checkpoint::latest_checkpoint(id).await {
        Ok(Some(path)) => match tokio::fs::read(&path).await {
            Ok(data) => (
                StatusCode::OK,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "application/octet-stream",
                )],
                data,
            )
                .into_response(),
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to read checkpoint: {e}"),
                "internal_error",
                "checkpoint_read_failed",
            )
            .into_response(),
        },
        Ok(None) => error_response(
            StatusCode::NOT_FOUND,
            format!("no checkpoint found for job {id}"),
            "not_found",
            "checkpoint_not_found",
        )
        .into_response(),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "internal_error",
            "checkpoint_error",
        )
        .into_response(),
    }
}

async fn sse_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let backend = backend_from(&state);

    // Validate bearer token
    let token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match token {
        Some(t) if backend.notifier.validate_job_token(&job_id, t).await => {}
        _ => {
            return error_response(
                StatusCode::UNAUTHORIZED,
                "invalid or missing SSE bearer token".into(),
                "auth_error",
                "invalid_sse_token",
            )
            .into_response();
        }
    }

    let rx = match backend.notifier.subscribe(&job_id).await {
        Some(r) => r,
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "subscriber capacity exceeded".into(),
                "capacity_error",
                "too_many_subscribers",
            )
            .into_response();
        }
    };
    let stream = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let data = serde_json::to_string(&event)
                .unwrap_or_else(|_| r#"{"error":"serialize"}"#.to_string());
            let sse_event = Event::default()
                .event(event.status.to_string())
                .data(data);
            Some(Ok::<_, std::convert::Infallible>(sse_event))
        }
        Err(_) => {
            tracing::warn!("SSE subscriber lagged or channel closed");
            None
        }
    });

    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

async fn health_check() -> impl IntoResponse {
    let gpu_status = match tangle_inference_core::detect_gpus().await {
        Ok(gpus) => serde_json::json!({
            "available": true,
            "count": gpus.len(),
            "gpus": gpus,
        }),
        Err(_) => serde_json::json!({
            "available": false,
            "count": 0,
        }),
    };

    Json(serde_json::json!({
        "status": "ok",
        "service": "distributed-training-operator",
        "gpu": gpu_status,
    }))
}
