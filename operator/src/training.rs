//! Training backend interface — abstracts the local training engine.
//!
//! Operators can use any training framework (unsloth, TRL, torchtune) as long
//! as it exposes the HTTP API implemented by `training-adapter/main.py`.
//! The backend runs as a separate process and communicates via HTTP.

use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;
use crate::demo::SparseUpdate;
use crate::eval_gate::HeldOutLosses;

/// Result returned by a training-step batch.
#[derive(Debug, Clone, Copy)]
pub struct TrainStepResult {
    pub steps_completed: u64,
    pub total_steps: u64,
    pub loss: f32,
}

/// Local training backend that calls the Python training server over HTTP.
///
/// Python adapter endpoints:
/// - `POST /v1/train/init` — load model + dataset + hyperparameters
/// - `POST /v1/train/step` — run N steps and return loss
/// - `POST /v1/train/save_state` — return raw torch-serialized state bytes
/// - `POST /v1/train/load_state` — restore from raw state bytes
/// - `POST /eval_held_out` — per-example held-out losses for base vs candidate
pub struct LocalTrainingBackend {
    endpoint: String,
    client: reqwest::Client,
}

impl LocalTrainingBackend {
    pub fn new(endpoint: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "The public adapter accepts the training protocol fields without an extra wrapper type."
    )]
    pub async fn init_model(
        &self,
        base_model: &str,
        method: &str,
        dataset_url: &str,
        total_epochs: u32,
        max_steps: u64,
        sync_interval_steps: u64,
        shard: Option<(u64, u64)>,
        cfg: &TrainingConfig,
    ) -> anyhow::Result<()> {
        let mut body = serde_json::json!({
            "base_model": base_model,
            "method": method,
            "dataset_url": dataset_url,
            "dataset_format": cfg.dataset_format,
            "max_seq_length": cfg.max_seq_length,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target_modules": cfg.lora_target_modules,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_epochs": total_epochs,
            "max_steps": max_steps,
            "warmup_steps": cfg.warmup_steps,
            "lr_scheduler": cfg.lr_scheduler,
            "weight_decay": cfg.weight_decay,
            "load_in_4bit": cfg.load_in_4bit,
            "sync_interval_steps": sync_interval_steps,
            "demo_top_k_ratio": cfg.demo_top_k_ratio,
        });
        if let Some((start, end)) = shard {
            body["shard_start"] = start.into();
            body["shard_end"] = end.into();
        }

        let resp = self
            .client
            .post(format!("{}/v1/train/init", self.endpoint))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("init_model failed: {body}");
        }

        Ok(())
    }

    pub async fn train_steps(&self, num_steps: u64) -> anyhow::Result<TrainStepResult> {
        let resp = self
            .client
            .post(format!("{}/v1/train/step", self.endpoint))
            .json(&serde_json::json!({ "num_steps": num_steps }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("train_steps failed: {body}");
        }

        let body: StepResponse = resp.json().await?;
        Ok(TrainStepResult {
            steps_completed: body.steps_completed,
            total_steps: body.total_steps,
            loss: body.loss as f32,
        })
    }

    pub async fn save_state(&self) -> anyhow::Result<Vec<u8>> {
        let resp = self
            .client
            .post(format!("{}/v1/train/save_state", self.endpoint))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("save_state failed: {body}");
        }

        Ok(resp.bytes().await?.to_vec())
    }

    pub async fn load_state(&self, checkpoint: &[u8]) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!("{}/v1/train/load_state", self.endpoint))
            .body(checkpoint.to_vec())
            .header("content-type", "application/octet-stream")
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("load_state failed: {body}");
        }

        Ok(())
    }

    /// Run a local training burst and produce a compressed DeMo momentum update
    /// relative to the baseline established at the previous sync round.
    pub async fn demo_step(&self, num_steps: u64) -> anyhow::Result<(Vec<SparseUpdate>, f32)> {
        let resp = self
            .client
            .post(format!("{}/v1/train/demo_step", self.endpoint))
            .json(&serde_json::json!({ "num_steps": num_steps }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("demo_step failed: {body}");
        }

        let body: DemoStepResponse = resp.json().await?;
        Ok((body.updates, body.loss as f32))
    }

    /// Apply aggregated peer DeMo momentum updates to the local optimizer.
    pub async fn demo_apply_sync(&self, peer_updates: &[Vec<SparseUpdate>]) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!("{}/v1/train/demo_apply_sync", self.endpoint))
            .json(&serde_json::json!({ "peer_updates": peer_updates }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("demo_apply_sync failed: {body}");
        }

        Ok(())
    }

    /// Score the base model and the current (candidate) checkpoint on the private
    /// held-out validation split, returning per-example losses for both. The
    /// protocol's eval gate consumes these to certify that the candidate actually
    /// improved on data no operator trained on.
    pub async fn held_out_losses(&self, base_model: &str) -> anyhow::Result<HeldOutLosses> {
        let resp = self
            .client
            .post(format!("{}/eval_held_out", self.endpoint))
            .json(&serde_json::json!({ "base_model": base_model }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("held_out_losses failed: {body}");
        }

        let body: HeldOutLossResponse = resp.json().await?;
        Ok(HeldOutLosses::new(body.base, body.candidate))
    }
}

/// Create a training backend from config.
pub fn create_backend(config: &TrainingConfig) -> anyhow::Result<LocalTrainingBackend> {
    Ok(LocalTrainingBackend::new(&config.endpoint))
}

// --- Wire types for training backend HTTP API ---

#[derive(Debug, Serialize, Deserialize)]
struct StepResponse {
    steps_completed: u64,
    total_steps: u64,
    loss: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DemoStepResponse {
    updates: Vec<SparseUpdate>,
    loss: f64,
    #[allow(dead_code)]
    steps_completed: u64,
    #[allow(dead_code)]
    total_steps: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct HeldOutLossResponse {
    /// Per-example loss of the base/reference model on the held-out split.
    base: Vec<f64>,
    /// Per-example loss of the candidate checkpoint on the same held-out examples.
    candidate: Vec<f64>,
}
