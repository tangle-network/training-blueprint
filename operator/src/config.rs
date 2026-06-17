//! Operator configuration — training-specific settings plus shared core types.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// Training backend configuration (training-specific).
    pub training: TrainingConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Networking (libp2p) configuration (training-specific).
    pub network: NetworkConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

/// Training backend configuration — the only truly training-specific config
/// section. Everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Training backend HTTP endpoint (e.g. "http://localhost:5000").
    #[serde(default = "default_training_endpoint")]
    pub endpoint: String,

    /// Price per GPU-hour in tsUSD base units (6 decimals).
    pub price_per_gpu_hour: u64,

    /// DeMo sync interval in local training steps.
    #[serde(default = "default_sync_interval")]
    pub sync_interval_steps: u64,

    /// Maximum operators per training job.
    #[serde(default = "default_max_operators")]
    pub max_operators: u32,

    /// Supported training methods.
    #[serde(default = "default_supported_methods")]
    pub supported_methods: Vec<String>,

    /// Network bandwidth in Mbps (for DeMo efficiency estimation).
    #[serde(default = "default_bandwidth")]
    pub network_bandwidth_mbps: u64,

    // --- Hyperparameters passed to the training adapter ---
    /// Dataset format expected by the adapter ("chat" or "text").
    #[serde(default = "default_dataset_format")]
    pub dataset_format: String,

    /// Maximum sequence length.
    #[serde(default = "default_max_seq_length")]
    pub max_seq_length: u32,

    /// Per-device train batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,

    /// Gradient accumulation steps.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: u32,

    /// LoRA rank.
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,

    /// LoRA alpha.
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,

    /// LoRA dropout.
    #[serde(default = "default_lora_dropout")]
    pub lora_dropout: f64,

    /// LoRA target modules.
    #[serde(default = "default_lora_target_modules")]
    pub lora_target_modules: Vec<String>,

    /// Learning rate.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Warmup steps.
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u32,

    /// LR scheduler type.
    #[serde(default = "default_lr_scheduler")]
    pub lr_scheduler: String,

    /// Weight decay.
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Whether to load the model in 4-bit (requires GPU).
    #[serde(default = "default_load_in_4bit")]
    pub load_in_4bit: bool,

    /// Top-k sparsification ratio used for DeMo momentum sync (0.001 = 0.1%).
    #[serde(default = "default_demo_top_k_ratio")]
    pub demo_top_k_ratio: f64,

    /// Minimum held-out loss reduction (base − candidate) that the bootstrap CI lower
    /// bound must exceed for the checkpoint to be certified. Default 0.02; smoke tests
    /// can lower it, production operators should keep it strict.
    #[serde(default = "default_held_out_min_improvement")]
    pub held_out_min_improvement: f64,
}

/// Networking (libp2p) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address for libp2p (multiaddr format).
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Bootstrap peer addresses for initial discovery.
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
}

// --- Defaults ---

fn default_training_endpoint() -> String {
    blueprint_sdk::std::env::var("TRAINING_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:5000".to_string())
}

fn default_sync_interval() -> u64 {
    500
}

fn default_max_operators() -> u32 {
    256
}

fn default_supported_methods() -> Vec<String> {
    vec![
        "sft".to_string(),
        "dpo".to_string(),
        "grpo".to_string(),
        "pretrain".to_string(),
    ]
}

fn default_bandwidth() -> u64 {
    1000 // 1 Gbps default
}

fn default_dataset_format() -> String {
    "text".to_string()
}

fn default_max_seq_length() -> u32 {
    64
}

fn default_batch_size() -> u32 {
    2
}

fn default_gradient_accumulation_steps() -> u32 {
    1
}

fn default_lora_r() -> u32 {
    4
}

fn default_lora_alpha() -> u32 {
    8
}

fn default_lora_dropout() -> f64 {
    0.05
}

fn default_lora_target_modules() -> Vec<String> {
    vec!["q_proj".to_string(), "v_proj".to_string()]
}

fn default_learning_rate() -> f64 {
    2e-4
}

fn default_warmup_steps() -> u32 {
    0
}

fn default_lr_scheduler() -> String {
    "cosine".to_string()
}

fn default_weight_decay() -> f64 {
    0.01
}

fn default_load_in_4bit() -> bool {
    false
}

fn default_demo_top_k_ratio() -> f64 {
    0.001
}

fn default_held_out_min_improvement() -> f64 {
    0.02
}

fn default_listen_addr() -> String {
    "/ip4/0.0.0.0/tcp/9000".to_string()
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: TRAIN_OP_ (e.g. TRAIN_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("TRAIN_OP")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true)
                .list_separator(",")
                .with_list_parse_key("training.lora_target_modules")
                .with_list_parse_key("training.supported_methods"),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "training": {
                "endpoint": "http://localhost:5000",
                "price_per_gpu_hour": 1000000,
                "sync_interval_steps": 500,
                "max_operators": 64,
                "network_bandwidth_mbps": 10000
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "network": {
                "listen_addr": "/ip4/0.0.0.0/tcp/9000",
                "bootstrap_peers": []
            },
            "billing": {
                "max_spend_per_request": 1000000,
                "min_credit_balance": 1000
            },
            "gpu": {
                "expected_gpu_count": 4,
                "min_vram_mib": 81920,
                "gpu_model": "NVIDIA H100"
            }
        }"#
    }

    #[test]
    fn test_deserialize_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.gpu.expected_gpu_count, 4);
        assert_eq!(cfg.gpu.min_vram_mib, 81920);
        assert_eq!(cfg.training.sync_interval_steps, 500);
        assert_eq!(cfg.training.price_per_gpu_hour, 1000000);
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn test_defaults_applied() {
        let json = r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xdead",
                "shielded_credits": "0x02",
                "blueprint_id": 1
            },
            "training": { "price_per_gpu_hour": 1000000 },
            "server": {},
            "network": {},
            "billing": { "max_spend_per_request": 1000000, "min_credit_balance": 1000 },
            "gpu": { "expected_gpu_count": 1, "min_vram_mib": 24000 }
        }"#;
        let cfg: OperatorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.training.sync_interval_steps, 500);
        assert_eq!(cfg.training.max_operators, 256);
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(cfg.network.listen_addr, "/ip4/0.0.0.0/tcp/9000");
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }
}
