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
                .separator("__")
                .try_parsing(true),
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
