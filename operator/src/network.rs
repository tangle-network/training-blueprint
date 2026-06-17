//! Training protocol message types exchanged over `blueprint-networking` gossip.
//!
//! The actual send/receive logic lives in `TrainingCoordinator`, which uses
//! `NetworkServiceHandle` directly — the canonical Blueprint SDK consumer pattern.
//! All training gossip is multiplexed through [`GossipEnvelope`] variants on the
//! blueprint protocol topic that the networking service subscribes to at startup.

use serde::{Deserialize, Serialize};

use crate::demo::SparseUpdate;

/// Messages exchanged over the coordination gossip channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    JoinJob {
        job_id: u64,
        peer_id: String,
        gpu_count: u32,
        vram_mib: u32,
    },
    LeaveJob {
        job_id: u64,
        peer_id: String,
    },
    SyncReady {
        job_id: u64,
        peer_id: String,
        step: u64,
    },
    CheckpointReady {
        job_id: u64,
        peer_id: String,
        step: u64,
        hash: [u8; 32],
    },
}

/// Unified gossip envelope carried over the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipEnvelope {
    Momentum(Vec<SparseUpdate>),
    Coordination(CoordinationMessage),
}
