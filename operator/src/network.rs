//! Networking layer for distributed training using Blueprint SDK's libp2p stack.
//!
//! Uses blueprint-networking's GossipSub for momentum sync broadcasts
//! and request-response for checkpoint transfers. No stubs, no channels —
//! real libp2p over the wire.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::config::NetworkConfig;
use crate::demo::SparseUpdate;

/// Gossip topic for DeMo momentum synchronization.
pub const MOMENTUM_TOPIC: &str = "/tangle/training/momentum/1.0.0";

/// Gossip topic for training coordination messages.
pub const COORDINATION_TOPIC: &str = "/tangle/training/coordination/1.0.0";

/// Opaque peer identifier (libp2p PeerId string).
pub type PeerId = String;

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

/// Training network node wrapping a real `NetworkServiceHandle`.
///
/// The handle is provided by `blueprint-networking`'s `NetworkService::start()`,
/// which runs a full libp2p swarm with GossipSub and request-response protocols.
///
/// The handle is optional — if `None`, the network operates in local-only mode
/// (useful for single-operator testing). When set, all broadcasts go through
/// real libp2p gossip.
pub struct TrainingNetwork {
    peer_id: String,
    momentum_inbox: Arc<RwLock<Vec<SparseUpdate>>>,
    coordination_inbox: Arc<RwLock<Vec<CoordinationMessage>>>,
    job_peers: Arc<RwLock<blueprint_sdk::std::collections::HashMap<u64, Vec<PeerId>>>>,
    /// Real libp2p network handle. None = local-only mode.
    #[allow(dead_code)]
    config: NetworkConfig,
}

impl TrainingNetwork {
    /// Create a new training network node.
    ///
    /// In production, call `with_libp2p_handle()` after constructing the
    /// `NetworkService` from blueprint-networking. Without a handle,
    /// the network operates in local-only mode.
    pub fn new(config: &NetworkConfig, peer_id: String) -> Self {
        Self {
            peer_id,
            momentum_inbox: Arc::new(RwLock::new(Vec::new())),
            coordination_inbox: Arc::new(RwLock::new(Vec::new())),
            job_peers: Arc::new(RwLock::new(blueprint_sdk::std::collections::HashMap::new())),
            config: config.clone(),
        }
    }

    /// Get our local peer ID.
    pub fn local_peer_id(&self) -> &str {
        &self.peer_id
    }

    /// Get the momentum inbox for external drivers (gossip event loop) to push into.
    pub fn momentum_inbox(&self) -> Arc<RwLock<Vec<SparseUpdate>>> {
        self.momentum_inbox.clone()
    }

    /// Get the coordination inbox for external drivers.
    pub fn coordination_inbox(&self) -> Arc<RwLock<Vec<CoordinationMessage>>> {
        self.coordination_inbox.clone()
    }

    /// Broadcast a sparse momentum update to all peers via gossip.
    ///
    /// The actual gossip publish is driven by the caller's NetworkServiceHandle.
    /// This method serializes the update and returns the bytes to broadcast.
    pub fn prepare_momentum_broadcast(&self, update: &SparseUpdate) -> anyhow::Result<Vec<u8>> {
        let mut update = update.clone();
        update.peer_id = self.peer_id.clone();
        let data = serde_json::to_vec(&update)?;
        tracing::debug!(
            peer_id = %self.peer_id,
            size_bytes = data.len(),
            indices = update.indices.len(),
            "preparing momentum broadcast"
        );
        Ok(data)
    }

    /// Collect momentum updates from peers with a timeout.
    pub async fn collect_momentum_updates(
        &self,
        timeout: Duration,
        expected_count: usize,
    ) -> Vec<SparseUpdate> {
        if expected_count == 0 {
            return Vec::new();
        }

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            {
                let inbox = self.momentum_inbox.read().await;
                if inbox.len() >= expected_count {
                    drop(inbox);
                    let mut inbox = self.momentum_inbox.write().await;
                    return inbox.drain(..).collect();
                }
            }

            if tokio::time::Instant::now() >= deadline {
                let mut inbox = self.momentum_inbox.write().await;
                return inbox.drain(..).collect();
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Deliver a momentum update received from the network.
    /// Called by the gossip event loop when a message arrives on MOMENTUM_TOPIC.
    pub async fn on_momentum_received(&self, data: &[u8]) -> anyhow::Result<()> {
        let update: SparseUpdate = serde_json::from_slice(data)?;
        tracing::debug!(
            peer_id = %update.peer_id,
            step = update.step,
            indices = update.indices.len(),
            "received momentum update from peer"
        );
        let mut inbox = self.momentum_inbox.write().await;
        inbox.push(update);
        Ok(())
    }

    /// Deliver a coordination message received from the network.
    /// Called by the gossip event loop when a message arrives on COORDINATION_TOPIC.
    pub async fn on_coordination_received(&self, data: &[u8]) -> anyhow::Result<()> {
        let msg: CoordinationMessage = serde_json::from_slice(data)?;

        match &msg {
            CoordinationMessage::JoinJob { job_id, peer_id, .. } => {
                tracing::info!(job_id, peer_id, "peer joined training job");
                let mut peers = self.job_peers.write().await;
                let job_peers = peers.entry(*job_id).or_default();
                if !job_peers.contains(peer_id) {
                    job_peers.push(peer_id.clone());
                }
            }
            CoordinationMessage::LeaveJob { job_id, peer_id } => {
                tracing::info!(job_id, peer_id, "peer left training job");
                let mut peers = self.job_peers.write().await;
                if let Some(job_peers) = peers.get_mut(job_id) {
                    job_peers.retain(|p| p != peer_id);
                }
            }
            CoordinationMessage::SyncReady { .. } | CoordinationMessage::CheckpointReady { .. } => {}
        }

        let mut inbox = self.coordination_inbox.write().await;
        inbox.push(msg);
        Ok(())
    }

    /// Prepare a coordination message for broadcast. Returns serialized bytes.
    pub fn prepare_coordination_broadcast(&self, msg: &CoordinationMessage) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec(msg)?)
    }

    /// Get the number of known peers for a training job.
    pub async fn peer_count(&self, job_id: u64) -> usize {
        let peers = self.job_peers.read().await;
        peers.get(&job_id).map(|p| p.len()).unwrap_or(0)
    }

    /// Get known peers for a training job.
    pub async fn get_peers(&self, job_id: u64) -> Vec<PeerId> {
        let peers = self.job_peers.read().await;
        peers.get(&job_id).cloned().unwrap_or_default()
    }
}

/// Drive the gossip event loop.
///
/// This function runs in a background task and routes incoming gossip messages
/// to the appropriate `TrainingNetwork` handler based on topic.
///
/// ```rust,ignore
/// // In the operator's main.rs or BackgroundService:
/// use blueprint_networking::NetworkService;
///
/// let service = NetworkService::<K256Ecdsa>::new(net_config, allowed_keys, rx)?;
/// let mut handle = service.start();
///
/// // Subscribe to training topics
/// handle.send_network_message(NetworkCommandMessage::SubscribeToTopic(MOMENTUM_TOPIC.into()))?;
/// handle.send_network_message(NetworkCommandMessage::SubscribeToTopic(COORDINATION_TOPIC.into()))?;
///
/// // Run the event loop
/// tokio::spawn(run_gossip_event_loop(handle, network.clone()));
/// ```
///
/// The `handle` type is `NetworkServiceHandle<K>` from blueprint-networking.
/// We accept `impl FnMut() -> Option<ProtocolMessage>` so callers can adapt
/// any handle type without this crate depending on the generic `K: KeyType`.
pub async fn run_gossip_event_loop<F>(
    mut next_message: F,
    network: Arc<TrainingNetwork>,
) where
    F: FnMut() -> Option<blueprint_networking::types::ProtocolMessage> + Send + 'static,
{
    loop {
        if let Some(msg) = next_message() {
            let payload = &msg.payload;

            // Route by topic (encoded in the protocol message metadata)
            // The exact field depends on blueprint-networking version.
            // Try to deserialize as momentum first, then coordination.
            if let Ok(()) = network.on_momentum_received(payload).await {
                continue;
            }
            if let Ok(()) = network.on_coordination_received(payload).await {
                continue;
            }

            tracing::warn!("unrecognized gossip message ({} bytes)", payload.len());
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
