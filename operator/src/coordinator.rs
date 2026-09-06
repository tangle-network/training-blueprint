//! Multi-operator training coordination.
//!
//! Manages the distributed training loop: data shard assignment, peer join/leave,
//! DeMo sync barriers, and on-chain checkpoint submission.

use blueprint_sdk::std::collections::HashMap;
use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::checkpoint;
use crate::config::OperatorConfig;
use crate::demo::{self, SparseUpdate};
use crate::eval_gate::{self, EvalCertificate, EvalGateConfig, HeldOutLosses};
use crate::network::{CoordinationMessage, GossipEnvelope};
use blueprint_crypto::k256::K256Ecdsa;
use blueprint_networking::service_handle::NetworkServiceHandle;
use blueprint_networking::types::MessageRouting;

/// Result of a completed or joined training job.
pub struct JobResult {
    pub checkpoint_hash: [u8; 32],
    pub total_steps: u64,
    pub final_epoch: u32,
    /// Held-out evaluation certificate for the final checkpoint. The protocol
    /// pays for certified improvement on the private held-out split, not for the
    /// checkpoint hash on its own.
    pub certificate: EvalCertificate,
}

/// Active training job state.
pub struct TrainingJob {
    pub job_id: u64,
    pub base_model: String,
    pub dataset_url: String,
    pub method: String,
    pub total_epochs: u32,
    pub max_steps: u64,
    pub current_epoch: u32,
    pub sync_interval_steps: u64,
    pub steps_completed: u64,
    pub current_loss: f32,
    pub operators: Vec<String>,
    pub shard_assignments: HashMap<String, DataShard>,
    pub latest_checkpoint_hash: [u8; 32],
    pub latest_checkpoint_step: u64,
    pub completed: bool,
}

/// Data shard assigned to an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    /// Start index in the dataset.
    pub start: u64,
    /// End index (exclusive) in the dataset.
    pub end: u64,
    /// Shard index for identification.
    pub shard_id: u32,
}

/// Training coordinator managing multi-operator distributed training.
pub struct TrainingCoordinator {
    config: Arc<OperatorConfig>,
    jobs: RwLock<HashMap<u64, TrainingJob>>,
    job_peers: Arc<RwLock<HashMap<u64, Vec<String>>>>,
    network: NetworkServiceHandle<K256Ecdsa>,
    our_peer_id: String,
    momentum_inbox: Arc<RwLock<Vec<SparseUpdate>>>,
    coordination_inbox: Arc<RwLock<Vec<CoordinationMessage>>>,
    _drain_handle: tokio::task::JoinHandle<()>,
}

impl TrainingCoordinator {
    /// Return this operator's peer id.
    pub fn our_peer_id(&self) -> &str {
        &self.our_peer_id
    }

    pub fn new(config: Arc<OperatorConfig>, network: NetworkServiceHandle<K256Ecdsa>) -> Self {
        let our_peer_id = network.local_peer_id.to_string();

        // Subscribe to the training gossip topics.
        // The network service already subscribes to the blueprint protocol topic on
        // startup; all training gossip is multiplexed through `GossipEnvelope` variants.

        let momentum_inbox = Arc::new(RwLock::new(Vec::new()));
        let coordination_inbox = Arc::new(RwLock::new(Vec::new()));

        let mut drain_handle = network.clone();
        let m_inbox = Arc::clone(&momentum_inbox);
        let c_inbox = Arc::clone(&coordination_inbox);
        let self_peer_id = our_peer_id.clone();
        let drain_task = tokio::spawn(async move {
            loop {
                let Some(msg) = drain_handle.next_protocol_message() else {
                    // `next_protocol_message` is a non-blocking `try_recv`; an empty
                    // channel is not a shutdown signal, so keep polling.
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                };

                let sender = msg.routing.sender.to_string();
                if sender == self_peer_id {
                    continue;
                }

                let Ok(envelope) = serde_json::from_slice::<GossipEnvelope>(&msg.payload) else {
                    tracing::warn!(
                        from = %sender,
                        bytes = msg.payload.len(),
                        "discarded non-training gossip message"
                    );
                    continue;
                };

                match envelope {
                    GossipEnvelope::Momentum(updates) => {
                        m_inbox.write().await.extend(updates);
                    }
                    GossipEnvelope::Coordination(msg) => {
                        c_inbox.write().await.push(msg);
                    }
                }

                // `next_protocol_message` is a non-blocking try_recv; avoid a busy loop.
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        Self {
            config,
            jobs: RwLock::new(HashMap::new()),
            job_peers: Arc::new(RwLock::new(HashMap::new())),
            network,
            our_peer_id,
            momentum_inbox,
            coordination_inbox,
            _drain_handle: drain_task,
        }
    }

    /// Get the number of known peers for a training job.
    pub async fn peer_count(&self, job_id: u64) -> usize {
        let peers = self.job_peers.read().await;
        peers.get(&job_id).map(|p| p.len()).unwrap_or(0)
    }

    /// Get known peers for a training job.
    pub async fn get_peers(&self, job_id: u64) -> Vec<String> {
        let peers = self.job_peers.read().await;
        peers.get(&job_id).cloned().unwrap_or_default()
    }

    /// Start or join a distributed training job.
    #[allow(
        clippy::too_many_arguments,
        reason = "The public job entrypoint preserves the independent chain and HTTP request fields."
    )]
    pub async fn start_or_join_job(
        &self,
        job_id: u64,
        base_model: &str,
        dataset_url: &str,
        method: &str,
        total_epochs: u32,
        sync_interval_steps: u64,
        max_steps: u64,
    ) -> anyhow::Result<JobResult> {
        // Discover existing peers for this job
        let peers = self.get_peers(job_id).await;

        let mut jobs = self.jobs.write().await;

        if jobs.contains_key(&job_id) {
            // Already participating — return current status
            let job = jobs.get(&job_id).unwrap();
            return Ok(JobResult {
                checkpoint_hash: job.latest_checkpoint_hash,
                total_steps: job.steps_completed,
                final_epoch: job.current_epoch,
                // Joining an in-flight job does not re-run the held-out gate; the
                // certificate is minted once, when the job's owner finalizes it.
                certificate: eval_gate::uncertified("job already in progress"),
            });
        }

        // Create job state
        let mut job = TrainingJob {
            job_id,
            base_model: base_model.to_string(),
            dataset_url: dataset_url.to_string(),
            method: method.to_string(),
            total_epochs,
            max_steps,
            current_epoch: 0,
            sync_interval_steps,
            steps_completed: 0,
            current_loss: f32::INFINITY,
            operators: vec![self.our_peer_id.clone()],
            shard_assignments: HashMap::new(),
            latest_checkpoint_hash: [0u8; 32],
            latest_checkpoint_step: 0,
            completed: false,
        };

        // Add existing peers
        for peer in &peers {
            let peer_str = peer.to_string();
            if !job.operators.contains(&peer_str) {
                job.operators.push(peer_str);
            }
        }

        // Assign data shards across all operators
        let dataset_size = 1_000_000; // placeholder — real implementation queries dataset metadata
        self.assign_data_shards(&mut job, dataset_size);

        let our_shard = self.our_shard(&job);

        jobs.insert(job_id, job);
        drop(jobs);

        // Announce our presence to any peers already in the job.
        let _ = self
            .broadcast_coordination(&CoordinationMessage::JoinJob {
                job_id,
                peer_id: self.our_peer_id.clone(),
                gpu_count: self.config.gpu.expected_gpu_count,
                vram_mib: self.config.gpu.min_vram_mib,
            })
            .await;

        // Run training loop
        let result = self
            .run_training_loop(job_id, sync_interval_steps, our_shard)
            .await?;

        Ok(result)
    }

    /// Return this operator's data shard, if one was assigned.
    fn our_shard(&self, job: &TrainingJob) -> Option<(u64, u64)> {
        job.shard_assignments
            .get(&self.our_peer_id)
            .map(|s| (s.start, s.end))
    }

    /// Assign data shards evenly across operators.
    pub fn assign_data_shards(&self, job: &mut TrainingJob, dataset_size: u64) {
        let n_operators = job.operators.len() as u64;
        if n_operators == 0 {
            return;
        }

        let shard_size = dataset_size / n_operators;
        job.shard_assignments.clear();

        for (i, operator) in job.operators.iter().enumerate() {
            let start = i as u64 * shard_size;
            let end = if i as u64 == n_operators - 1 {
                dataset_size
            } else {
                start + shard_size
            };

            job.shard_assignments.insert(
                operator.clone(),
                DataShard {
                    start,
                    end,
                    shard_id: i as u32,
                },
            );
        }
    }

    /// Handle a new peer joining an active training job.
    pub async fn handle_peer_join(&self, job_id: u64, peer: &str) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;

        if job.operators.contains(&peer.to_string()) {
            return Ok(()); // already in job
        }

        job.operators.push(peer.to_string());

        // Redistribute shards: take half of the largest shard for the new peer
        let largest_operator = job
            .shard_assignments
            .iter()
            .max_by_key(|(_, shard)| shard.end - shard.start)
            .map(|(op, _)| op.clone());

        if let Some(largest_op) = largest_operator {
            if let Some(shard) = job.shard_assignments.get(&largest_op).cloned() {
                let midpoint = shard.start + (shard.end - shard.start) / 2;

                // Shrink the largest shard
                job.shard_assignments.insert(
                    largest_op,
                    DataShard {
                        start: shard.start,
                        end: midpoint,
                        shard_id: shard.shard_id,
                    },
                );

                // Give the new peer the second half
                let new_shard_id = job.shard_assignments.len() as u32;
                job.shard_assignments.insert(
                    peer.to_string(),
                    DataShard {
                        start: midpoint,
                        end: shard.end,
                        shard_id: new_shard_id,
                    },
                );
            }
        }

        tracing::info!(
            job_id,
            peer,
            operators = job.operators.len(),
            "peer joined training job"
        );

        Ok(())
    }

    /// Handle a peer leaving an active training job.
    pub async fn handle_peer_leave(&self, job_id: u64, peer: &str) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;

        // Remove the peer
        job.operators.retain(|p| p != peer);

        // Absorb their shard into remaining operators
        if let Some(orphan_shard) = job.shard_assignments.remove(peer) {
            let remaining: Vec<String> = job.shard_assignments.keys().cloned().collect();
            if !remaining.is_empty() {
                let chunk_size = (orphan_shard.end - orphan_shard.start) / remaining.len() as u64;
                let mut offset = orphan_shard.start;

                for (i, op) in remaining.iter().enumerate() {
                    if let Some(shard) = job.shard_assignments.get_mut(op) {
                        // Extend each remaining operator's shard
                        let extra_end = if i == remaining.len() - 1 {
                            orphan_shard.end
                        } else {
                            offset + chunk_size
                        };
                        // For simplicity, we extend by recording the additional range.
                        // A real implementation would merge ranges.
                        shard.end += extra_end - offset;
                        offset = extra_end;
                    }
                }
            }
        }

        tracing::info!(
            job_id,
            peer,
            remaining = job.operators.len(),
            "peer left training job, shards redistributed"
        );

        Ok(())
    }

    /// Handle the current operator leaving a job (on-chain LEAVE_JOB).
    pub async fn handle_leave(&self, job_id: u64) -> anyhow::Result<()> {
        self.handle_peer_leave(job_id, &self.our_peer_id.clone())
            .await
    }

    /// DeMo sync barrier: coordinate momentum synchronization across operators.
    ///
    /// NOTE: This is wired for real multi-operator sync in Phase 2. In the
    /// current single-operator path `expected_peers == 0`, so this barrier is
    /// a no-op and no fake updates are generated.
    #[allow(dead_code)]
    pub async fn sync_barrier(
        &self,
        job_id: u64,
        local_updates: Vec<SparseUpdate>,
    ) -> anyhow::Result<Vec<ndarray::Array2<f32>>> {
        // Broadcast our sparse updates
        self.broadcast_momentum_updates(&local_updates).await?;

        // Collect updates from peers with timeout
        let timeout = Duration::from_secs(30);
        let jobs = self.jobs.read().await;
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;
        let expected_peers = job.operators.len().saturating_sub(1);
        drop(jobs);

        let peer_updates = self.collect_momentum_updates(timeout, expected_peers).await;

        // Aggregate: combine all peer updates with our own
        let mut all_updates = local_updates;
        all_updates.extend(peer_updates);

        // Group by parameter index (using step as a proxy — all have same step)
        // For now, treat all updates as same parameter group and aggregate
        let aggregated = demo::aggregate_updates(&all_updates);

        // Apply inverse DCT to get spatial-domain momentum update
        let result = demo::idct_2d(&aggregated);

        Ok(vec![result])
    }

    /// Submit a checkpoint hash on-chain.
    pub async fn submit_checkpoint(
        &self,
        job_id: u64,
        hash: [u8; 32],
        epoch: u32,
    ) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.latest_checkpoint_hash = hash;
            job.current_epoch = epoch;
        }

        tracing::info!(
            job_id,
            epoch,
            hash = hex::encode(hash),
            "checkpoint submitted"
        );

        Ok(())
    }

    /// Get the status of a training job.
    pub async fn get_job_status(&self, job_id: u64) -> Option<JobStatus> {
        let jobs = self.jobs.read().await;
        let job = jobs.get(&job_id)?;

        Some(JobStatus {
            job_id,
            base_model: job.base_model.clone(),
            method: job.method.clone(),
            current_epoch: job.current_epoch,
            total_epochs: job.total_epochs,
            steps_completed: job.steps_completed,
            current_loss: job.current_loss,
            operators: job.operators.len() as u32,
            completed: job.completed,
            latest_checkpoint_hash: hex::encode(job.latest_checkpoint_hash),
        })
    }

    /// Run the main training loop for a job.
    async fn run_training_loop(
        &self,
        job_id: u64,
        sync_interval: u64,
        our_shard: Option<(u64, u64)>,
    ) -> anyhow::Result<JobResult> {
        let (total_epochs, base_model, method, dataset_url, max_steps, expected_peers) = {
            let jobs = self.jobs.read().await;
            let job = jobs
                .get(&job_id)
                .ok_or_else(|| anyhow::anyhow!("job not found"))?;
            (
                job.total_epochs,
                job.base_model.clone(),
                job.method.clone(),
                job.dataset_url.clone(),
                job.max_steps,
                job.operators.len().saturating_sub(1),
            )
        };

        // Initialize training backend with the real Python adapter.
        let backend = crate::training::create_backend(&self.config.training)?;
        backend
            .init_model(
                &base_model,
                &method,
                &dataset_url,
                total_epochs,
                max_steps,
                sync_interval,
                our_shard,
                &self.config.training,
            )
            .await?;

        if expected_peers == 0 {
            // Single-operator fast path: no fake DeMo sync, just train and checkpoint.
            self.run_single_operator_loop(job_id, &backend, total_epochs, max_steps, sync_interval)
                .await?;
        } else {
            // Multi-operator path: real DeMo momentum sync over libp2p gossip.
            self.run_demo_loop(
                job_id,
                &backend,
                total_epochs,
                max_steps,
                sync_interval,
                expected_peers,
            )
            .await?;
        }

        // Held-out evaluation gate: score the base model and the final checkpoint
        // on the private held-out split and certify the improvement before the
        // result goes on-chain.
        let certificate = self.certify_final_checkpoint(&base_model).await;

        // Mark completed
        let mut jobs = self.jobs.write().await;
        let job = jobs.get_mut(&job_id).unwrap();
        job.completed = true;

        Ok(JobResult {
            checkpoint_hash: job.latest_checkpoint_hash,
            total_steps: job.steps_completed,
            final_epoch: job.current_epoch,
            certificate,
        })
    }

    async fn run_single_operator_loop(
        &self,
        job_id: u64,
        backend: &crate::training::LocalTrainingBackend,
        total_epochs: u32,
        max_steps: u64,
        sync_interval: u64,
    ) -> anyhow::Result<()> {
        for epoch in 0..total_epochs {
            tracing::info!(job_id, epoch, "starting epoch (single-operator)");

            let steps_to_run = if max_steps > 0 {
                max_steps
            } else {
                sync_interval.max(1)
            };

            let result = backend.train_steps(steps_to_run).await?;

            let mut jobs = self.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.steps_completed = result.total_steps;
                job.current_loss = result.loss;
            }
            drop(jobs);

            self.save_epoch_checkpoint(job_id, epoch, result.loss)
                .await?;
        }
        Ok(())
    }

    async fn run_demo_loop(
        &self,
        job_id: u64,
        backend: &crate::training::LocalTrainingBackend,
        total_epochs: u32,
        max_steps: u64,
        sync_interval: u64,
        expected_peers: usize,
    ) -> anyhow::Result<()> {
        let steps_per_epoch = if max_steps > 0 {
            max_steps
        } else {
            sync_interval.max(1)
        };

        for epoch in 0..total_epochs {
            tracing::info!(job_id, epoch, "starting epoch (DeMo multi-operator)");

            let mut steps_this_epoch = 0u64;
            while steps_this_epoch < steps_per_epoch {
                let steps_to_run = sync_interval.min(steps_per_epoch - steps_this_epoch);

                let (local_updates, loss) = backend.demo_step(steps_to_run).await?;

                // Broadcast our compressed momentum update(s).
                self.broadcast_momentum_updates(&local_updates).await?;

                // Collect peer updates until we hear from all expected peers or time out.
                let timeout = Duration::from_secs(60);
                let collected = self.collect_momentum_updates(timeout, expected_peers).await;

                // Group collected updates by peer id. Our own updates are already
                // accounted for in `local_updates`.
                let mut peer_updates: Vec<Vec<SparseUpdate>> = vec![local_updates];
                let mut by_peer: std::collections::HashMap<String, Vec<SparseUpdate>> =
                    std::collections::HashMap::new();
                for update in collected {
                    if update.peer_id == self.our_peer_id || update.peer_id.is_empty() {
                        continue;
                    }
                    let peer_id = update.peer_id.clone();
                    by_peer.entry(peer_id).or_default().push(update);
                }
                peer_updates.extend(by_peer.into_values());

                // Apply the aggregated peer momentum update.
                backend.demo_apply_sync(&peer_updates).await?;

                steps_this_epoch += steps_to_run;

                let mut jobs = self.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.steps_completed += steps_to_run;
                    job.current_loss = loss;
                }
            }

            self.save_epoch_checkpoint(job_id, epoch, 0.0).await?;
        }
        Ok(())
    }

    async fn save_epoch_checkpoint(
        &self,
        job_id: u64,
        epoch: u32,
        loss: f32,
    ) -> anyhow::Result<()> {
        let backend = crate::training::create_backend(&self.config.training)?;
        let state_bytes = backend.save_state().await?;
        if state_bytes.is_empty() {
            anyhow::bail!("backend returned empty checkpoint state");
        }

        let ckpt_path = checkpoint::checkpoint_path(job_id, epoch as u64);
        checkpoint::save_checkpoint_file(&ckpt_path, &state_bytes).await?;
        let hash = checkpoint::hash_checkpoint(&ckpt_path)?;

        {
            let mut jobs = self.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.current_epoch = epoch + 1;
                job.latest_checkpoint_hash = hash;
                job.latest_checkpoint_step = epoch as u64;
                if !loss.is_nan() {
                    job.current_loss = loss;
                }
            }
        }

        self.submit_checkpoint(job_id, hash, epoch + 1).await?;
        Ok(())
    }

    /// Score the base model and the final checkpoint on the private held-out split
    /// and certify the improvement. Returns an uncertified result if the backend
    /// cannot supply held-out losses, so the chain fails closed (no proof, no pay).
    async fn certify_final_checkpoint(&self, base_model: &str) -> EvalCertificate {
        let backend = match crate::training::create_backend(&self.config.training) {
            Ok(b) => b,
            Err(e) => return eval_gate::uncertified(&format!("eval backend unavailable: {e}")),
        };

        match backend.held_out_losses(base_model).await {
            Ok(losses) => self.certify_losses(&losses),
            Err(e) => eval_gate::uncertified(&format!("held-out eval failed: {e}")),
        }
    }

    /// Run the held-out evaluation gate over already-collected per-example losses.
    /// Split out so it can be exercised deterministically in tests without a backend.
    pub fn certify_losses(&self, losses: &HeldOutLosses) -> EvalCertificate {
        let cfg = EvalGateConfig {
            min_improvement: self.config.training.held_out_min_improvement,
            ..EvalGateConfig::default()
        };
        eval_gate::certify(losses, &cfg)
    }

    /// Broadcast compressed momentum updates to all connected peers.
    pub async fn broadcast_momentum_updates(&self, updates: &[SparseUpdate]) -> anyhow::Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut updates: Vec<SparseUpdate> = updates.to_vec();
        for u in &mut updates {
            u.peer_id = self.our_peer_id.clone();
        }
        let payload = serde_json::to_vec(&GossipEnvelope::Momentum(updates))?;
        let routing = MessageRouting {
            message_id: 0,
            round_id: 0,
            sender: self.network.local_peer_id,
            recipient: None,
        };
        self.network
            .send(routing, payload)
            .map_err(|e| anyhow::anyhow!("broadcast momentum failed: {e}"))?;
        Ok(())
    }

    /// Broadcast a coordination message to all connected peers.
    pub async fn broadcast_coordination(&self, msg: &CoordinationMessage) -> anyhow::Result<()> {
        let payload = serde_json::to_vec(&GossipEnvelope::Coordination(msg.clone()))?;
        let routing = MessageRouting {
            message_id: 0,
            round_id: 0,
            sender: self.network.local_peer_id,
            recipient: None,
        };
        self.network
            .send(routing, payload)
            .map_err(|e| anyhow::anyhow!("broadcast coordination failed: {e}"))?;
        Ok(())
    }

    /// Wait until updates from at least `expected_peers` distinct non-self peers
    /// have arrived, then drain and return the collected updates.
    pub async fn collect_momentum_updates(
        &self,
        timeout: Duration,
        expected_peers: usize,
    ) -> Vec<SparseUpdate> {
        if expected_peers == 0 {
            return Vec::new();
        }

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let inbox = self.momentum_inbox.read().await;
                let mut by_peer: HashMap<String, Vec<SparseUpdate>> = HashMap::new();
                for update in inbox.iter() {
                    if update.peer_id.is_empty() || update.peer_id == self.our_peer_id {
                        continue;
                    }
                    by_peer
                        .entry(update.peer_id.clone())
                        .or_default()
                        .push(update.clone());
                }
                if by_peer.len() >= expected_peers {
                    let to_return: Vec<SparseUpdate> = by_peer.into_values().flatten().collect();
                    drop(inbox);
                    let mut inbox = self.momentum_inbox.write().await;
                    // Remove only the updates we are returning, keeping any late arrivals.
                    let returned_ids: std::collections::HashSet<(String, u64)> = to_return
                        .iter()
                        .map(|u| (u.peer_id.clone(), u.step))
                        .collect();
                    inbox.retain(|u| !returned_ids.contains(&(u.peer_id.clone(), u.step)));
                    return to_return;
                }
            }

            if tokio::time::Instant::now() >= deadline {
                let mut inbox = self.momentum_inbox.write().await;
                let mut by_peer: HashMap<String, Vec<SparseUpdate>> = HashMap::new();
                for update in inbox.drain(..) {
                    let peer_id = update.peer_id.clone();
                    if peer_id.is_empty() || peer_id == self.our_peer_id {
                        continue;
                    }
                    by_peer.entry(peer_id).or_default().push(update);
                }
                return by_peer.into_values().flatten().collect();
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Return the next unprocessed coordination message, if any arrives before
    /// the timeout.
    pub async fn next_coordination(&self, timeout: Duration) -> Option<CoordinationMessage> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let mut inbox = self.coordination_inbox.write().await;
                if !inbox.is_empty() {
                    return Some(inbox.remove(0));
                }
            }

            if tokio::time::Instant::now() >= deadline {
                return None;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Process incoming coordination messages (JoinJob / LeaveJob) from the
    /// network inbox. This should be spawned as a background task when running
    /// with a real libp2p handle.
    pub async fn run_coordination_inbox(&self) {
        loop {
            match self.next_coordination(Duration::from_secs(1)).await {
                Some(CoordinationMessage::JoinJob {
                    job_id, peer_id, ..
                }) => {
                    // Record in the local job peer registry.
                    {
                        let mut peers = self.job_peers.write().await;
                        let job_peers = peers.entry(job_id).or_default();
                        if !job_peers.contains(&peer_id) {
                            job_peers.push(peer_id.clone());
                        }
                    }
                    if let Err(e) = self.handle_peer_join(job_id, &peer_id).await {
                        tracing::warn!(job_id, peer_id, error = %e, "failed to handle peer join");
                    }
                }
                Some(CoordinationMessage::LeaveJob { job_id, peer_id }) => {
                    {
                        let mut peers = self.job_peers.write().await;
                        if let Some(job_peers) = peers.get_mut(&job_id) {
                            job_peers.retain(|p| p != &peer_id);
                        }
                    }
                    if let Err(e) = self.handle_peer_leave(job_id, &peer_id).await {
                        tracing::warn!(job_id, peer_id, error = %e, "failed to handle peer leave");
                    }
                }
                Some(CoordinationMessage::SyncReady { .. })
                | Some(CoordinationMessage::CheckpointReady { .. })
                | None => {}
            }
        }
    }
}

/// Public job status for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: u64,
    pub base_model: String,
    pub method: String,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub steps_completed: u64,
    pub current_loss: f32,
    pub operators: u32,
    pub completed: bool,
    pub latest_checkpoint_hash: String,
}
