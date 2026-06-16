//! Held-out evaluation gate for certifying training contributions.
//!
//! The protocol must not pay for a checkpoint merely because it exists and hashes
//! correctly. A candidate checkpoint is only certified as a *contribution* if it
//! measurably improves model quality on a **private held-out validation split** —
//! data that no operator trains on — with statistical confidence.
//!
//! This mirrors held-out-gated competition markets: a candidate must beat the
//! baseline on a private split with a confidence-interval lower bound above a
//! minimum margin. Self-reported gradient norms (see [`crate::verification`]) prove
//! an operator *did work*; this gate proves the work *helped*. Reward attaches to
//! the certified improvement, not to GPU-minutes or a checkpoint hash.
//!
//! The estimator is a deterministic, seeded **bootstrap** of the per-example loss
//! reduction (base loss − candidate loss). Determinism matters: any party holding
//! the same held-out losses and the same seed reconstructs the identical
//! certificate, so the gate is itself re-verifiable rather than a trusted claim.

use serde::{Deserialize, Serialize};

/// Minimum held-out improvement (mean loss reduction) that the bootstrap lower
/// bound must clear for a checkpoint to be certified. Matches the held-out-gated
/// market bar: a candidate must beat the baseline by a real, non-trivial margin.
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 0.02;

/// Default bootstrap resample count. 1000 gives a stable 2.5% quantile while
/// staying cheap enough to run on every epoch boundary.
pub const DEFAULT_BOOTSTRAP_SAMPLES: usize = 1000;

/// Fixed seed so the certificate is reproducible by any verifier.
pub const BOOTSTRAP_SEED: u64 = 0x5eed_d157_1b07_ea51;

/// Per-example losses of the base model and the candidate checkpoint on the same
/// held-out examples. Element `i` of both vectors must refer to held-out example
/// `i`; the gate pairs them to compute per-example improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeldOutLosses {
    /// Loss of the base/reference model per held-out example (lower is better).
    pub base: Vec<f64>,
    /// Loss of the candidate checkpoint per held-out example (lower is better).
    pub candidate: Vec<f64>,
}

impl HeldOutLosses {
    pub fn new(base: Vec<f64>, candidate: Vec<f64>) -> Self {
        Self { base, candidate }
    }

    /// Per-example loss reduction (base − candidate). Positive = candidate is better.
    fn improvements(&self) -> Vec<f64> {
        self.base
            .iter()
            .zip(self.candidate.iter())
            .map(|(b, c)| b - c)
            .collect()
    }
}

/// Tunable parameters for the gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalGateConfig {
    /// Bootstrap lower bound must exceed this to certify.
    pub min_improvement: f64,
    /// Number of bootstrap resamples.
    pub bootstrap_samples: usize,
    /// Confidence level for the one-sided lower bound (e.g. 0.95).
    pub confidence: f64,
}

impl Default for EvalGateConfig {
    fn default() -> Self {
        Self {
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
            bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
            confidence: 0.95,
        }
    }
}

/// Certificate produced by the gate. Its `certified` verdict is carried into the
/// on-chain job result and recorded per operator through the authenticated
/// `updateContribution`/`recordCertification` path; `DistributedTrainingBSM.distributePayment`
/// then pays ZERO to any operator whose recorded contribution is not certified, so
/// payout follows certified held-out improvement rather than a bare checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalCertificate {
    /// Whether the candidate cleared the held-out improvement bar.
    pub certified: bool,
    /// Mean per-example loss reduction (base − candidate) over the held-out split.
    pub mean_improvement: f64,
    /// One-sided lower bound of the bootstrap confidence interval on the mean
    /// improvement. This is the number the gate decides on.
    pub ci_lower_bound: f64,
    /// The margin the lower bound had to clear.
    pub min_improvement: f64,
    /// Number of held-out examples scored.
    pub n_examples: usize,
    /// Human-readable reason when not certified.
    pub reason: Option<String>,
}

/// Certify a candidate checkpoint against the base model on a held-out split.
///
/// Returns a deterministic [`EvalCertificate`]: the candidate is certified iff the
/// bootstrap lower bound on the mean held-out loss reduction exceeds
/// `config.min_improvement`. A lower bound (not the point estimate) is required so
/// that noise on a small or lucky split cannot mint a false certificate.
pub fn certify(losses: &HeldOutLosses, config: &EvalGateConfig) -> EvalCertificate {
    let n = losses.base.len();

    if n == 0 || losses.candidate.is_empty() {
        return EvalCertificate {
            certified: false,
            mean_improvement: 0.0,
            ci_lower_bound: 0.0,
            min_improvement: config.min_improvement,
            n_examples: 0,
            reason: Some("empty held-out split".to_string()),
        };
    }

    if losses.base.len() != losses.candidate.len() {
        return EvalCertificate {
            certified: false,
            mean_improvement: 0.0,
            ci_lower_bound: 0.0,
            min_improvement: config.min_improvement,
            n_examples: 0,
            reason: Some(format!(
                "base/candidate length mismatch: {} vs {}",
                losses.base.len(),
                losses.candidate.len()
            )),
        };
    }

    if losses
        .base
        .iter()
        .chain(losses.candidate.iter())
        .any(|v| !v.is_finite())
    {
        return EvalCertificate {
            certified: false,
            mean_improvement: 0.0,
            ci_lower_bound: 0.0,
            min_improvement: config.min_improvement,
            n_examples: n,
            reason: Some("non-finite loss value in held-out split".to_string()),
        };
    }

    let improvements = losses.improvements();
    let mean_improvement = mean(&improvements);
    let ci_lower_bound = bootstrap_lower_bound(&improvements, config);

    let certified = ci_lower_bound > config.min_improvement;
    let reason = if certified {
        None
    } else {
        Some(format!(
            "held-out CI lower bound {ci_lower_bound:.4} did not exceed margin {:.4}",
            config.min_improvement
        ))
    };

    EvalCertificate {
        certified,
        mean_improvement,
        ci_lower_bound,
        min_improvement: config.min_improvement,
        n_examples: n,
        reason,
    }
}

/// Build an uncertified certificate (fail-closed) with a reason. Used when the
/// held-out split cannot be scored: recorded on-chain as not-certified, so
/// `distributePayment` pays ZERO for that operator rather than paying on an
/// unverified claim.
pub fn uncertified(reason: &str) -> EvalCertificate {
    EvalCertificate {
        certified: false,
        mean_improvement: 0.0,
        ci_lower_bound: 0.0,
        min_improvement: DEFAULT_MIN_IMPROVEMENT,
        n_examples: 0,
        reason: Some(reason.to_string()),
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// One-sided lower bound of the bootstrap distribution of the mean improvement.
///
/// Resamples the per-example improvements with replacement `config.bootstrap_samples`
/// times, takes the mean of each resample, and returns the `(1 - confidence)`
/// quantile of those means. Uses a deterministic SplitMix64 PRNG seeded with a
/// fixed constant so the bound is reproducible by any verifier.
fn bootstrap_lower_bound(improvements: &[f64], config: &EvalGateConfig) -> f64 {
    let n = improvements.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        // Single example: no resampling signal, fall back to the point value.
        return improvements[0];
    }

    let samples = config.bootstrap_samples.max(1);
    let mut rng = SplitMix64::new(BOOTSTRAP_SEED);
    let mut means: Vec<f64> = Vec::with_capacity(samples);

    for _ in 0..samples {
        let mut acc = 0.0;
        for _ in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            acc += improvements[idx];
        }
        means.push(acc / n as f64);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // (1 - confidence) one-sided lower quantile.
    let alpha = (1.0 - config.confidence).clamp(0.0, 1.0);
    let rank = (alpha * (means.len() as f64 - 1.0)).round() as usize;
    means[rank.min(means.len() - 1)]
}

/// Deterministic SplitMix64 PRNG. Standalone (no `rand` dependency) so the gate's
/// output depends only on the inputs and the seed — reproducible across machines.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A candidate that is uniformly ~0.3 better per example is certified, and the
    /// certificate is fully reproducible (same seed → identical numbers).
    #[test]
    fn certifies_real_improvement_and_is_deterministic() {
        let base = vec![1.0; 200];
        let candidate = vec![0.7; 200];
        let losses = HeldOutLosses::new(base, candidate);
        let cfg = EvalGateConfig::default();

        let c1 = certify(&losses, &cfg);
        let c2 = certify(&losses, &cfg);

        assert!(c1.certified, "0.3 uniform improvement must certify");
        assert!(c1.ci_lower_bound > cfg.min_improvement);
        assert!((c1.mean_improvement - 0.3).abs() < 1e-9);
        assert_eq!(c1, c2, "certificate must be deterministic");
    }

    /// A candidate that is no better than the base model (zero mean improvement)
    /// is rejected — this is the freeloader the gate exists to stop.
    #[test]
    fn rejects_no_improvement() {
        let base = vec![1.0; 200];
        let candidate = vec![1.0; 200];
        let cert = certify(
            &HeldOutLosses::new(base, candidate),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert!(cert.ci_lower_bound <= cert.min_improvement);
        assert!(cert.reason.is_some());
    }

    /// A candidate that is *worse* (negative improvement) is rejected with a
    /// negative lower bound.
    #[test]
    fn rejects_regression() {
        let base = vec![0.5; 100];
        let candidate = vec![0.9; 100];
        let cert = certify(
            &HeldOutLosses::new(base, candidate),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert!(cert.mean_improvement < 0.0);
        assert!(cert.ci_lower_bound < 0.0);
    }

    /// A tiny mean improvement that does not clear the 0.02 margin is rejected:
    /// the gate pays only for non-trivial held-out gains.
    #[test]
    fn rejects_below_margin() {
        // ~0.01 mean improvement, well under the 0.02 default margin.
        let base = vec![1.0; 300];
        let candidate = vec![0.99; 300];
        let cert = certify(
            &HeldOutLosses::new(base, candidate),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert!(cert.mean_improvement < cert.min_improvement);
    }

    /// A noisy split where the *mean* looks good but the lower bound does not clear
    /// the margin is rejected — the CI lower bound, not the point estimate, decides.
    #[test]
    fn lower_bound_guards_against_noise() {
        // Half the examples improve a lot, half regress a lot: mean ~ small positive,
        // but high variance pushes the lower bound down.
        let mut base = Vec::new();
        let mut candidate = Vec::new();
        for i in 0..200 {
            base.push(1.0);
            if i % 2 == 0 {
                candidate.push(0.0); // +1.0 improvement
            } else {
                candidate.push(1.9); // -0.9 regression
            }
        }
        let cert = certify(
            &HeldOutLosses::new(base, candidate),
            &EvalGateConfig::default(),
        );
        // Mean improvement is +0.05 (> margin) but variance is huge.
        assert!(cert.mean_improvement > cert.min_improvement);
        assert!(
            cert.ci_lower_bound < cert.mean_improvement,
            "lower bound must sit below the noisy mean"
        );
        // The whole point of gating on the lower bound: a noisy split whose mean
        // clears the margin must still be rejected, so no false certificate is minted.
        assert!(
            !cert.certified,
            "noisy split must not be certified even though the mean clears the margin"
        );
    }

    #[test]
    fn rejects_empty_split() {
        let cert = certify(
            &HeldOutLosses::new(vec![], vec![]),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert_eq!(cert.n_examples, 0);
    }

    #[test]
    fn rejects_length_mismatch() {
        let cert = certify(
            &HeldOutLosses::new(vec![1.0, 1.0], vec![0.5]),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert!(cert.reason.unwrap().contains("mismatch"));
    }

    #[test]
    fn rejects_non_finite() {
        let cert = certify(
            &HeldOutLosses::new(vec![1.0, f64::NAN], vec![0.5, 0.5]),
            &EvalGateConfig::default(),
        );
        assert!(!cert.certified);
        assert!(cert.reason.unwrap().contains("non-finite"));
    }
}
