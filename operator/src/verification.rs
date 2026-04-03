//! TOPLOC-inspired verification for untrusted operator contributions.
//!
//! Operators submit state transition proofs after each sync round. The protocol
//! verifies that gradient norms are statistically consistent across operators
//! training on similar data — outliers are flagged for slashing.
//!
//! Reference: INTELLECT-2 (https://arxiv.org/abs/2505.07291)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Proof of a valid state transition during local training.
///
/// Hash of (model_state_before, data_shard_hash, model_state_after) — proves
/// the operator actually trained on the claimed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionProof {
    /// Operator address or peer ID.
    pub operator: String,
    /// Number of training steps covered by this proof.
    pub steps: u64,
    /// Hash of (model_state_before || data_hash || model_state_after).
    pub state_transition_hash: [u8; 32],
    /// Random sample of gradient norms for statistical verification.
    pub gradient_norm_samples: Vec<f32>,
    /// Epoch this proof covers.
    pub epoch: u32,
}

/// Random sampling of gradient norms for lightweight statistical verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientNormSample {
    /// Layer index the sample was taken from.
    pub layer_index: u32,
    /// L2 norm of the gradient at this layer.
    pub norm: f32,
    /// Step number the sample was taken at.
    pub step: u64,
}

/// Generate a state transition proof.
///
/// Computes SHA-256(model_state_before || data_hash || model_state_after)
/// and includes randomly sampled gradient norms for statistical checks.
pub fn generate_proof(
    operator: &str,
    model_state_before: &[u8],
    data_hash: &[u8],
    model_state_after: &[u8],
    gradient_norms: &[f32],
    steps: u64,
    epoch: u32,
) -> StateTransitionProof {
    let mut hasher = Sha256::new();
    hasher.update(model_state_before);
    hasher.update(data_hash);
    hasher.update(model_state_after);
    let hash: [u8; 32] = hasher.finalize().into();

    StateTransitionProof {
        operator: operator.to_string(),
        steps,
        state_transition_hash: hash,
        gradient_norm_samples: gradient_norms.to_vec(),
        epoch,
    }
}

/// Verify that a set of operator proofs are statistically consistent.
///
/// Checks:
/// 1. All proofs have non-zero gradient norms (operator actually computed gradients).
/// 2. Gradient norms are within a reasonable range of the median (no fake/zero gradients).
/// 3. No single operator has wildly different norms (potential freeloading or poisoning).
///
/// Returns true if proofs are consistent, false if any operator looks suspicious.
pub fn verify_contributions(proofs: &[StateTransitionProof]) -> VerificationResult {
    if proofs.is_empty() {
        return VerificationResult {
            valid: true,
            suspicious_operators: vec![],
            reason: None,
        };
    }

    if proofs.len() == 1 {
        // Single operator — can only check for non-zero norms
        let proof = &proofs[0];
        if proof.gradient_norm_samples.iter().all(|&n| n == 0.0) {
            return VerificationResult {
                valid: false,
                suspicious_operators: vec![proof.operator.clone()],
                reason: Some("all gradient norms are zero".to_string()),
            };
        }
        return VerificationResult {
            valid: true,
            suspicious_operators: vec![],
            reason: None,
        };
    }

    // Compute median gradient norm across all operators
    let mut all_norms: Vec<f32> = proofs
        .iter()
        .flat_map(|p| p.gradient_norm_samples.iter().copied())
        .collect();

    if all_norms.is_empty() {
        return VerificationResult {
            valid: false,
            suspicious_operators: proofs.iter().map(|p| p.operator.clone()).collect(),
            reason: Some("no gradient norm samples provided".to_string()),
        };
    }

    all_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = all_norms[all_norms.len() / 2];

    if median == 0.0 {
        return VerificationResult {
            valid: false,
            suspicious_operators: proofs.iter().map(|p| p.operator.clone()).collect(),
            reason: Some("median gradient norm is zero".to_string()),
        };
    }

    // Check each operator's average norm against the median.
    // Threshold: operator's mean must be within 10x of the median.
    // This is deliberately loose — DeMo doesn't require identical gradients.
    let tolerance_factor = 10.0;
    let mut suspicious = Vec::new();

    for proof in proofs {
        if proof.gradient_norm_samples.is_empty() {
            suspicious.push(proof.operator.clone());
            continue;
        }

        let mean: f32 =
            proof.gradient_norm_samples.iter().sum::<f32>() / proof.gradient_norm_samples.len() as f32;

        if mean == 0.0 {
            suspicious.push(proof.operator.clone());
            continue;
        }

        let ratio = if mean > median {
            mean / median
        } else {
            median / mean
        };

        if ratio > tolerance_factor {
            suspicious.push(proof.operator.clone());
        }
    }

    VerificationResult {
        valid: suspicious.is_empty(),
        suspicious_operators: suspicious,
        reason: None,
    }
}

/// Result of verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub suspicious_operators: Vec<String>,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_proof() {
        let proof = generate_proof(
            "operator_a",
            b"state_before",
            b"data_hash",
            b"state_after",
            &[1.5, 2.3, 1.8],
            100,
            1,
        );

        assert_eq!(proof.operator, "operator_a");
        assert_eq!(proof.steps, 100);
        assert_eq!(proof.gradient_norm_samples.len(), 3);
        assert_ne!(proof.state_transition_hash, [0u8; 32]);
    }

    #[test]
    fn test_proof_deterministic() {
        let p1 = generate_proof("a", b"before", b"data", b"after", &[1.0], 1, 0);
        let p2 = generate_proof("a", b"before", b"data", b"after", &[1.0], 1, 0);
        assert_eq!(p1.state_transition_hash, p2.state_transition_hash);
    }

    #[test]
    fn test_verify_consistent_operators() {
        let proofs = vec![
            generate_proof("a", b"s1", b"d1", b"s2", &[1.5, 1.6, 1.4], 100, 1),
            generate_proof("b", b"s3", b"d2", b"s4", &[1.7, 1.5, 1.6], 100, 1),
            generate_proof("c", b"s5", b"d3", b"s6", &[1.4, 1.8, 1.5], 100, 1),
        ];

        let result = verify_contributions(&proofs);
        assert!(result.valid);
        assert!(result.suspicious_operators.is_empty());
    }

    #[test]
    fn test_verify_detects_zero_gradients() {
        let proofs = vec![
            generate_proof("a", b"s1", b"d1", b"s2", &[1.5, 1.6], 100, 1),
            generate_proof("b", b"s3", b"d2", b"s4", &[0.0, 0.0], 100, 1), // freeloader
        ];

        let result = verify_contributions(&proofs);
        assert!(!result.valid);
        assert!(result.suspicious_operators.contains(&"b".to_string()));
    }

    #[test]
    fn test_verify_detects_outlier() {
        let proofs = vec![
            generate_proof("a", b"s1", b"d1", b"s2", &[1.5, 1.6], 100, 1),
            generate_proof("b", b"s3", b"d2", b"s4", &[1.4, 1.7], 100, 1),
            generate_proof("c", b"s5", b"d3", b"s6", &[100.0, 200.0], 100, 1), // outlier
        ];

        let result = verify_contributions(&proofs);
        assert!(!result.valid);
        assert!(result.suspicious_operators.contains(&"c".to_string()));
    }
}
