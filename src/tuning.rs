use crate::distance::DistanceMetric;

/// Suggested parameters for the LSH index, produced by auto-tuning.
#[derive(Debug, Clone)]
pub struct SuggestedParams {
    pub num_hashes: usize,
    pub num_tables: usize,
    pub num_probes: usize,
    pub estimated_recall: f64,
}

/// Suggest LSH parameters based on dataset characteristics and desired recall.
///
/// Uses the theoretical collision probability for random hyperplane LSH:
///   P(collision per bit) = 1 - theta / pi
///
/// For K hash bits per table: P_table = P^K
/// For L tables: P_total = 1 - (1 - P_table)^L
///
/// # Arguments
/// * `target_recall` - Desired recall in [0.5, 0.999]
/// * `dataset_size` - Expected number of vectors
/// * `_dim` - Vector dimensionality (reserved for future heuristics)
/// * `metric` - Distance metric being used
pub fn suggest_params(
    target_recall: f64,
    dataset_size: usize,
    _dim: usize,
    metric: DistanceMetric,
) -> SuggestedParams {
    let target_recall = target_recall.clamp(0.5, 0.999);

    // Assume average angle of ~60 degrees between relevant (nearby) pairs.
    // P(sign match) = 1 - 60/180 = 0.667 for cosine/angular LSH.
    let p_collision = match metric {
        DistanceMetric::Cosine | DistanceMetric::DotProduct => 0.667,
        DistanceMetric::Euclidean => 0.6,
    };

    let mut best = SuggestedParams {
        num_hashes: 16,
        num_tables: 8,
        num_probes: 2,
        estimated_recall: 0.0,
    };
    let mut best_cost = f64::MAX;

    for k in 4..=32usize {
        let p_table = p_collision.powi(k as i32);

        // Minimum L tables so that 1 - (1 - p_table)^L >= target_recall
        let l_frac = (1.0 - target_recall).ln() / (1.0 - p_table).ln();
        let l = (l_frac.ceil() as usize).clamp(1, 100);

        let recall = 1.0 - (1.0 - p_table).powi(l as i32);

        // Cost heuristic balancing memory (proportional to L) and query time (L * K).
        let cost = l as f64 * (1.0 + k as f64);

        if recall >= target_recall && cost < best_cost {
            best_cost = cost;
            let probes = (k / 4).clamp(1, 8);
            best = SuggestedParams {
                num_hashes: k,
                num_tables: l,
                num_probes: probes,
                estimated_recall: recall,
            };
        }
    }

    // Larger datasets benefit from more tables.
    if dataset_size > 100_000 {
        let scale = ((dataset_size as f64 / 100_000.0).ln() + 1.0).ceil() as usize;
        best.num_tables = (best.num_tables * scale).min(50);
    }

    if target_recall > 0.95 {
        best.num_probes = (best.num_probes * 2).min(best.num_hashes);
    }

    best
}

/// Estimate recall for a given set of LSH parameters.
///
/// Accounts for multi-probe by approximating the additional collision probability
/// from flipping the most uncertain bits.
pub fn estimate_recall(
    num_hashes: usize,
    num_tables: usize,
    num_probes: usize,
    metric: DistanceMetric,
) -> f64 {
    let p_collision = match metric {
        DistanceMetric::Cosine | DistanceMetric::DotProduct => 0.667,
        DistanceMetric::Euclidean => 0.6,
    };

    let p_table = p_collision.powi(num_hashes as i32);

    // Each probe adds roughly p^(K-1) * (1-p) probability.
    let p_probe_bonus = if num_hashes > 1 {
        num_probes as f64
            * p_collision.powi((num_hashes - 1) as i32)
            * (1.0 - p_collision)
    } else {
        0.0
    };
    let p_effective = (p_table + p_probe_bonus).min(1.0);

    1.0 - (1.0 - p_effective).powi(num_tables as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_params_reasonable() {
        let params = suggest_params(0.9, 100_000, 768, DistanceMetric::Cosine);
        assert!(params.num_hashes >= 4);
        assert!(params.num_hashes <= 32);
        assert!(params.num_tables >= 1);
        assert!(params.estimated_recall >= 0.9);
    }

    #[test]
    fn test_higher_recall_needs_more_resources() {
        let low = suggest_params(0.8, 10_000, 128, DistanceMetric::Cosine);
        let high = suggest_params(0.95, 10_000, 128, DistanceMetric::Cosine);
        // Higher recall should need more tables or more probes.
        assert!(
            high.num_tables >= low.num_tables || high.num_probes >= low.num_probes,
            "high recall params should use more resources: low={low:?} high={high:?}"
        );
    }

    #[test]
    fn test_estimate_recall_increases_with_tables() {
        let r4 = estimate_recall(16, 4, 2, DistanceMetric::Cosine);
        let r8 = estimate_recall(16, 8, 2, DistanceMetric::Cosine);
        assert!(r8 > r4);
    }
}
