use ndarray::Array1;
use rand::Rng;
use rand_distr::StandardNormal;

/// A random-projection hash family for one hash table.
///
/// Uses sign-of-random-projection (SimHash / hyperplane LSH) to map vectors
/// to bit signatures. Each bit corresponds to the sign of the dot product
/// with a random Gaussian vector.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RandomProjectionHasher {
    projections: Vec<Array1<f32>>,
    num_hashes: usize,
}

impl RandomProjectionHasher {
    /// Create a new hasher with `num_hashes` random projection vectors of dimension `dim`.
    pub fn new(dim: usize, num_hashes: usize, rng: &mut impl Rng) -> Self {
        let projections = (0..num_hashes)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
                Array1::from_vec(v)
            })
            .collect();
        Self {
            projections,
            num_hashes,
        }
    }

    /// Compute the hash key for a vector, along with margin information for multi-probe.
    ///
    /// Returns `(hash_key, margins)` where margins is a vec of `(bit_index, |dot_product|)`
    /// sorted by ascending margin (most uncertain bits first).
    pub fn hash_vector(&self, vector: &ndarray::ArrayView1<f32>) -> (u64, Vec<(usize, f32)>) {
        let mut hash: u64 = 0;
        let mut margins: Vec<(usize, f32)> = Vec::with_capacity(self.num_hashes);

        for (i, proj) in self.projections.iter().enumerate() {
            let dot = vector.dot(proj);
            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
            margins.push((i, dot.abs()));
        }

        margins.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        (hash, margins)
    }

    /// Compute just the hash key (fast path, no margin data).
    pub fn hash_vector_fast(&self, vector: &ndarray::ArrayView1<f32>) -> u64 {
        let mut hash: u64 = 0;
        for (i, proj) in self.projections.iter().enumerate() {
            if vector.dot(proj) >= 0.0 {
                hash |= 1u64 << i;
            }
        }
        hash
    }

    /// Number of hash functions (bits in the signature).
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

/// Generate multi-probe hash keys by flipping the most uncertain bits.
///
/// Given the base hash and margin info (sorted ascending by uncertainty),
/// produces the base key plus `num_probes` perturbed keys.
pub fn multi_probe_keys(
    base_hash: u64,
    margins: &[(usize, f32)],
    num_probes: usize,
) -> Vec<u64> {
    let mut keys = Vec::with_capacity(1 + num_probes);
    keys.push(base_hash);

    for &(bit_idx, _) in margins.iter().take(num_probes) {
        keys.push(base_hash ^ (1u64 << bit_idx));
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_deterministic_hash() {
        let mut rng = StdRng::seed_from_u64(42);
        let hasher = RandomProjectionHasher::new(4, 8, &mut rng);
        let v = array![1.0, 2.0, 3.0, 4.0];
        let h1 = hasher.hash_vector_fast(&v.view());
        let h2 = hasher.hash_vector_fast(&v.view());
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_similar_vectors_likely_same_hash() {
        let mut rng = StdRng::seed_from_u64(42);
        let hasher = RandomProjectionHasher::new(4, 4, &mut rng);
        let v1 = array![1.0, 2.0, 3.0, 4.0];
        let v2 = array![1.01, 2.01, 3.01, 4.01];
        let h1 = hasher.hash_vector_fast(&v1.view());
        let h2 = hasher.hash_vector_fast(&v2.view());
        // Very similar vectors should often (but not always) hash the same
        // With only 4 bits, probability is high
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_multi_probe_keys() {
        let base = 0b1010u64;
        let margins = vec![(0, 0.1), (2, 0.5), (1, 0.8), (3, 1.2)];
        let keys = multi_probe_keys(base, &margins, 2);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], 0b1010); // base
        assert_eq!(keys[1], 0b1011); // flip bit 0
        assert_eq!(keys[2], 0b1110); // flip bit 2
    }
}
