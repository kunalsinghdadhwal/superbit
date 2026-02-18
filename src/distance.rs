use ndarray::{Array1, ArrayView1};

/// Distance metric used for nearest-neighbor comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cos(a, b). Range [0, 2]. 0 = identical direction.
    Cosine,
    /// Euclidean (L2) distance. Range [0, inf).
    Euclidean,
    /// Negative dot product (so smaller = more similar). Range (-inf, inf).
    DotProduct,
}

impl DistanceMetric {
    /// Compute the distance between two vectors using this metric.
    pub fn compute(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b),
        }
    }
}

/// Cosine distance: 1 - cos(a, b).
pub fn cosine_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}

/// Euclidean (L2) distance between two vectors.
pub fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Dot product of two vectors.
pub fn dot_product(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.dot(b)
}

/// Normalize a vector to unit length (L2 norm). Leaves zero vectors unchanged.
pub fn normalize(v: &mut Array1<f32>) {
    let norm = v.dot(v).sqrt();
    if norm > f32::EPSILON {
        *v /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cosine_identical() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 0.0];
        let d = cosine_distance(&a.view(), &b.view());
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let d = cosine_distance(&a.view(), &b.view());
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = euclidean_distance(&a.view(), &b.view());
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = array![3.0, 4.0];
        normalize(&mut v);
        let norm = v.dot(&v).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = array![0.0, 0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, array![0.0, 0.0, 0.0]);
    }
}
