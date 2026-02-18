//! Python bindings via PyO3.
//!
//! Requires the `python` feature flag. Build with [maturin](https://github.com/PyO3/maturin):
//!
//! ```sh
//! pip install maturin
//! maturin develop --features python
//! ```

use pyo3::prelude::*;

use crate::distance::DistanceMetric;
use crate::index::{IndexConfig, LshIndex};

/// Python-visible wrapper around [`LshIndex`].
#[pyclass(name = "LshIndex")]
pub struct PyLshIndex {
    inner: LshIndex,
}

#[pymethods]
impl PyLshIndex {
    /// Create a new LSH index.
    ///
    /// Args:
    ///     dim: Vector dimensionality.
    ///     num_hashes: Hash bits per table (1-64).
    ///     num_tables: Number of independent hash tables.
    ///     num_probes: Extra buckets to probe per table.
    ///     metric: One of "cosine", "euclidean", "dot".
    ///     seed: Optional RNG seed for reproducibility.
    #[new]
    #[pyo3(signature = (dim=768, num_hashes=8, num_tables=16, num_probes=3, metric="cosine", seed=None))]
    fn new(
        dim: usize,
        num_hashes: usize,
        num_tables: usize,
        num_probes: usize,
        metric: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let distance_metric = match metric {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" => DistanceMetric::Euclidean,
            "dot" | "dot_product" => DistanceMetric::DotProduct,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown metric: {other}"
                )))
            }
        };

        let config = IndexConfig {
            dim,
            num_hashes,
            num_tables,
            num_probes,
            distance_metric,
            normalize_vectors: true,
            seed,
        };

        let index = LshIndex::new(config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner: index })
    }

    /// Insert a vector with the given integer ID.
    fn insert(&self, id: usize, vector: Vec<f32>) -> PyResult<()> {
        self.inner
            .insert(id, &vector)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Query for the `k` nearest neighbors. Returns list of (id, distance).
    fn query(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let results = self
            .inner
            .query(&vector, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(results.into_iter().map(|r| (r.id, r.distance)).collect())
    }

    /// Remove a vector by ID.
    fn remove(&self, id: usize) -> PyResult<()> {
        self.inner
            .remove(id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Number of stored vectors.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner.stats())
    }

    /// Check if an ID exists.
    fn __contains__(&self, id: usize) -> bool {
        self.inner.contains(id)
    }
}

/// Register the module with Python.
#[pymodule]
fn superbit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLshIndex>()?;
    Ok(())
}
