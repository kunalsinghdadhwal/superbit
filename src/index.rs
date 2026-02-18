use std::sync::Arc;

use hashbrown::HashMap;
use ndarray::Array1;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distance::{self, DistanceMetric};
use crate::error::{LshError, Result};
use crate::hash::{multi_probe_keys, RandomProjectionHasher};
use crate::metrics::{MetricsCollector, MetricsSnapshot, QueryTimer};

/// Configuration for the LSH index.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct IndexConfig {
    /// Dimensionality of vectors.
    pub dim: usize,
    /// Number of hash bits per table (1..=64).
    pub num_hashes: usize,
    /// Number of independent hash tables.
    pub num_tables: usize,
    /// Number of extra buckets to probe per table during queries.
    pub num_probes: usize,
    /// Distance metric for ranking candidates.
    pub distance_metric: DistanceMetric,
    /// Whether to L2-normalize vectors on insertion (recommended for cosine).
    pub normalize_vectors: bool,
    /// Optional RNG seed for reproducible projections.
    pub seed: Option<u64>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            num_hashes: 8,
            num_tables: 16,
            num_probes: 3,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
            seed: None,
        }
    }
}

/// A single nearest-neighbor result.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The vector ID.
    pub id: usize,
    /// Distance from the query vector (lower is closer).
    pub distance: f32,
}

/// Aggregate statistics about the index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub num_hashes: usize,
    pub dimension: usize,
    pub total_buckets: usize,
    pub avg_bucket_size: f64,
    pub max_bucket_size: usize,
    pub memory_estimate_bytes: usize,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LshIndex {{ vectors: {}, tables: {}, hashes/table: {}, dim: {}, \
             buckets: {}, avg_bucket: {:.1}, max_bucket: {}, mem: ~{:.1}MB }}",
            self.num_vectors,
            self.num_tables,
            self.num_hashes,
            self.dimension,
            self.total_buckets,
            self.avg_bucket_size,
            self.max_bucket_size,
            self.memory_estimate_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

// ---------------------------------------------------------------------------
// Inner state (behind RwLock)
// ---------------------------------------------------------------------------

#[cfg_attr(
    feature = "persistence",
    derive(serde::Serialize, serde::Deserialize)
)]
pub(crate) struct IndexInner {
    pub(crate) vectors: HashMap<usize, Array1<f32>>,
    pub(crate) tables: Vec<HashMap<u64, Vec<usize>>>,
    pub(crate) hashers: Vec<RandomProjectionHasher>,
    pub(crate) config: IndexConfig,
    pub(crate) next_id: usize,
}

// ---------------------------------------------------------------------------
// LshIndex
// ---------------------------------------------------------------------------

/// A locality-sensitive hashing index for approximate nearest-neighbor search.
///
/// Thread-safe: concurrent reads (queries) proceed in parallel; writes
/// (inserts, removes) acquire exclusive access via `parking_lot::RwLock`.
pub struct LshIndex {
    pub(crate) inner: RwLock<IndexInner>,
    pub(crate) metrics: Option<Arc<MetricsCollector>>,
}

impl std::fmt::Debug for LshIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("LshIndex")
            .field("num_vectors", &inner.vectors.len())
            .field("config", &inner.config)
            .field("has_metrics", &self.metrics.is_some())
            .finish()
    }
}

impl LshIndex {
    /// Start building an index with the builder pattern.
    pub fn builder() -> LshIndexBuilder {
        LshIndexBuilder::new()
    }

    /// Create an index directly from an [`IndexConfig`].
    pub fn new(config: IndexConfig) -> Result<Self> {
        Self::new_with_metrics(config, false)
    }

    fn new_with_metrics(config: IndexConfig, enable_metrics: bool) -> Result<Self> {
        if config.dim == 0 {
            return Err(LshError::ZeroDimension);
        }
        if config.num_hashes == 0 || config.num_hashes > 64 {
            return Err(LshError::InvalidNumHashes(config.num_hashes));
        }
        if config.num_tables == 0 {
            return Err(LshError::InvalidConfig(
                "num_tables must be > 0".into(),
            ));
        }

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let hashers: Vec<RandomProjectionHasher> = (0..config.num_tables)
            .map(|_| RandomProjectionHasher::new(config.dim, config.num_hashes, &mut rng))
            .collect();

        let tables = (0..config.num_tables).map(|_| HashMap::new()).collect();

        let inner = IndexInner {
            vectors: HashMap::new(),
            tables,
            hashers,
            config,
            next_id: 0,
        };

        let metrics = if enable_metrics {
            Some(Arc::new(MetricsCollector::new()))
        } else {
            None
        };

        Ok(Self {
            inner: RwLock::new(inner),
            metrics,
        })
    }

    // ------------------------------------------------------------------
    // Insertion
    // ------------------------------------------------------------------

    /// Insert a vector with the given ID.
    ///
    /// If a vector with this ID already exists it is silently replaced.
    pub fn insert(&self, id: usize, vector: &[f32]) -> Result<()> {
        let mut inner = self.inner.write();

        if vector.len() != inner.config.dim {
            return Err(LshError::DimensionMismatch {
                expected: inner.config.dim,
                got: vector.len(),
            });
        }

        // If the id already exists, remove old hashes first.
        if let Some(old_vec) = inner.vectors.get(&id) {
            let old_vec = old_vec.clone();
            let old_hashes: Vec<u64> = inner
                .hashers
                .iter()
                .map(|h| h.hash_vector_fast(&old_vec.view()))
                .collect();
            for (i, old_hash) in old_hashes.into_iter().enumerate() {
                if let Some(bucket) = inner.tables[i].get_mut(&old_hash) {
                    bucket.retain(|&x| x != id);
                    if bucket.is_empty() {
                        inner.tables[i].remove(&old_hash);
                    }
                }
            }
        }

        let mut arr = Array1::from_vec(vector.to_vec());
        if inner.config.normalize_vectors {
            distance::normalize(&mut arr);
        }

        let new_hashes: Vec<u64> = inner
            .hashers
            .iter()
            .map(|h| h.hash_vector_fast(&arr.view()))
            .collect();
        for (i, hash) in new_hashes.into_iter().enumerate() {
            inner.tables[i].entry(hash).or_default().push(id);
        }

        inner.vectors.insert(id, arr);

        if id >= inner.next_id {
            inner.next_id = id + 1;
        }

        if let Some(ref m) = self.metrics {
            m.record_insert();
        }

        Ok(())
    }

    /// Insert a vector and receive an auto-assigned ID.
    ///
    /// The ID is assigned atomically under the write lock, so concurrent
    /// calls will never produce duplicate IDs.
    pub fn insert_auto(&self, vector: &[f32]) -> Result<usize> {
        let mut inner = self.inner.write();

        if vector.len() != inner.config.dim {
            return Err(LshError::DimensionMismatch {
                expected: inner.config.dim,
                got: vector.len(),
            });
        }

        let id = inner.next_id;

        let mut arr = Array1::from_vec(vector.to_vec());
        if inner.config.normalize_vectors {
            distance::normalize(&mut arr);
        }

        let new_hashes: Vec<u64> = inner
            .hashers
            .iter()
            .map(|h| h.hash_vector_fast(&arr.view()))
            .collect();
        for (i, hash) in new_hashes.into_iter().enumerate() {
            inner.tables[i].entry(hash).or_default().push(id);
        }

        inner.vectors.insert(id, arr);
        inner.next_id = id + 1;

        if let Some(ref m) = self.metrics {
            m.record_insert();
        }

        Ok(id)
    }

    /// Insert multiple vectors at once. Aborts on first error.
    pub fn insert_batch(&self, vectors: &[(usize, &[f32])]) -> Result<()> {
        for &(id, v) in vectors {
            self.insert(id, v)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Find the `k` approximate nearest neighbors for `vector`.
    ///
    /// Returns results sorted by ascending distance (closest first).
    pub fn query(&self, vector: &[f32], k: usize) -> Result<Vec<QueryResult>> {
        let timer = self.metrics.as_ref().map(|_| QueryTimer::new());
        let inner = self.inner.read();

        if vector.len() != inner.config.dim {
            return Err(LshError::DimensionMismatch {
                expected: inner.config.dim,
                got: vector.len(),
            });
        }

        if inner.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut query_vec = Array1::from_vec(vector.to_vec());
        if inner.config.normalize_vectors {
            distance::normalize(&mut query_vec);
        }

        // Collect candidate IDs across all tables.
        // Use a bitvec for O(1) dedup when IDs are dense sequential integers,
        // falling back to HashMap when IDs are sparse (next_id > 4 * num_vectors).
        let num_vectors = inner.vectors.len();
        let use_bitvec = inner.next_id <= num_vectors.saturating_mul(4);
        let mut seen = if use_bitvec {
            vec![false; inner.next_id]
        } else {
            Vec::new()
        };
        let mut candidate_set: HashMap<usize, ()> = if use_bitvec {
            HashMap::new() // unused, but need a binding
        } else {
            HashMap::with_capacity(num_vectors / 4)
        };
        let mut candidate_ids: Vec<usize> = Vec::new();

        for (i, hasher) in inner.hashers.iter().enumerate() {
            let (hash, margins) = hasher.hash_vector(&query_vec.view());

            let probe_keys = if inner.config.num_probes > 0 {
                multi_probe_keys(hash, &margins, inner.config.num_probes)
            } else {
                vec![hash]
            };

            for key in probe_keys {
                if let Some(bucket) = inner.tables[i].get(&key) {
                    if let Some(ref m) = self.metrics {
                        m.record_bucket_hit();
                    }
                    for &id in bucket {
                        if use_bitvec {
                            if !seen[id] {
                                seen[id] = true;
                                candidate_ids.push(id);
                            }
                        } else if candidate_set.insert(id, ()).is_none() {
                            candidate_ids.push(id);
                        }
                    }
                } else if let Some(ref m) = self.metrics {
                    m.record_bucket_miss();
                }
            }
        }

        // Exact re-ranking of candidates.
        // When vectors are pre-normalized (cosine mode), use the fast 1-dot path
        // which avoids two redundant norm computations per candidate.
        let use_fast_cosine = inner.config.normalize_vectors
            && inner.config.distance_metric == distance::DistanceMetric::Cosine;
        let query_view = query_vec.view();

        let num_candidates = candidate_ids.len();

        let mut results: Vec<QueryResult> = candidate_ids
            .iter()
            .filter_map(|&id| {
                inner.vectors.get(&id).map(|stored| {
                    let dist = if use_fast_cosine {
                        distance::cosine_distance_normalized(&query_view, &stored.view())
                    } else {
                        inner
                            .config
                            .distance_metric
                            .compute(&query_view, &stored.view())
                    };
                    QueryResult { id, distance: dist }
                })
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        if let Some(ref m) = self.metrics {
            if let Some(t) = timer {
                m.record_query(num_candidates as u64, t.elapsed_ns());
            }
        }

        Ok(results)
    }

    // ------------------------------------------------------------------
    // Removal / lookup
    // ------------------------------------------------------------------

    /// Remove a vector by ID.
    pub fn remove(&self, id: usize) -> Result<()> {
        let mut inner = self.inner.write();

        let vec = inner.vectors.remove(&id).ok_or(LshError::NotFound(id))?;

        let hashes: Vec<u64> = inner
            .hashers
            .iter()
            .map(|h| h.hash_vector_fast(&vec.view()))
            .collect();
        for (i, hash) in hashes.into_iter().enumerate() {
            if let Some(bucket) = inner.tables[i].get_mut(&hash) {
                bucket.retain(|&x| x != id);
                if bucket.is_empty() {
                    inner.tables[i].remove(&hash);
                }
            }
        }

        Ok(())
    }

    /// Check whether a vector ID is present.
    pub fn contains(&self, id: usize) -> bool {
        self.inner.read().vectors.contains_key(&id)
    }

    // ------------------------------------------------------------------
    // Stats / metrics
    // ------------------------------------------------------------------

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.inner.read().vectors.len()
    }

    /// True when the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.read().vectors.is_empty()
    }

    /// Compute aggregate statistics about the index.
    pub fn stats(&self) -> IndexStats {
        let inner = self.inner.read();

        let total_buckets: usize = inner.tables.iter().map(|t| t.len()).sum();
        let total_entries: usize = inner
            .tables
            .iter()
            .flat_map(|t| t.values())
            .map(|v| v.len())
            .sum();
        let max_bucket_size = inner
            .tables
            .iter()
            .flat_map(|t| t.values())
            .map(|v| v.len())
            .max()
            .unwrap_or(0);

        let avg_bucket_size = if total_buckets > 0 {
            total_entries as f64 / total_buckets as f64
        } else {
            0.0
        };

        let vector_mem =
            inner.vectors.len() * (inner.config.dim * 4 + std::mem::size_of::<usize>());
        let table_mem = total_buckets * (std::mem::size_of::<u64>() + 24);
        let entry_mem = total_entries * std::mem::size_of::<usize>();
        let proj_mem =
            inner.config.num_tables * inner.config.num_hashes * inner.config.dim * 4;

        IndexStats {
            num_vectors: inner.vectors.len(),
            num_tables: inner.config.num_tables,
            num_hashes: inner.config.num_hashes,
            dimension: inner.config.dim,
            total_buckets,
            avg_bucket_size,
            max_bucket_size,
            memory_estimate_bytes: vector_mem + table_mem + entry_mem + proj_mem,
        }
    }

    /// Snapshot of runtime metrics (`None` if metrics were not enabled).
    pub fn metrics(&self) -> Option<MetricsSnapshot> {
        self.metrics.as_ref().map(|m| m.snapshot())
    }

    /// Reset metrics counters.
    pub fn reset_metrics(&self) {
        if let Some(ref m) = self.metrics {
            m.reset();
        }
    }

    /// Remove all vectors from the index (projections are preserved).
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.vectors.clear();
        for table in &mut inner.tables {
            table.clear();
        }
        inner.next_id = 0;
    }

    /// Return a clone of the current configuration.
    pub fn config(&self) -> IndexConfig {
        self.inner.read().config.clone()
    }
}

// ---------------------------------------------------------------------------
// Parallel batch ops (behind `parallel` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
impl LshIndex {
    /// Insert many vectors using rayon for parallel hash computation.
    pub fn par_insert_batch(&self, vectors: &[(usize, Vec<f32>)]) -> Result<()> {
        use rayon::prelude::*;

        // Snapshot config + hashers so we don't hold the lock during parallel work.
        let (config, hashers) = {
            let inner = self.inner.read();
            (inner.config.clone(), inner.hashers.clone())
        };

        // Validate dimensions.
        for (_, v) in vectors {
            if v.len() != config.dim {
                return Err(LshError::DimensionMismatch {
                    expected: config.dim,
                    got: v.len(),
                });
            }
        }

        // Parallel: normalise + hash.
        let prepared: Vec<(usize, Array1<f32>, Vec<u64>)> = vectors
            .par_iter()
            .map(|(id, v)| {
                let mut arr = Array1::from_vec(v.clone());
                if config.normalize_vectors {
                    distance::normalize(&mut arr);
                }
                let hashes: Vec<u64> = hashers
                    .iter()
                    .map(|h| h.hash_vector_fast(&arr.view()))
                    .collect();
                (*id, arr, hashes)
            })
            .collect();

        // Sequential: write to shared state.
        let mut inner = self.inner.write();
        for (id, arr, hashes) in prepared {
            // If the id already exists, remove old hashes first (same as insert).
            if let Some(old_vec) = inner.vectors.get(&id) {
                let old_vec = old_vec.clone();
                let old_hashes: Vec<u64> = hashers
                    .iter()
                    .map(|h| h.hash_vector_fast(&old_vec.view()))
                    .collect();
                for (i, old_hash) in old_hashes.into_iter().enumerate() {
                    if let Some(bucket) = inner.tables[i].get_mut(&old_hash) {
                        bucket.retain(|&x| x != id);
                        if bucket.is_empty() {
                            inner.tables[i].remove(&old_hash);
                        }
                    }
                }
            }

            for (i, hash) in hashes.into_iter().enumerate() {
                inner.tables[i].entry(hash).or_default().push(id);
            }
            inner.vectors.insert(id, arr);
            if id >= inner.next_id {
                inner.next_id = id + 1;
            }
        }

        Ok(())
    }

    /// Query multiple vectors in parallel.
    pub fn par_query_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<QueryResult>>> {
        use rayon::prelude::*;

        queries
            .par_iter()
            .map(|q| self.query(q, k))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`LshIndex`].
#[derive(Default)]
pub struct LshIndexBuilder {
    config: IndexConfig,
    enable_metrics: bool,
}

impl LshIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dim(mut self, dim: usize) -> Self {
        self.config.dim = dim;
        self
    }

    pub fn num_hashes(mut self, n: usize) -> Self {
        self.config.num_hashes = n;
        self
    }

    pub fn num_tables(mut self, n: usize) -> Self {
        self.config.num_tables = n;
        self
    }

    pub fn num_probes(mut self, n: usize) -> Self {
        self.config.num_probes = n;
        self
    }

    pub fn distance_metric(mut self, m: DistanceMetric) -> Self {
        self.config.distance_metric = m;
        self
    }

    pub fn normalize(mut self, yes: bool) -> Self {
        self.config.normalize_vectors = yes;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    pub fn enable_metrics(mut self) -> Self {
        self.enable_metrics = true;
        self
    }

    /// Build the index, returning an error on invalid configuration.
    pub fn build(self) -> Result<LshIndex> {
        LshIndex::new_with_metrics(self.config, self.enable_metrics)
    }
}
