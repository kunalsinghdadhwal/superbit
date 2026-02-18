use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Collects runtime statistics about index operations using lock-free atomic counters.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    query_count: AtomicU64,
    insert_count: AtomicU64,
    total_candidates_examined: AtomicU64,
    total_query_time_ns: AtomicU64,
    bucket_hits: AtomicU64,
    bucket_misses: AtomicU64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_query(&self, candidates: u64, duration_ns: u64) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_candidates_examined
            .fetch_add(candidates, Ordering::Relaxed);
        self.total_query_time_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    pub fn record_insert(&self) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_bucket_hit(&self) {
        self.bucket_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_bucket_miss(&self) {
        self.bucket_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let query_count = self.query_count.load(Ordering::Relaxed);
        let total_query_time_ns = self.total_query_time_ns.load(Ordering::Relaxed);
        let total_candidates = self.total_candidates_examined.load(Ordering::Relaxed);
        let hits = self.bucket_hits.load(Ordering::Relaxed);
        let misses = self.bucket_misses.load(Ordering::Relaxed);

        MetricsSnapshot {
            query_count,
            insert_count: self.insert_count.load(Ordering::Relaxed),
            avg_query_time_us: if query_count > 0 {
                total_query_time_ns as f64 / query_count as f64 / 1000.0
            } else {
                0.0
            },
            avg_candidates_per_query: if query_count > 0 {
                total_candidates as f64 / query_count as f64
            } else {
                0.0
            },
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.query_count.store(0, Ordering::Relaxed);
        self.insert_count.store(0, Ordering::Relaxed);
        self.total_candidates_examined.store(0, Ordering::Relaxed);
        self.total_query_time_ns.store(0, Ordering::Relaxed);
        self.bucket_hits.store(0, Ordering::Relaxed);
        self.bucket_misses.store(0, Ordering::Relaxed);
    }
}

/// A point-in-time snapshot of index metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub query_count: u64,
    pub insert_count: u64,
    pub avg_query_time_us: f64,
    pub avg_candidates_per_query: f64,
    /// Fraction of bucket probes that found at least one candidate.
    pub hit_rate: f64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Queries: {}, Inserts: {}, Avg query: {:.2}us, Avg candidates: {:.1}, Hit rate: {:.1}%",
            self.query_count,
            self.insert_count,
            self.avg_query_time_us,
            self.avg_candidates_per_query,
            self.hit_rate * 100.0,
        )
    }
}

/// RAII timer for measuring operation durations.
pub(crate) struct QueryTimer {
    start: Instant,
}

impl QueryTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
}
