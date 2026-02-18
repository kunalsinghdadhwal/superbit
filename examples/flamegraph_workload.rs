//! Heavy workload that exercises every major code path in `superbit`.
//!
//! Designed for profiling with `cargo flamegraph`:
//!
//! ```sh
//! PERF=/usr/lib/linux-tools/6.8.0-100-generic/perf \
//!   cargo flamegraph --example flamegraph_workload -o flamegraph.svg
//! ```
//!
//! The workload is structured in phases so the flamegraph shows clear
//! sections for insert, query, multi-probe, re-ranking, removal, and
//! auto-tuning.

use superbit::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

const DIM: usize = 768;
const NUM_VECTORS: usize = 100_000;
const NUM_QUERIES: usize = 500;
const TOP_K: usize = 20;
const NUM_REMOVALS: usize = 10_000;

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn main() {
    let total = Instant::now();

    // ------------------------------------------------------------------
    // Phase 1: Insert 100k vectors (768-d) -- profiles hashing + bucketing
    // ------------------------------------------------------------------
    eprintln!("[1/7] Building index and inserting {} vectors (dim={})...", NUM_VECTORS, DIM);

    let index = LshIndex::builder()
        .dim(DIM)
        .num_hashes(8)
        .num_tables(16)
        .num_probes(3)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .enable_metrics()
        .build()
        .expect("build failed");

    let vectors = generate_vectors(NUM_VECTORS, DIM, 7);

    let insert_start = Instant::now();
    for (id, v) in vectors.iter().enumerate() {
        index.insert(id, v).expect("insert failed");
    }
    eprintln!("    Inserted {} vectors in {:.2?}", NUM_VECTORS, insert_start.elapsed());

    // ------------------------------------------------------------------
    // Phase 2: Cosine queries (top-20) -- profiles multi-probe + re-rank
    // ------------------------------------------------------------------
    eprintln!("[2/7] Running {} cosine queries (top-{})...", NUM_QUERIES, TOP_K);

    let query_vecs = generate_vectors(NUM_QUERIES, DIM, 99);

    let query_start = Instant::now();
    let mut total_results = 0usize;
    for q in &query_vecs {
        let results = index.query(q, TOP_K).expect("query failed");
        total_results += results.len();
    }
    let query_elapsed = query_start.elapsed();
    eprintln!(
        "    {} queries in {:.2?} ({:.1} us/query, {} total results)",
        NUM_QUERIES,
        query_elapsed,
        query_elapsed.as_micros() as f64 / NUM_QUERIES as f64,
        total_results,
    );

    // ------------------------------------------------------------------
    // Phase 3: Euclidean queries -- profiles a different distance path
    // ------------------------------------------------------------------
    eprintln!("[3/7] Rebuilding with Euclidean metric and querying...");

    let euc_index = LshIndex::builder()
        .dim(DIM)
        .num_hashes(8)
        .num_tables(16)
        .num_probes(3)
        .distance_metric(DistanceMetric::Euclidean)
        .seed(42)
        .build()
        .expect("build failed");

    for (id, v) in vectors.iter().enumerate() {
        euc_index.insert(id, v).expect("insert failed");
    }

    let euc_start = Instant::now();
    for q in &query_vecs {
        let _ = euc_index.query(q, TOP_K).expect("query failed");
    }
    eprintln!("    Euclidean queries in {:.2?}", euc_start.elapsed());

    // ------------------------------------------------------------------
    // Phase 4: Dot-product queries -- profiles the third distance path
    // ------------------------------------------------------------------
    eprintln!("[4/7] Rebuilding with DotProduct metric and querying...");

    let dot_index = LshIndex::builder()
        .dim(DIM)
        .num_hashes(8)
        .num_tables(16)
        .num_probes(3)
        .distance_metric(DistanceMetric::DotProduct)
        .seed(42)
        .build()
        .expect("build failed");

    for (id, v) in vectors.iter().enumerate() {
        dot_index.insert(id, v).expect("insert failed");
    }

    let dot_start = Instant::now();
    for q in &query_vecs {
        let _ = dot_index.query(q, TOP_K).expect("query failed");
    }
    eprintln!("    DotProduct queries in {:.2?}", dot_start.elapsed());

    // ------------------------------------------------------------------
    // Phase 5: Removals -- profiles hash table cleanup
    // ------------------------------------------------------------------
    eprintln!("[5/7] Removing {} vectors...", NUM_REMOVALS);

    let remove_start = Instant::now();
    for id in 0..NUM_REMOVALS {
        index.remove(id).expect("remove failed");
    }
    eprintln!(
        "    Removed {} vectors in {:.2?} (len={})",
        NUM_REMOVALS,
        remove_start.elapsed(),
        index.len(),
    );

    // ------------------------------------------------------------------
    // Phase 6: Re-insert + duplicate overwrite -- profiles the replace path
    // ------------------------------------------------------------------
    eprintln!("[6/7] Re-inserting {} vectors (duplicate overwrite path)...", NUM_REMOVALS);

    let new_vectors = generate_vectors(NUM_REMOVALS, DIM, 555);
    let reinsert_start = Instant::now();
    // First insert fresh IDs back
    for (i, v) in new_vectors.iter().enumerate() {
        index.insert(i, v).expect("insert failed");
    }
    // Now overwrite them to exercise the duplicate-removal path
    for (i, v) in vectors[..NUM_REMOVALS].iter().enumerate() {
        index.insert(i, v).expect("insert failed");
    }
    eprintln!("    Re-inserted in {:.2?} (len={})", reinsert_start.elapsed(), index.len());

    // ------------------------------------------------------------------
    // Phase 7: Auto-tuning sweep -- profiles the tuning math
    // ------------------------------------------------------------------
    eprintln!("[7/7] Running auto-tuning sweep...");

    let tune_start = Instant::now();
    for target_recall in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99] {
        for dataset_size in [10_000, 50_000, 100_000, 500_000, 1_000_000] {
            for dim in [128, 256, 512, 768, 1536] {
                let _ = suggest_params(target_recall, dataset_size, dim, DistanceMetric::Cosine);
                let _ = suggest_params(target_recall, dataset_size, dim, DistanceMetric::Euclidean);
            }
        }
    }
    // Also sweep estimate_recall
    for h in 1..=16 {
        for t in 1..=32 {
            for p in 0..=4 {
                let _ = estimate_recall(h, t, p, DistanceMetric::Cosine);
            }
        }
    }
    eprintln!("    Auto-tuning sweep in {:.2?}", tune_start.elapsed());

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    let stats = index.stats();
    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("  Total wall time:   {:.2?}", total.elapsed());
    eprintln!("  Index:             {}", stats);
    if let Some(m) = index.metrics() {
        eprintln!("  Metrics:           {}", m);
    }
    eprintln!("  Done. Open flamegraph.svg in a browser.");
}
