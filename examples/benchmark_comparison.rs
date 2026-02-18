//! Benchmark comparison: LSH approximate search vs. brute-force exact search.
//!
//! Generates random vectors, builds an LSH index, and compares query speed
//! and recall against a simple linear scan.
//!
//! Run with:
//!   cargo run --example benchmark_comparison --release

use lsh_vec_index::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

const N: usize = 50_000;
const DIM: usize = 256;
const NUM_QUERIES: usize = 100;
const K: usize = 10;

/// Brute-force nearest-neighbor search using cosine distance.
///
/// Computes the distance from `query` to every vector in `dataset`,
/// then returns the top-k (id, distance) pairs sorted by distance.
fn brute_force_search(
    dataset: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_arr = ndarray::Array1::from_vec(query.to_vec());
    let query_norm = query_arr.dot(&query_arr).sqrt();

    let mut dists: Vec<(usize, f32)> = dataset
        .iter()
        .enumerate()
        .map(|(id, v)| {
            let v_arr = ndarray::ArrayView1::from(v.as_slice());
            let dot = query_arr.dot(&v_arr);
            let v_norm = v_arr.dot(&v_arr).sqrt();
            let denom = query_norm * v_norm;
            let cosine_dist = if denom < f32::EPSILON {
                1.0
            } else {
                1.0 - (dot / denom)
            };
            (id, cosine_dist)
        })
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

fn main() {
    println!("========================================");
    println!("  LSH vs Brute-Force Benchmark");
    println!("========================================");
    println!("  Vectors:    {}", N);
    println!("  Dimension:  {}", DIM);
    println!("  Queries:    {}", NUM_QUERIES);
    println!("  Top-K:      {}", K);
    println!();

    // ---------------------------------------------------------------
    // 1. Generate random vectors.
    // ---------------------------------------------------------------
    println!("[1/5] Generating {} random {}-d vectors...", N, DIM);
    let mut rng = StdRng::seed_from_u64(7);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();

    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| normal.sample(&mut rng)).collect())
        .collect();
    println!("      Done.\n");

    // ---------------------------------------------------------------
    // 2. Build the LSH index.
    // ---------------------------------------------------------------
    println!("[2/5] Building LSH index (16 hashes, 12 tables, cosine)...");
    let build_start = Instant::now();
    let index = LshIndex::builder()
        .dim(DIM)
        .num_hashes(16)
        .num_tables(12)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .build()
        .expect("failed to build index");

    for (id, v) in vectors.iter().enumerate() {
        index.insert(id, v).expect("insert failed");
    }
    let build_elapsed = build_start.elapsed();
    println!("      Built in {:.2?}.\n", build_elapsed);

    // ---------------------------------------------------------------
    // 3. Select query vectors (pick evenly spaced IDs from the dataset).
    // ---------------------------------------------------------------
    let query_ids: Vec<usize> = (0..NUM_QUERIES)
        .map(|i| i * (N / NUM_QUERIES))
        .collect();

    // ---------------------------------------------------------------
    // 4. Run LSH queries and measure time.
    // ---------------------------------------------------------------
    println!("[3/5] Running {} LSH queries (top-{})...", NUM_QUERIES, K);
    let lsh_start = Instant::now();
    let lsh_results: Vec<Vec<QueryResult>> = query_ids
        .iter()
        .map(|&qid| index.query(&vectors[qid], K).expect("lsh query failed"))
        .collect();
    let lsh_elapsed = lsh_start.elapsed();
    println!("      LSH total time:    {:.2?}", lsh_elapsed);
    println!(
        "      LSH avg per query: {:.2?}\n",
        lsh_elapsed / NUM_QUERIES as u32
    );

    // ---------------------------------------------------------------
    // 5. Run brute-force queries and measure time.
    // ---------------------------------------------------------------
    println!("[4/5] Running {} brute-force queries (top-{})...", NUM_QUERIES, K);
    let bf_start = Instant::now();
    let bf_results: Vec<Vec<(usize, f32)>> = query_ids
        .iter()
        .map(|&qid| brute_force_search(&vectors, &vectors[qid], K))
        .collect();
    let bf_elapsed = bf_start.elapsed();
    println!("      Brute-force total time:    {:.2?}", bf_elapsed);
    println!(
        "      Brute-force avg per query: {:.2?}\n",
        bf_elapsed / NUM_QUERIES as u32
    );

    // ---------------------------------------------------------------
    // 6. Compute recall@K.
    //    For each query, check how many of the true top-K IDs the LSH
    //    index found.
    // ---------------------------------------------------------------
    println!("[5/5] Computing recall@{} ...", K);
    let mut total_recall = 0.0_f64;
    for (lsh_res, bf_res) in lsh_results.iter().zip(bf_results.iter()) {
        let true_ids: std::collections::HashSet<usize> =
            bf_res.iter().map(|&(id, _)| id).collect();
        let found = lsh_res
            .iter()
            .filter(|r| true_ids.contains(&r.id))
            .count();
        total_recall += found as f64 / K as f64;
    }
    let avg_recall = total_recall / NUM_QUERIES as f64;

    // ---------------------------------------------------------------
    // 7. Print summary.
    // ---------------------------------------------------------------
    let speedup = bf_elapsed.as_secs_f64() / lsh_elapsed.as_secs_f64();

    println!();
    println!("========================================");
    println!("  Summary");
    println!("========================================");
    println!(
        "  {:<28} {:>12}",
        "Metric", "Value"
    );
    println!("  {:-<28} {:-<12}", "", "");
    println!(
        "  {:<28} {:>12}",
        "Vectors", N
    );
    println!(
        "  {:<28} {:>12}",
        "Dimension", DIM
    );
    println!(
        "  {:<28} {:>12}",
        "Queries", NUM_QUERIES
    );
    println!(
        "  {:<28} {:>12}",
        "Top-K", K
    );
    println!(
        "  {:<28} {:>12.2?}",
        "LSH total time", lsh_elapsed
    );
    println!(
        "  {:<28} {:>12.2?}",
        "Brute-force total time", bf_elapsed
    );
    println!(
        "  {:<28} {:>11.1}x",
        "Speedup (brute/LSH)", speedup
    );
    println!(
        "  {:<28} {:>11.1}%",
        "Recall@10", avg_recall * 100.0
    );
    println!("========================================");

    if speedup > 1.0 {
        println!(
            "\nLSH was {:.1}x faster than brute-force with {:.1}% recall.",
            speedup,
            avg_recall * 100.0
        );
    } else {
        println!(
            "\nBrute-force was faster (LSH speedup: {:.2}x). \
             This can happen with small datasets or in debug mode.",
            speedup
        );
    }
}
