//! Basic usage of `lsh_vec_index`.
//!
//! Demonstrates the builder pattern, inserting random vectors, querying,
//! inspecting stats and metrics, auto-tuning, removal, and clearing.
//!
//! Run with:
//!   cargo run --example basic_usage

use lsh_vec_index::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const DIM: usize = 128;
const NUM_VECTORS: usize = 10_000;

fn main() {
    // ---------------------------------------------------------------
    // 1. Build an LSH index using the builder pattern.
    // ---------------------------------------------------------------
    println!("=== Step 1: Build the index ===");
    let index = LshIndex::builder()
        .dim(DIM)
        .num_hashes(8)
        .num_tables(16)
        .num_probes(3)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .enable_metrics()
        .build()
        .expect("failed to build index");

    println!(
        "Created index: dim={}, hashes=8, tables=16, probes=3, metric=Cosine, seed=42, metrics=on\n",
        DIM
    );

    // ---------------------------------------------------------------
    // 2. Generate and insert 10,000 random 128-d vectors.
    // ---------------------------------------------------------------
    println!("=== Step 2: Insert {} random vectors ===", NUM_VECTORS);

    let mut rng = StdRng::seed_from_u64(123);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();

    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|_| (0..DIM).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    for (id, v) in vectors.iter().enumerate() {
        index.insert(id, v).expect("insert failed");
    }

    println!("Inserted {} vectors.\n", index.len());

    // ---------------------------------------------------------------
    // 3. Query with one of the inserted vectors; print top-5 results.
    // ---------------------------------------------------------------
    println!("=== Step 3: Query top-5 nearest neighbors for vector #0 ===");

    let results = index
        .query(&vectors[0], 5)
        .expect("query failed");

    for (rank, r) in results.iter().enumerate() {
        println!("  rank={} id={:<6} distance={:.6}", rank + 1, r.id, r.distance);
    }
    println!();

    // ---------------------------------------------------------------
    // 4. Show index stats.
    // ---------------------------------------------------------------
    println!("=== Step 4: Index statistics ===");
    let stats = index.stats();
    println!("  {}", stats);
    println!("  Vectors:          {}", stats.num_vectors);
    println!("  Tables:           {}", stats.num_tables);
    println!("  Hashes per table: {}", stats.num_hashes);
    println!("  Dimension:        {}", stats.dimension);
    println!("  Total buckets:    {}", stats.total_buckets);
    println!("  Avg bucket size:  {:.2}", stats.avg_bucket_size);
    println!("  Max bucket size:  {}", stats.max_bucket_size);
    println!(
        "  Memory estimate:  {:.2} MB",
        stats.memory_estimate_bytes as f64 / (1024.0 * 1024.0)
    );
    println!();

    // ---------------------------------------------------------------
    // 5. Show metrics snapshot.
    // ---------------------------------------------------------------
    println!("=== Step 5: Metrics snapshot ===");
    if let Some(m) = index.metrics() {
        println!("  {}", m);
        println!("  Query count:              {}", m.query_count);
        println!("  Insert count:             {}", m.insert_count);
        println!("  Avg query time:           {:.2} us", m.avg_query_time_us);
        println!("  Avg candidates per query: {:.1}", m.avg_candidates_per_query);
        println!("  Bucket hit rate:          {:.1}%", m.hit_rate * 100.0);
    } else {
        println!("  Metrics not enabled.");
    }
    println!();

    // ---------------------------------------------------------------
    // 6. Auto-tuning: call suggest_params and print recommendations.
    // ---------------------------------------------------------------
    println!("=== Step 6: Auto-tuning suggestions ===");

    let suggested = suggest_params(0.9, NUM_VECTORS, DIM, DistanceMetric::Cosine);
    println!("  Target recall: 0.90");
    println!("  Suggested num_hashes:     {}", suggested.num_hashes);
    println!("  Suggested num_tables:     {}", suggested.num_tables);
    println!("  Suggested num_probes:     {}", suggested.num_probes);
    println!("  Estimated recall:         {:.4}", suggested.estimated_recall);

    // Also show the estimate_recall utility for our current parameters.
    let current_recall = estimate_recall(8, 16, 3, DistanceMetric::Cosine);
    println!(
        "  Current config estimated recall: {:.4}",
        current_recall
    );
    println!();

    // ---------------------------------------------------------------
    // 7. Remove a vector and verify it is gone.
    // ---------------------------------------------------------------
    println!("=== Step 7: Remove vector #42 ===");
    println!("  Contains #42 before remove: {}", index.contains(42));
    index.remove(42).expect("remove failed");
    println!("  Contains #42 after  remove: {}", index.contains(42));
    println!("  Index size after removal:   {}", index.len());
    println!();

    // ---------------------------------------------------------------
    // 8. Clear the index.
    // ---------------------------------------------------------------
    println!("=== Step 8: Clear the index ===");
    index.clear();
    println!("  Index size after clear: {}", index.len());
    println!("  Index is empty: {}", index.is_empty());
    println!();

    println!("Done.");
}
