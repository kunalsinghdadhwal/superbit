use lsh_vec_index::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..dim).map(|_| normal.sample(rng)).collect()
}

fn make_index(dim: usize, seed: u64) -> LshIndex {
    LshIndex::builder()
        .dim(dim)
        .num_hashes(16)
        .num_tables(8)
        .num_probes(2)
        .distance_metric(DistanceMetric::Cosine)
        .seed(seed)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// 1. Basic insert and query
// ---------------------------------------------------------------------------

#[test]
fn test_basic_insert_and_query() {
    let index = make_index(32, 42);
    let v = vec![1.0_f32; 32];
    index.insert(0, &v).unwrap();

    let results = index.query(&v, 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!(results[0].distance < 1e-5, "self-query distance should be ~0");
}

// ---------------------------------------------------------------------------
// 2. Builder pattern (all options)
// ---------------------------------------------------------------------------

#[test]
fn test_builder_all_options() {
    let index = LshIndex::builder()
        .dim(64)
        .num_hashes(8)
        .num_tables(4)
        .num_probes(3)
        .distance_metric(DistanceMetric::Euclidean)
        .normalize(false)
        .seed(99)
        .enable_metrics()
        .build()
        .unwrap();

    let cfg = index.config();
    assert_eq!(cfg.dim, 64);
    assert_eq!(cfg.num_hashes, 8);
    assert_eq!(cfg.num_tables, 4);
    assert_eq!(cfg.num_probes, 3);
    assert_eq!(cfg.distance_metric, DistanceMetric::Euclidean);
    assert!(!cfg.normalize_vectors);
    assert_eq!(cfg.seed, Some(99));
    assert!(index.metrics().is_some());
}

// ---------------------------------------------------------------------------
// 3. Dimension mismatch error
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_mismatch_on_insert() {
    let index = make_index(32, 42);
    let wrong = vec![1.0_f32; 64];
    let err = index.insert(0, &wrong).unwrap_err();
    assert!(
        matches!(err, LshError::DimensionMismatch { expected: 32, got: 64 }),
        "expected DimensionMismatch, got: {err:?}"
    );
}

#[test]
fn test_dimension_mismatch_on_query() {
    let index = make_index(32, 42);
    index.insert(0, &[1.0; 32]).unwrap();
    let wrong = vec![1.0_f32; 16];
    let err = index.query(&wrong, 5).unwrap_err();
    assert!(
        matches!(err, LshError::DimensionMismatch { expected: 32, got: 16 }),
        "expected DimensionMismatch, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. Empty index query returns empty vec
// ---------------------------------------------------------------------------

#[test]
fn test_empty_index_query() {
    let index = make_index(32, 42);
    let results = index.query(&[1.0; 32], 10).unwrap();
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// 5. Zero dimension error
// ---------------------------------------------------------------------------

#[test]
fn test_zero_dimension_error() {
    let result = LshIndex::builder()
        .dim(0)
        .num_hashes(8)
        .num_tables(4)
        .build();
    match result {
        Err(ref e) => assert!(
            matches!(e, LshError::ZeroDimension),
            "expected ZeroDimension, got: {e:?}"
        ),
        Ok(_) => panic!("expected ZeroDimension error, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// 6. Invalid num_hashes (0 and >64)
// ---------------------------------------------------------------------------

#[test]
fn test_invalid_num_hashes_zero() {
    let result = LshIndex::builder()
        .dim(32)
        .num_hashes(0)
        .num_tables(4)
        .build();
    match result {
        Err(ref e) => assert!(
            matches!(e, LshError::InvalidNumHashes(0)),
            "expected InvalidNumHashes(0), got: {e:?}"
        ),
        Ok(_) => panic!("expected InvalidNumHashes error, got Ok"),
    }
}

#[test]
fn test_invalid_num_hashes_too_large() {
    let result = LshIndex::builder()
        .dim(32)
        .num_hashes(65)
        .num_tables(4)
        .build();
    match result {
        Err(ref e) => assert!(
            matches!(e, LshError::InvalidNumHashes(65)),
            "expected InvalidNumHashes(65), got: {e:?}"
        ),
        Ok(_) => panic!("expected InvalidNumHashes error, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// 7. Duplicate insert (same ID replaces old vector)
// ---------------------------------------------------------------------------

#[test]
fn test_duplicate_insert_replaces() {
    let index = LshIndex::builder()
        .dim(4)
        .num_hashes(8)
        .num_tables(4)
        .num_probes(2)
        .distance_metric(DistanceMetric::Euclidean)
        .normalize(false)
        .seed(42)
        .build()
        .unwrap();

    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 0.0, 0.0, 1.0];

    index.insert(0, &v1).unwrap();
    assert_eq!(index.len(), 1);

    // Replace with a different vector under the same ID.
    index.insert(0, &v2).unwrap();
    assert_eq!(index.len(), 1);

    // Query with v2 -- the stored vector should now be v2.
    let results = index.query(&v2, 1).unwrap();
    assert_eq!(results[0].id, 0);
    assert!(
        results[0].distance < 1e-5,
        "after replacement, distance to v2 should be ~0, got {}",
        results[0].distance
    );
}

// ---------------------------------------------------------------------------
// 8. Remove and verify not found
// ---------------------------------------------------------------------------

#[test]
fn test_remove() {
    let index = make_index(32, 42);
    let v = vec![1.0; 32];
    index.insert(0, &v).unwrap();
    assert!(index.contains(0));

    index.remove(0).unwrap();
    assert!(!index.contains(0));
    assert_eq!(index.len(), 0);
}

#[test]
fn test_remove_not_found() {
    let index = make_index(32, 42);
    let err = index.remove(999).unwrap_err();
    assert!(
        matches!(err, LshError::NotFound(999)),
        "expected NotFound(999), got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 9. insert_auto returns incrementing IDs
// ---------------------------------------------------------------------------

#[test]
fn test_insert_auto_incrementing_ids() {
    let index = make_index(8, 42);
    let v = vec![1.0; 8];

    let id0 = index.insert_auto(&v).unwrap();
    let id1 = index.insert_auto(&v).unwrap();
    let id2 = index.insert_auto(&v).unwrap();

    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
    assert_eq!(index.len(), 3);
}

// ---------------------------------------------------------------------------
// 10. insert_batch works correctly
// ---------------------------------------------------------------------------

#[test]
fn test_insert_batch() {
    let index = make_index(8, 42);
    let v0 = vec![1.0; 8];
    let v1 = vec![2.0; 8];
    let v2 = vec![3.0; 8];

    let batch: Vec<(usize, &[f32])> = vec![(10, &v0), (20, &v1), (30, &v2)];
    index.insert_batch(&batch).unwrap();

    assert_eq!(index.len(), 3);
    assert!(index.contains(10));
    assert!(index.contains(20));
    assert!(index.contains(30));
}

#[test]
fn test_insert_batch_dimension_mismatch() {
    let index = make_index(8, 42);
    let good = vec![1.0; 8];
    let bad = vec![1.0; 4];
    let batch: Vec<(usize, &[f32])> = vec![(0, &good), (1, &bad)];
    let err = index.insert_batch(&batch).unwrap_err();
    assert!(matches!(err, LshError::DimensionMismatch { .. }));
}

// ---------------------------------------------------------------------------
// 11. Cosine vs Euclidean vs DotProduct metrics ordering
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_metric_ordering() {
    // 1 hash bit + 1 probe = all 2 buckets checked per table, guaranteeing
    // that both vectors appear as candidates regardless of projection angles.
    let index = LshIndex::builder()
        .dim(3)
        .num_hashes(1)
        .num_tables(32)
        .num_probes(1)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .build()
        .unwrap();

    // v_close is in the same direction as query, v_far is orthogonal.
    let query = vec![1.0, 0.0, 0.0];
    let v_close = vec![1.0, 0.1, 0.0];
    let v_far = vec![0.0, 1.0, 0.0];

    index.insert(0, &v_close).unwrap();
    index.insert(1, &v_far).unwrap();

    let results = index.query(&query, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0, "closer vector should rank first");
    assert!(results[0].distance < results[1].distance);
}

#[test]
fn test_euclidean_metric_ordering() {
    // Same 1-bit strategy for guaranteed recall on 2 vectors.
    let index = LshIndex::builder()
        .dim(3)
        .num_hashes(1)
        .num_tables(32)
        .num_probes(1)
        .distance_metric(DistanceMetric::Euclidean)
        .normalize(false)
        .seed(42)
        .build()
        .unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let v_close = vec![1.0, 0.0, 0.0];
    let v_far = vec![10.0, 10.0, 10.0];

    index.insert(0, &v_close).unwrap();
    index.insert(1, &v_far).unwrap();

    let results = index.query(&query, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0, "closer vector should rank first");
    assert!(results[0].distance < results[1].distance);
}

#[test]
fn test_dot_product_metric_ordering() {
    let index = LshIndex::builder()
        .dim(3)
        .num_hashes(8)
        .num_tables(16)
        .num_probes(4)
        .distance_metric(DistanceMetric::DotProduct)
        .normalize(false)
        .seed(42)
        .build()
        .unwrap();

    // DotProduct distance = -dot(a, b), so higher dot product means smaller distance.
    let query = vec![1.0, 1.0, 1.0];
    let v_high_dot = vec![10.0, 10.0, 10.0]; // dot = 30
    let v_low_dot = vec![0.1, 0.1, 0.1]; // dot = 0.3

    index.insert(0, &v_high_dot).unwrap();
    index.insert(1, &v_low_dot).unwrap();

    let results = index.query(&query, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].id, 0,
        "vector with higher dot product should rank first (lower negative distance)"
    );
    assert!(results[0].distance < results[1].distance);
}

// ---------------------------------------------------------------------------
// 12. Multi-probe improves recall
// ---------------------------------------------------------------------------

#[test]
fn test_multi_probe_improves_recall() {
    let dim = 64;
    let n = 500;
    let k = 10;
    let mut rng = StdRng::seed_from_u64(123);

    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(&mut rng, dim)).collect();

    // Build index without multi-probe.
    let index_no_probe = LshIndex::builder()
        .dim(dim)
        .num_hashes(16)
        .num_tables(8)
        .num_probes(0)
        .distance_metric(DistanceMetric::Cosine)
        .seed(77)
        .build()
        .unwrap();

    // Build index with multi-probe.
    let index_with_probe = LshIndex::builder()
        .dim(dim)
        .num_hashes(16)
        .num_tables(8)
        .num_probes(4)
        .distance_metric(DistanceMetric::Cosine)
        .seed(77)
        .build()
        .unwrap();

    for (i, v) in vectors.iter().enumerate() {
        index_no_probe.insert(i, v).unwrap();
        index_with_probe.insert(i, v).unwrap();
    }

    // Measure average number of candidates returned for several queries.
    let num_queries = 20;
    let mut total_no_probe = 0usize;
    let mut total_with_probe = 0usize;

    for _ in 0..num_queries {
        let q = random_vector(&mut rng, dim);
        let r_no = index_no_probe.query(&q, k).unwrap();
        let r_yes = index_with_probe.query(&q, k).unwrap();
        total_no_probe += r_no.len();
        total_with_probe += r_yes.len();
    }

    // Multi-probe should return at least as many results (more candidates found).
    assert!(
        total_with_probe >= total_no_probe,
        "multi-probe ({total_with_probe}) should find at least as many results as no-probe ({total_no_probe})"
    );
}

// ---------------------------------------------------------------------------
// 13. Thread safety: concurrent inserts and queries
// ---------------------------------------------------------------------------

#[test]
fn test_thread_safety_concurrent_ops() {
    let index = Arc::new(make_index(32, 42));
    let mut handles = vec![];

    // Spawn writer threads.
    for t in 0..4 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64);
            for i in 0..50 {
                let id = t * 1000 + i;
                let v = random_vector(&mut rng, 32);
                idx.insert(id, &v).unwrap();
            }
        }));
    }

    // Spawn reader threads.
    for t in 0..4 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(100 + t as u64);
            for _ in 0..50 {
                let q = random_vector(&mut rng, 32);
                let _ = idx.query(&q, 5);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(index.len(), 200, "4 threads x 50 inserts = 200 vectors");
}

// ---------------------------------------------------------------------------
// 14. Stats reporting
// ---------------------------------------------------------------------------

#[test]
fn test_stats_reporting() {
    let index = make_index(32, 42);
    let mut rng = StdRng::seed_from_u64(1);

    for i in 0..100 {
        let v = random_vector(&mut rng, 32);
        index.insert(i, &v).unwrap();
    }

    let stats = index.stats();
    assert_eq!(stats.num_vectors, 100);
    assert_eq!(stats.dimension, 32);
    assert_eq!(stats.num_tables, 8);
    assert_eq!(stats.num_hashes, 16);
    assert!(stats.total_buckets > 0);
    assert!(stats.avg_bucket_size > 0.0);
    assert!(stats.max_bucket_size >= 1);
    assert!(stats.memory_estimate_bytes > 0);
}

// ---------------------------------------------------------------------------
// 15. Metrics collection when enabled
// ---------------------------------------------------------------------------

#[test]
fn test_metrics_collection() {
    let index = LshIndex::builder()
        .dim(16)
        .num_hashes(8)
        .num_tables(4)
        .seed(42)
        .enable_metrics()
        .build()
        .unwrap();

    let v = vec![1.0; 16];
    index.insert(0, &v).unwrap();
    index.insert(1, &v).unwrap();
    let _ = index.query(&v, 5).unwrap();

    let m = index.metrics().expect("metrics should be Some");
    assert_eq!(m.insert_count, 2);
    assert_eq!(m.query_count, 1);
    assert!(m.avg_query_time_us >= 0.0);
}

#[test]
fn test_metrics_disabled_by_default() {
    let index = make_index(16, 42);
    assert!(index.metrics().is_none());
}

// ---------------------------------------------------------------------------
// 16. Clear removes all vectors
// ---------------------------------------------------------------------------

#[test]
fn test_clear() {
    let index = make_index(8, 42);
    for i in 0..10 {
        index.insert(i, &[i as f32; 8]).unwrap();
    }
    assert_eq!(index.len(), 10);

    index.clear();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());

    // After clear, queries return nothing.
    let results = index.query(&[1.0; 8], 5).unwrap();
    assert!(results.is_empty());

    // insert_auto should start from 0 again.
    let id = index.insert_auto(&[1.0; 8]).unwrap();
    assert_eq!(id, 0);
}

// ---------------------------------------------------------------------------
// 17. suggest_params returns reasonable values
// ---------------------------------------------------------------------------

#[test]
fn test_suggest_params_returns_reasonable_values() {
    let params = suggest_params(0.9, 50_000, 128, DistanceMetric::Cosine);
    assert!(params.num_hashes >= 1 && params.num_hashes <= 64);
    assert!(params.num_tables >= 1);
    assert!(params.num_probes >= 1);
    assert!(
        params.estimated_recall >= 0.9,
        "estimated recall {} should be >= 0.9",
        params.estimated_recall
    );
}

#[test]
fn test_suggest_params_euclidean() {
    let params = suggest_params(0.8, 10_000, 256, DistanceMetric::Euclidean);
    assert!(params.num_hashes >= 1);
    assert!(params.num_tables >= 1);
    assert!(params.estimated_recall >= 0.8);
}

#[test]
fn test_suggest_params_high_recall_more_resources() {
    let low = suggest_params(0.7, 10_000, 128, DistanceMetric::Cosine);
    let high = suggest_params(0.95, 10_000, 128, DistanceMetric::Cosine);
    assert!(
        high.num_tables >= low.num_tables || high.num_probes >= low.num_probes,
        "higher recall should require more resources"
    );
}

// ---------------------------------------------------------------------------
// 18. estimate_recall increases with more tables
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_recall_increases_with_tables() {
    let r4 = estimate_recall(16, 4, 2, DistanceMetric::Cosine);
    let r8 = estimate_recall(16, 8, 2, DistanceMetric::Cosine);
    let r16 = estimate_recall(16, 16, 2, DistanceMetric::Cosine);
    assert!(r8 > r4, "r8={r8} should be > r4={r4}");
    assert!(r16 > r8, "r16={r16} should be > r8={r8}");
}

#[test]
fn test_estimate_recall_increases_with_probes() {
    let r0 = estimate_recall(16, 8, 0, DistanceMetric::Cosine);
    let r4 = estimate_recall(16, 8, 4, DistanceMetric::Cosine);
    assert!(r4 >= r0, "more probes should not decrease recall");
}

#[test]
fn test_estimate_recall_bounded() {
    let r = estimate_recall(16, 50, 10, DistanceMetric::Cosine);
    assert!((0.0..=1.0).contains(&r), "recall should be in [0, 1], got {r}");
}

// ---------------------------------------------------------------------------
// 19. Recall test: insert 1000 random vectors, compare LSH vs brute force
// ---------------------------------------------------------------------------

#[test]
fn test_recall_at_10() {
    let dim = 64;
    let n = 1000;
    let k = 10;
    let num_queries = 50;
    let mut rng = StdRng::seed_from_u64(2024);

    // Use fewer hash bits for higher collision probability, and many tables
    // with generous probing so that LSH achieves reasonable recall even on
    // uniformly random vectors (which are hard for any ANN method).
    let index = LshIndex::builder()
        .dim(dim)
        .num_hashes(4)
        .num_tables(40)
        .num_probes(4)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .build()
        .unwrap();

    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(&mut rng, dim)).collect();
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i, v).unwrap();
    }

    let mut total_recall = 0.0;

    for _ in 0..num_queries {
        let q = random_vector(&mut rng, dim);

        // Brute-force ground truth using cosine distance on raw vectors.
        let mut brute_force: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let denom = norm_q * norm_v;
                let cos_dist = if denom > f32::EPSILON {
                    1.0 - dot / denom
                } else {
                    1.0
                };
                (i, cos_dist)
            })
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: HashSet<usize> = brute_force.iter().take(k).map(|&(id, _)| id).collect();

        let lsh_results = index.query(&q, k).unwrap();
        let lsh_ids: HashSet<usize> = lsh_results.iter().map(|r| r.id).collect();

        let overlap = ground_truth.intersection(&lsh_ids).count();
        total_recall += overlap as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    // Random Gaussian vectors in 64-d are nearly equidistant, making ANN
    // inherently hard. We only require a modest recall threshold here.
    assert!(
        avg_recall >= 0.15,
        "average recall@{k} = {avg_recall:.3}, expected >= 0.15 for these parameters"
    );
}

// ---------------------------------------------------------------------------
// 20. Seeded index produces deterministic results
// ---------------------------------------------------------------------------

#[test]
fn test_seeded_determinism() {
    let dim = 32;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(&mut rng, dim)).collect();
    let query = random_vector(&mut rng, dim);

    let build_and_query = |seed: u64| -> Vec<QueryResult> {
        let index = LshIndex::builder()
            .dim(dim)
            .num_hashes(16)
            .num_tables(8)
            .num_probes(2)
            .distance_metric(DistanceMetric::Cosine)
            .seed(seed)
            .build()
            .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }
        index.query(&query, 10).unwrap()
    };

    let results_a = build_and_query(42);
    let results_b = build_and_query(42);

    assert_eq!(results_a.len(), results_b.len());
    for (a, b) in results_a.iter().zip(results_b.iter()) {
        assert_eq!(a.id, b.id, "IDs should match for deterministic seed");
        assert!(
            (a.distance - b.distance).abs() < 1e-9,
            "distances should match for deterministic seed"
        );
    }

    // Different seeds should (very likely) produce different results.
    let results_c = build_and_query(9999);
    let ids_a: Vec<usize> = results_a.iter().map(|r| r.id).collect();
    let ids_c: Vec<usize> = results_c.iter().map(|r| r.id).collect();
    // With different random projections the ordering will almost certainly differ.
    // We do not hard-assert inequality since it is theoretically possible (but astronomically
    // unlikely) for them to match.
    if ids_a == ids_c {
        eprintln!(
            "WARNING: same results with different seeds -- extremely unlikely but not impossible"
        );
    }
}

// ---------------------------------------------------------------------------
// Additional edge-case tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_k_larger_than_index() {
    let index = make_index(8, 42);
    index.insert(0, &[1.0; 8]).unwrap();
    index.insert(1, &[2.0; 8]).unwrap();

    let results = index.query(&[1.0; 8], 100).unwrap();
    assert!(results.len() <= 2, "cannot return more results than stored vectors");
}

#[test]
fn test_is_empty_and_len() {
    let index = make_index(8, 42);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    index.insert(0, &[1.0; 8]).unwrap();
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);
}

#[test]
fn test_config_returns_correct_values() {
    let index = LshIndex::builder()
        .dim(128)
        .num_hashes(20)
        .num_tables(10)
        .num_probes(5)
        .distance_metric(DistanceMetric::DotProduct)
        .normalize(false)
        .seed(12345)
        .build()
        .unwrap();

    let cfg = index.config();
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.num_hashes, 20);
    assert_eq!(cfg.num_tables, 10);
    assert_eq!(cfg.num_probes, 5);
    assert_eq!(cfg.distance_metric, DistanceMetric::DotProduct);
    assert!(!cfg.normalize_vectors);
    assert_eq!(cfg.seed, Some(12345));
}

#[test]
fn test_invalid_num_tables_zero() {
    let result = LshIndex::builder()
        .dim(32)
        .num_hashes(8)
        .num_tables(0)
        .build();
    match result {
        Err(ref e) => assert!(
            matches!(e, LshError::InvalidConfig(_)),
            "expected InvalidConfig for num_tables=0, got: {e:?}"
        ),
        Ok(_) => panic!("expected InvalidConfig error, got Ok"),
    }
}

#[test]
fn test_insert_auto_after_manual_insert() {
    let index = make_index(8, 42);
    // Manual insert at id 5.
    index.insert(5, &[1.0; 8]).unwrap();
    // insert_auto should pick id >= 6.
    let auto_id = index.insert_auto(&[2.0; 8]).unwrap();
    assert_eq!(auto_id, 6, "insert_auto should pick next_id after highest manual insert");
}

#[test]
fn test_remove_then_query_excludes_removed() {
    let index = make_index(16, 42);
    let mut rng = StdRng::seed_from_u64(7);

    for i in 0..20 {
        let v = random_vector(&mut rng, 16);
        index.insert(i, &v).unwrap();
    }

    index.remove(5).unwrap();
    index.remove(10).unwrap();

    let q = random_vector(&mut rng, 16);
    let results = index.query(&q, 20).unwrap();
    let ids: HashSet<usize> = results.iter().map(|r| r.id).collect();
    assert!(!ids.contains(&5), "removed id 5 should not appear in results");
    assert!(!ids.contains(&10), "removed id 10 should not appear in results");
}

#[test]
fn test_metrics_reset() {
    let index = LshIndex::builder()
        .dim(8)
        .num_hashes(8)
        .num_tables(4)
        .seed(42)
        .enable_metrics()
        .build()
        .unwrap();

    index.insert(0, &[1.0; 8]).unwrap();
    let _ = index.query(&[1.0; 8], 5).unwrap();

    let m = index.metrics().unwrap();
    assert_eq!(m.insert_count, 1);
    assert_eq!(m.query_count, 1);

    index.reset_metrics();
    let m2 = index.metrics().unwrap();
    assert_eq!(m2.insert_count, 0);
    assert_eq!(m2.query_count, 0);
}

#[test]
fn test_stats_display() {
    let index = make_index(8, 42);
    index.insert(0, &[1.0; 8]).unwrap();
    let stats = index.stats();
    let display = format!("{stats}");
    assert!(display.contains("vectors: 1"), "display should contain vector count");
    assert!(display.contains("dim: 8"), "display should contain dimension");
}

#[test]
fn test_boundary_num_hashes_1_and_64() {
    // num_hashes = 1 should be valid.
    let index = LshIndex::builder()
        .dim(8)
        .num_hashes(1)
        .num_tables(4)
        .seed(42)
        .build()
        .unwrap();
    index.insert(0, &[1.0; 8]).unwrap();
    let results = index.query(&[1.0; 8], 1).unwrap();
    assert!(!results.is_empty());

    // num_hashes = 64 should be valid.
    let index64 = LshIndex::builder()
        .dim(8)
        .num_hashes(64)
        .num_tables(4)
        .seed(42)
        .build()
        .unwrap();
    index64.insert(0, &[1.0; 8]).unwrap();
    let results64 = index64.query(&[1.0; 8], 1).unwrap();
    assert!(!results64.is_empty());
}

// ---------------------------------------------------------------------------
// 21. Persistence round-trip (JSON and bincode)
// ---------------------------------------------------------------------------

#[cfg(feature = "persistence")]
#[test]
fn test_persistence_json_round_trip() {
    let dir = std::env::temp_dir().join("lsh_test_json");
    let path = dir.with_extension("json");

    let index = LshIndex::builder()
        .dim(16)
        .num_hashes(4)
        .num_tables(4)
        .seed(42)
        .build()
        .unwrap();

    let mut rng = StdRng::seed_from_u64(1);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    for i in 0..100 {
        let v: Vec<f32> = (0..16).map(|_| normal.sample(&mut rng)).collect();
        index.insert(i, &v).unwrap();
    }

    // Save and reload.
    index.save_json(&path).unwrap();
    let loaded = LshIndex::load_json(&path).unwrap();

    assert_eq!(loaded.len(), 100);
    assert_eq!(loaded.config().dim, 16);
    assert!(loaded.contains(0));
    assert!(loaded.contains(99));

    // Queries should produce the same results.
    let q: Vec<f32> = (0..16).map(|_| normal.sample(&mut rng)).collect();
    let original_results = index.query(&q, 5).unwrap();
    let loaded_results = loaded.query(&q, 5).unwrap();
    assert_eq!(original_results.len(), loaded_results.len());
    for (a, b) in original_results.iter().zip(loaded_results.iter()) {
        assert_eq!(a.id, b.id);
        assert!((a.distance - b.distance).abs() < 1e-6);
    }

    // Clean up.
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "persistence")]
#[test]
fn test_persistence_bincode_round_trip() {
    let dir = std::env::temp_dir().join("lsh_test_bincode");
    let path = dir.with_extension("bin");

    let index = LshIndex::builder()
        .dim(16)
        .num_hashes(4)
        .num_tables(4)
        .seed(42)
        .build()
        .unwrap();

    let mut rng = StdRng::seed_from_u64(2);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    for i in 0..50 {
        let v: Vec<f32> = (0..16).map(|_| normal.sample(&mut rng)).collect();
        index.insert(i, &v).unwrap();
    }

    // Save and reload.
    index.save_bincode(&path).unwrap();
    let loaded = LshIndex::load_bincode(&path).unwrap();

    assert_eq!(loaded.len(), 50);
    assert!(loaded.contains(0));

    let q: Vec<f32> = (0..16).map(|_| normal.sample(&mut rng)).collect();
    let original_results = index.query(&q, 3).unwrap();
    let loaded_results = loaded.query(&q, 3).unwrap();
    assert_eq!(original_results.len(), loaded_results.len());

    // Clean up.
    let _ = std::fs::remove_file(&path);
}
