use lsh_vec_index::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
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
        .num_hashes(10)
        .num_tables(6)
        .num_probes(2)
        .distance_metric(DistanceMetric::Cosine)
        .seed(seed)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// 1. Concurrent insert + query stress test
//    Spawn 8 threads: 4 inserting, 4 querying simultaneously.
//    Insert 10,000 vectors total, query continuously.
//    Verify no panics, no data corruption, final len() is correct.
// ---------------------------------------------------------------------------

#[test]
fn stress_concurrent_insert_and_query() {
    let dim = 64;
    let vectors_per_thread = 2_500;
    let num_writer_threads = 4;
    let num_reader_threads = 4;

    let index = Arc::new(make_index(dim, 42));
    let done = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Writer threads: each inserts 2,500 vectors with non-overlapping IDs.
    for t in 0..num_writer_threads {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64);
            for i in 0..vectors_per_thread {
                let id = t * vectors_per_thread + i;
                let v = random_vector(&mut rng, dim);
                idx.insert(id, &v).unwrap();
            }
        }));
    }

    // Reader threads: query continuously until writers are done.
    for t in 0..num_reader_threads {
        let idx = Arc::clone(&index);
        let done_flag = Arc::clone(&done);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(100 + t as u64);
            let mut query_count = 0u64;
            while !done_flag.load(Ordering::Relaxed) {
                let q = random_vector(&mut rng, dim);
                let results = idx.query(&q, 10).unwrap();
                // Verify returned results have valid distances.
                for r in &results {
                    assert!(
                        r.distance.is_finite(),
                        "query returned non-finite distance: {}",
                        r.distance
                    );
                }
                query_count += 1;
                // Do not busy-spin forever if writers finish quickly.
                if query_count > 50_000 {
                    break;
                }
            }
        }));
    }

    // Wait for writers first, then signal readers.
    for h in handles.drain(..num_writer_threads) {
        h.join().expect("writer thread panicked");
    }
    done.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().expect("reader thread panicked");
    }

    let expected = num_writer_threads * vectors_per_thread;
    assert_eq!(
        index.len(),
        expected,
        "expected {} vectors, got {}",
        expected,
        index.len()
    );

    // Verify every inserted ID is present.
    for id in 0..expected {
        assert!(index.contains(id), "id {} should be present", id);
    }
}

// ---------------------------------------------------------------------------
// 2. Large-scale insert test
//    Insert 100,000 vectors of dim=128, verify len, stats, and query works.
// ---------------------------------------------------------------------------

#[test]
fn stress_large_scale_insert() {
    let dim = 128;
    let n = 100_000;

    let index = LshIndex::builder()
        .dim(dim)
        .num_hashes(12)
        .num_tables(4)
        .num_probes(1)
        .distance_metric(DistanceMetric::Cosine)
        .seed(7)
        .build()
        .unwrap();

    let mut rng = StdRng::seed_from_u64(123);

    for i in 0..n {
        let v = random_vector(&mut rng, dim);
        index.insert(i, &v).unwrap();
    }

    assert_eq!(index.len(), n);

    let stats = index.stats();
    assert_eq!(stats.num_vectors, n);
    assert_eq!(stats.dimension, dim);
    assert!(stats.total_buckets > 0);
    assert!(stats.memory_estimate_bytes > 0);

    // Query should return results.
    let q = random_vector(&mut rng, dim);
    let results = index.query(&q, 20).unwrap();
    assert!(
        !results.is_empty(),
        "querying 100k-vector index should return results"
    );
    // Results should be sorted by ascending distance.
    for pair in results.windows(2) {
        assert!(
            pair[0].distance <= pair[1].distance,
            "results not sorted: {} > {}",
            pair[0].distance,
            pair[1].distance
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Concurrent duplicate ID handling
//    Multiple threads insert with overlapping IDs.
//    Verify final state is consistent (each ID maps to exactly one vector).
// ---------------------------------------------------------------------------

#[test]
fn stress_concurrent_duplicate_ids() {
    let dim = 32;
    let num_threads = 8;
    let num_ids = 500; // Each thread writes to IDs 0..500

    let index = Arc::new(make_index(dim, 42));
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 1000);
            for id in 0..num_ids {
                let v = random_vector(&mut rng, dim);
                idx.insert(id, &v).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during concurrent duplicate insert");
    }

    // Each ID should be present exactly once.
    assert_eq!(
        index.len(),
        num_ids,
        "expected {} unique IDs, got {}",
        num_ids,
        index.len()
    );

    for id in 0..num_ids {
        assert!(index.contains(id), "id {} should be present", id);
    }

    // Querying should work without issues.
    let mut rng = StdRng::seed_from_u64(999);
    let q = random_vector(&mut rng, dim);
    let results = index.query(&q, 10).unwrap();
    // All returned IDs should be in the valid range.
    for r in &results {
        assert!(r.id < num_ids, "returned id {} out of range", r.id);
    }
}

// ---------------------------------------------------------------------------
// 4. Remove under concurrent reads
//    Insert 1000 vectors, spawn readers and removers concurrently.
//    Verify no panics.
// ---------------------------------------------------------------------------

#[test]
fn stress_remove_under_concurrent_reads() {
    let dim = 32;
    let n = 1_000;

    let index = Arc::new(make_index(dim, 42));

    // Pre-populate.
    let mut rng = StdRng::seed_from_u64(10);
    for i in 0..n {
        let v = random_vector(&mut rng, dim);
        index.insert(i, &v).unwrap();
    }
    assert_eq!(index.len(), n);

    let done = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Reader threads: query continuously.
    for t in 0..4 {
        let idx = Arc::clone(&index);
        let done_flag = Arc::clone(&done);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(200 + t as u64);
            let mut count = 0u64;
            while !done_flag.load(Ordering::Relaxed) {
                let q = random_vector(&mut rng, dim);
                // Query should never panic, even if vectors are being removed.
                let _results = idx.query(&q, 10).unwrap();
                count += 1;
                if count > 50_000 {
                    break;
                }
            }
        }));
    }

    // Remover threads: remove IDs from different ranges.
    for t in 0..4 {
        let idx = Arc::clone(&index);
        let start = t * (n / 4);
        let end = start + (n / 4);
        handles.push(thread::spawn(move || {
            for id in start..end {
                // Ignore NotFound errors -- another thread may have removed it
                // or the ID may not exist in a particular range partition.
                let _ = idx.remove(id);
            }
        }));
    }

    // Wait for removers first.
    // handles[4..8] are removers, handles[0..4] are readers.
    // We drain removers, then signal readers.
    let reader_handles: Vec<_> = handles.drain(..4).collect();
    for h in handles {
        h.join().expect("remover thread panicked");
    }
    done.store(true, Ordering::Relaxed);
    for h in reader_handles {
        h.join().expect("reader thread panicked");
    }

    // All vectors should have been removed.
    assert_eq!(
        index.len(),
        0,
        "all vectors should be removed, but {} remain",
        index.len()
    );
}

// ---------------------------------------------------------------------------
// 5. Rapid insert-remove cycles
//    Repeatedly insert and remove the same ID from multiple threads.
//    Verify consistency.
// ---------------------------------------------------------------------------

#[test]
fn stress_rapid_insert_remove_cycles() {
    let dim = 16;
    let num_threads = 8;
    let cycles_per_thread = 1_000;

    let index = Arc::new(make_index(dim, 42));
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64);
            // Each thread uses its own ID to avoid cross-thread conflicts on remove.
            let my_id = t;
            for _ in 0..cycles_per_thread {
                let v = random_vector(&mut rng, dim);
                idx.insert(my_id, &v).unwrap();
                // Remove might fail if another operation interleaved, but with
                // unique IDs per thread it should always succeed.
                let _ = idx.remove(my_id);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during rapid insert-remove");
    }

    // After all cycles, each thread's last operation was remove, so index
    // should be empty (each thread inserted then removed its own ID).
    assert_eq!(
        index.len(),
        0,
        "index should be empty after insert-remove cycles, but has {} vectors",
        index.len()
    );
}

// ---------------------------------------------------------------------------
// 5b. Rapid insert-remove cycles with shared IDs
//     Multiple threads fight over the same set of IDs.
//     Verify no panics and that the index is in a consistent state.
// ---------------------------------------------------------------------------

#[test]
fn stress_rapid_insert_remove_shared_ids() {
    let dim = 16;
    let num_threads = 8;
    let cycles_per_thread = 500;
    let shared_ids = 10; // Only 10 IDs shared across all threads.

    let index = Arc::new(make_index(dim, 42));
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 31);
            for cycle in 0..cycles_per_thread {
                let id = cycle % shared_ids;
                let v = random_vector(&mut rng, dim);
                idx.insert(id, &v).unwrap();
                // Sometimes remove, sometimes leave it.
                if cycle % 3 == 0 {
                    let _ = idx.remove(id); // Ignore NotFound
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during shared-ID insert-remove");
    }

    // The index should be in a consistent state: len() <= shared_ids.
    let final_len = index.len();
    assert!(
        final_len <= shared_ids,
        "index has {} vectors but only {} IDs exist",
        final_len,
        shared_ids
    );

    // Every ID that is reportedly present should be queryable.
    let mut rng = StdRng::seed_from_u64(42);
    let q = random_vector(&mut rng, dim);
    let results = index.query(&q, shared_ids).unwrap();
    let result_ids: HashSet<usize> = results.iter().map(|r| r.id).collect();
    for id in 0..shared_ids {
        if index.contains(id) {
            // The query might or might not find it (LSH is approximate),
            // but the index should at least not panic.
            let _ = result_ids.contains(&id);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Edge case: query immediately after clear
//    One thread clears, another queries simultaneously.
//    Should not panic.
// ---------------------------------------------------------------------------

#[test]
fn stress_query_after_clear() {
    let dim = 32;
    let n = 500;

    let index = Arc::new(make_index(dim, 42));

    // Pre-populate.
    let mut rng = StdRng::seed_from_u64(55);
    for i in 0..n {
        let v = random_vector(&mut rng, dim);
        index.insert(i, &v).unwrap();
    }

    let done = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Reader threads.
    for t in 0..4 {
        let idx = Arc::clone(&index);
        let done_flag = Arc::clone(&done);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(300 + t as u64);
            let mut count = 0u64;
            while !done_flag.load(Ordering::Relaxed) {
                let q = random_vector(&mut rng, dim);
                // Must not panic regardless of clear happening concurrently.
                let _results = idx.query(&q, 10).unwrap();
                count += 1;
                if count > 50_000 {
                    break;
                }
            }
        }));
    }

    // Clear + re-insert thread.
    {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(400);
            for _ in 0..20 {
                idx.clear();
                // Re-insert some vectors.
                for i in 0..100 {
                    let v = random_vector(&mut rng, dim);
                    idx.insert(i, &v).unwrap();
                }
            }
        }));
    }

    // Wait for clear/insert thread.
    let clear_handle = handles.pop().unwrap();
    clear_handle.join().expect("clear thread panicked");

    // Signal readers to stop.
    done.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("reader thread panicked after clear");
    }

    // Index should be in a valid state.
    let len = index.len();
    assert!(
        len <= 100,
        "after final re-insert of 100, len should be <= 100, got {}",
        len
    );

    // One more query to verify consistency.
    let q = random_vector(&mut rng, dim);
    let results = index.query(&q, 10).unwrap();
    for r in &results {
        assert!(r.distance.is_finite());
    }
}
