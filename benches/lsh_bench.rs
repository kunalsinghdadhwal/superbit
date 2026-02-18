use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use superbit::{DistanceMetric, LshIndex};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect()
}

fn brute_force_query(
    dataset: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    let q = Array1::from_vec(query.to_vec());
    let mut dists: Vec<(usize, f32)> = dataset
        .iter()
        .enumerate()
        .map(|(id, v)| {
            let arr = Array1::from_vec(v.clone());
            let d = metric.compute(&q.view(), &arr.view());
            (id, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

fn build_index(dim: usize) -> LshIndex {
    LshIndex::builder()
        .dim(dim)
        .num_hashes(16)
        .num_tables(8)
        .distance_metric(DistanceMetric::Cosine)
        .seed(42)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Insert throughput
// ---------------------------------------------------------------------------

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for &dim in &[128, 768] {
        for &n in &[1_000usize, 10_000] {
            let vecs = generate_vectors(n, dim, 99);

            group.bench_with_input(
                BenchmarkId::new(format!("dim={dim}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let index = build_index(dim);
                        for (id, v) in vecs.iter().enumerate() {
                            index.insert(id, v).unwrap();
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Single query latency
// ---------------------------------------------------------------------------

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    let k = 10;

    for &dim in &[128, 768] {
        for &n in &[1_000usize, 10_000, 100_000] {
            let vecs = generate_vectors(n, dim, 99);
            let query_vec = generate_vectors(1, dim, 1234)[0].clone();

            // --- LSH query ---
            let index = build_index(dim);
            for (id, v) in vecs.iter().enumerate() {
                index.insert(id, v).unwrap();
            }

            group.bench_with_input(
                BenchmarkId::new(format!("lsh/dim={dim}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        index.query(&query_vec, k).unwrap();
                    });
                },
            );

            // --- Brute-force linear scan ---
            group.bench_with_input(
                BenchmarkId::new(format!("brute/dim={dim}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        brute_force_query(&vecs, &query_vec, k, DistanceMetric::Cosine);
                    });
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Batch query (100 queries)
// ---------------------------------------------------------------------------

fn bench_batch_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_query_100");
    let k = 10;
    let num_queries = 100;

    for &dim in &[128, 768] {
        for &n in &[1_000usize, 10_000, 100_000] {
            let vecs = generate_vectors(n, dim, 99);
            let queries = generate_vectors(num_queries, dim, 5678);

            // Build the LSH index once.
            let index = build_index(dim);
            for (id, v) in vecs.iter().enumerate() {
                index.insert(id, v).unwrap();
            }

            group.bench_with_input(
                BenchmarkId::new(format!("lsh/dim={dim}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        for q in &queries {
                            index.query(q, k).unwrap();
                        }
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("brute/dim={dim}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        for q in &queries {
                            brute_force_query(&vecs, q, k, DistanceMetric::Cosine);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_insert, bench_query, bench_batch_query);
criterion_main!(benches);
