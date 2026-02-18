//! # lsh_vec_index
//!
//! A lightweight, in-memory vector index for approximate nearest-neighbor (ANN)
//! search using Locality-Sensitive Hashing (LSH).
//!
//! Designed for prototyping ML applications such as retrieval-augmented
//! generation (RAG) or recommendation systems, without the overhead of a full
//! vector database.
//!
//! ## Quick start
//!
//! ```rust
//! use lsh_vec_index::{LshIndex, DistanceMetric};
//!
//! let index = LshIndex::builder()
//!     .dim(128)
//!     .num_hashes(16)
//!     .num_tables(8)
//!     .distance_metric(DistanceMetric::Cosine)
//!     .seed(42)
//!     .build()
//!     .unwrap();
//!
//! // Insert a vector.
//! let v = vec![0.1_f32; 128];
//! index.insert(0, &v).unwrap();
//!
//! // Query for similar vectors.
//! let results = index.query(&v, 5).unwrap();
//! for r in &results {
//!     println!("id={} dist={:.4}", r.id, r.distance);
//! }
//! ```
//!
//! ## Feature flags
//!
//! | Flag          | Effect                                       |
//! |---------------|----------------------------------------------|
//! | `parallel`    | Parallel bulk insert/query via rayon          |
//! | `persistence` | Save/load index to disk (serde + bincode)     |
//! | `python`      | Python bindings via PyO3                      |
//! | `full`        | Enables `parallel` + `persistence`            |

pub mod distance;
pub mod error;
pub mod hash;
pub mod index;
pub mod metrics;
pub mod tuning;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenience.
pub use distance::DistanceMetric;
pub use error::{LshError, Result};
pub use index::{IndexConfig, IndexStats, LshIndex, LshIndexBuilder, QueryResult};
pub use metrics::{MetricsCollector, MetricsSnapshot};
pub use tuning::{estimate_recall, suggest_params, SuggestedParams};
