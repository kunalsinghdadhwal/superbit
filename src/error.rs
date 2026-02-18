use thiserror::Error;

/// Errors that can occur when using the LSH index.
#[derive(Debug, Error)]
pub enum LshError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("index is empty")]
    EmptyIndex,

    #[error("vector not found: id={0}")]
    NotFound(usize),

    #[error("num_hashes must be between 1 and 64, got {0}")]
    InvalidNumHashes(usize),

    #[error("dimension must be greater than 0")]
    ZeroDimension,

    #[cfg(feature = "persistence")]
    #[error("serialization error: {0}")]
    Serialization(String),

    #[cfg(feature = "persistence")]
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A specialized Result type for LSH index operations.
pub type Result<T> = std::result::Result<T, LshError>;
