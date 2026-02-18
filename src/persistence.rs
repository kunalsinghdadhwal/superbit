//! Save and load the index to/from disk.
//!
//! Requires the `persistence` feature flag.

use std::path::Path;

use parking_lot::RwLock;

use crate::error::{LshError, Result};
use crate::index::{IndexInner, LshIndex};

impl LshIndex {
    /// Serialize the index to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let inner = self.inner.read();
        let json = serde_json::to_string_pretty(&*inner)
            .map_err(|e| LshError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialize an index from a JSON file.
    pub fn load_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let inner: IndexInner =
            serde_json::from_str(&data).map_err(|e| LshError::Serialization(e.to_string()))?;
        Ok(Self {
            inner: RwLock::new(inner),
            metrics: None,
        })
    }

    /// Serialize the index to a compact bincode file.
    pub fn save_bincode(&self, path: &Path) -> Result<()> {
        let inner = self.inner.read();
        let bytes = bincode::serialize(&*inner)
            .map_err(|e| LshError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Deserialize an index from a bincode file.
    pub fn load_bincode(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let inner: IndexInner =
            bincode::deserialize(&data).map_err(|e| LshError::Serialization(e.to_string()))?;
        Ok(Self {
            inner: RwLock::new(inner),
            metrics: None,
        })
    }
}
