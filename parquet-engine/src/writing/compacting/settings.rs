//! The tunables of a compaction's read in merge order.

use std::sync::OnceLock;

use dispatch::RECORD_BATCH_SIZE;

/// The tunables of a compaction's read in merge order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeSortSettings {
    /// How many rows of the output one merged batch holds. The rows of an
    /// output row group are sent on one merged batch at a time.
    pub pull_rows: usize,
    /// How many bytes the read may hold at a time: fetched chunks,
    /// decompressed pages and decoded rows.
    pub memory_bytes: usize,
}

impl MergeSortSettings {
    /// The settings from the environment:
    ///
    /// | variable | setting | default |
    /// |---|---|---|
    /// | `PIVOT_COMPACTION_PULL_ROWS` | `pull_rows` | 8192 (one record batch) |
    /// | `PIVOT_COMPACTION_MEMORY_BYTES` | `memory_bytes` | 1 GiB |
    pub fn from_env() -> Self {
        static PULL_ROWS: OnceLock<usize> = OnceLock::new();
        static MEMORY_BYTES: OnceLock<usize> = OnceLock::new();
        Self {
            pull_rows: dispatch::env::get_tunable(
                &PULL_ROWS,
                "PIVOT_COMPACTION_PULL_ROWS",
                RECORD_BATCH_SIZE,
                |rows| rows > 0,
            ),
            memory_bytes: dispatch::env::get_tunable(
                &MEMORY_BYTES,
                "PIVOT_COMPACTION_MEMORY_BYTES",
                1 << 30,
                |bytes| bytes > 0,
            ),
        }
    }
}
