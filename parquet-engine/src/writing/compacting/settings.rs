//! The tunables of a compaction.

use std::sync::OnceLock;

use dispatch::RECORD_BATCH_SIZE;

/// The tunables of a compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeSortSettings {
    /// How many rows of the output one merged batch holds. The rows of an
    /// output row group are sent on one merged batch at a time.
    pub pull_rows: usize,
    /// How many bytes the compaction may hold at a time: fetched chunks,
    /// decompressed pages, decoded rows, and rows and pages being encoded
    /// or waiting to be stored.
    pub memory_bytes: usize,
    /// How many bytes of encoded column chunks are passed to the writer
    /// together. A row group's last column chunks are passed on however
    /// few they are.
    pub write_batch_bytes: usize,
}

impl MergeSortSettings {
    /// The settings from the environment:
    ///
    /// | variable | setting | default |
    /// |---|---|---|
    /// | `PIVOT_COMPACTION_PULL_ROWS` | `pull_rows` | 8192 (one record batch) |
    /// | `PIVOT_COMPACTION_MEMORY_BYTES` | `memory_bytes` | 1 GiB |
    /// | `PIVOT_COMPACTION_WRITE_BATCH_BYTES` | `write_batch_bytes` | 8 MiB |
    pub fn from_env() -> Self {
        static PULL_ROWS: OnceLock<usize> = OnceLock::new();
        static MEMORY_BYTES: OnceLock<usize> = OnceLock::new();
        static WRITE_BATCH_BYTES: OnceLock<usize> = OnceLock::new();
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
            write_batch_bytes: dispatch::env::get_tunable(
                &WRITE_BATCH_BYTES,
                "PIVOT_COMPACTION_WRITE_BATCH_BYTES",
                8 << 20,
                |bytes| bytes > 0,
            ),
        }
    }
}
