//! Terminal (write) stage: the fan-in side of the dataflow. Every worker's row
//! groups arrive on worker 0, which accumulates them and — at `finish` —
//! regroups them into the [`LoadedFiles`] and hands it to the `commit` closure.
//! Workers `1..n` receive nothing (empty receiver, no commit) and no-op. Emits
//! no rows.

use super::{IndexedRowGroup, LoadedFiles};
use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory};

/// Per-worker factory for [`TableBuildSink`]. Only the worker-0 factory carries
/// the `commit` (the rest are `None`).
pub(super) struct TableBuildSinkFactory<C> {
    commit: Option<C>,
    file_count: usize,
}

impl<C> TableBuildSinkFactory<C> {
    pub(super) fn new(commit: Option<C>, file_count: usize) -> Self {
        Self { commit, file_count }
    }
}

impl<C> UnaryFactory<IndexedRowGroup, RecordBatch> for TableBuildSinkFactory<C>
where
    C: FnOnce(LoadedFiles) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    type Unary = TableBuildSink<C>;

    fn build_unary(self) -> TableBuildSink<C> {
        TableBuildSink {
            rows: Vec::new(),
            commit: self.commit,
            file_count: self.file_count,
        }
    }
}

pub(super) struct TableBuildSink<C> {
    rows: Vec<IndexedRowGroup>,
    /// `Some` only on the receiving worker; `take`n so the commit runs once even
    /// if `finish` is called more than once.
    commit: Option<C>,
    file_count: usize,
}

impl<C> Unary<IndexedRowGroup, RecordBatch> for TableBuildSink<C>
where
    C: FnOnce(LoadedFiles) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        row_group: IndexedRowGroup,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        self.rows.push(row_group);
        Ok(())
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> dispatch::UnaryResult<bool> {
        if let Some(commit) = self.commit.take() {
            let loaded = LoadedFiles::assemble(std::mem::take(&mut self.rows), self.file_count);
            commit(loaded).map_err(|e| crate::parquet::op_err(CommitFailed(e)))?;
        }
        Ok(true)
    }
}

/// Adapts the boxed error a `commit` returns into a sized `Error` the
/// operator layer can carry.
#[derive(Debug)]
struct CommitFailed(Box<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for CommitFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CommitFailed {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}
