//! Terminal (write) stage: the fan-in side of the dataflow. Every worker's row
//! groups arrive on worker 0, which accumulates them and — at `finish` —
//! regroups them into the [`LoadedFiles`] and hands it to the `commit` closure.
//! Workers `1..n` receive nothing (empty receiver, no commit) and no-op. Emits
//! no rows.

use std::mem;
use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory};
use crate::catalog::TableFile;

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

impl<C> UnaryFactory<TableFile, RecordBatch> for TableBuildSinkFactory<C>
where
    C: FnOnce(Vec<TableFile>) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    type Unary = TableBuildSink<C>;

    fn build_unary(self) -> TableBuildSink<C> {
        TableBuildSink {
            table_files: vec![],
            commit: self.commit,
        }
    }
}

pub(super) struct TableBuildSink<C> {
    table_files: Vec<TableFile>,
    /// `Some` only on the receiving worker; `take`n so the commit runs once even
    /// if `finish` is called more than once.
    commit: Option<C>,
}

impl<C> Unary<TableFile, RecordBatch> for TableBuildSink<C>
where
    C: FnOnce(Vec<TableFile>) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        + Send
        + 'static,
{
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        table_file: TableFile,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        self.table_files.push(table_file);
        Ok(())
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> dispatch::UnaryResult<bool> {
        if let Some(commit) = self.commit.take() {
            commit(mem::take(&mut self.table_files)).map_err(|e| crate::parquet::op_err(CommitFailed(e)))?;
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
