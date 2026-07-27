//! Terminal (write) stage: the fan-in side of the dataflow. Every worker's
//! [`TableFile`]s arrive on worker 0, which accumulates them and — at `finish` —
//! hands the `Vec<TableFile>` to the staging closure. Workers `1..n` receive
//! nothing (empty receiver, no commit) and no-op. Emits no rows.

use crate::catalog::TableFile;
use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory};
use std::mem;

/// Per-worker factory for [`TableBuildSink`]. Only the worker-0 factory carries
/// the staging closure (the rest are `None`).
pub(super) struct TableBuildSinkFactory<C> {
    stage: Option<C>,
}

impl<C> TableBuildSinkFactory<C> {
    pub(super) fn new(stage: Option<C>) -> Self {
        Self { stage }
    }
}

impl<C> UnaryFactory<TableFile, RecordBatch> for TableBuildSinkFactory<C>
where
    C: FnOnce(Vec<TableFile>) + Send + 'static,
{
    type Unary = TableBuildSink<C>;

    fn build_unary(self) -> TableBuildSink<C> {
        TableBuildSink {
            table_files: vec![],
            stage: self.stage,
        }
    }
}

pub(super) struct TableBuildSink<C> {
    table_files: Vec<TableFile>,
    /// `Some` only on the receiving worker; `take`n so staging runs once even
    /// if `finish` is called more than once.
    stage: Option<C>,
}

impl<C> Unary<TableFile, RecordBatch> for TableBuildSink<C>
where
    C: FnOnce(Vec<TableFile>) + Send + 'static,
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
        if let Some(stage) = self.stage.take() {
            stage(mem::take(&mut self.table_files));
        }
        Ok(true)
    }
}
