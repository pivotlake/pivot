//! Terminal (write) stage: the fan-in side of the dataflow. Every worker's
//! [`FileRowGroups`] arrive on worker 0, which accumulates them and — at `finish` —
//! hands the `Vec<FileRowGroups>` to the staging closure. Workers `1..n` receive
//! nothing (empty receiver, no commit) and no-op. Emits no rows.

use super::FileRowGroups;
use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory};
use std::mem;

/// Per-worker factory for [`FileRowGroupsSink`]. Only the worker-0 factory carries
/// the staging closure (the rest are `None`).
pub(super) struct FileRowGroupsSinkFactory<C> {
    stage: Option<C>,
}

impl<C> FileRowGroupsSinkFactory<C> {
    pub(super) fn new(stage: Option<C>) -> Self {
        Self { stage }
    }
}

impl<C> UnaryFactory<FileRowGroups, RecordBatch> for FileRowGroupsSinkFactory<C>
where
    C: FnOnce(Vec<FileRowGroups>) + Send + 'static,
{
    type Unary = FileRowGroupsSink<C>;

    fn build_unary(self) -> FileRowGroupsSink<C> {
        FileRowGroupsSink {
            files: vec![],
            stage: self.stage,
        }
    }
}

pub(super) struct FileRowGroupsSink<C> {
    files: Vec<FileRowGroups>,
    /// `Some` only on the receiving worker; `take`n so staging runs once even
    /// if `finish` is called more than once.
    stage: Option<C>,
}

impl<C> Unary<FileRowGroups, RecordBatch> for FileRowGroupsSink<C>
where
    C: FnOnce(Vec<FileRowGroups>) + Send + 'static,
{
    fn consume(
        &mut self,
        file: FileRowGroups,
        _sender: &mut dyn Sender<RecordBatch>,
    ) -> dispatch::UnaryResult<()> {
        self.files.push(file);
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<bool> {
        if let Some(stage) = self.stage.take() {
            stage(mem::take(&mut self.files));
        }
        Ok(true)
    }
}
