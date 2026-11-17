//! Merges the input files' rows into the output's row order.
//!
//! Every input file is sorted, so the merge only ever needs each file's
//! current row group: the rows of the others come later in the output.
//!
//! - [`InputFileRows`](input_file_rows::InputFileRows) keeps one input
//!   file's rows: which row groups are still to be asked for, which are
//!   arriving, and which rows have arrived and are not merged yet.
//! - [`DecodeGate`](decode_gate::DecodeGate) holds the rows of a row group
//!   that was read ahead, undecoded, until the planner releases them to be
//!   decoded, a range at a time.
//! - [`MergePlanner`] is the operator that holds the files' rows, on one
//!   worker. Whenever every file with rows still to come has some ready,
//!   the rows whose sort columns sort at or before the smallest of those
//!   files' last arrived sort columns can no longer be preceded by a row
//!   not read yet. It takes those rows out of every file and plans the
//!   k-way merge of their sort columns as [`MergeJob`]s. Any worker then
//!   merges a job into a [`MergedRun`]: the next rows of the output, as
//!   rows of the batches they were read in.

mod decode_gate;
mod input_file_rows;
mod planner;

pub(super) use planner::MergePlanner;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::memory::SlabAllocator;
use dispatch::{KWayMergeTask, UnaryResult};

use super::input_batch::InputBatch;
use crate::writing::error::{WriteError, WriteResult};

/// Which rows of a job's source batches are to be merged.
enum RowsToMerge {
    /// A slice of a planned merge of sort columns. Each key in it is
    /// followed by two columns that say which row it is: the number of its
    /// source batch, and its row there.
    Slice(KWayMergeTask),
    /// Rows that are already in output order: consecutive rows of one
    /// source batch.
    Consecutive {
        source: u32,
        first_row: u32,
        row_count: usize,
    },
}

/// The next rows of the output, still to be put in order.
pub(super) struct MergeJob {
    /// The job's place among the jobs, which are numbered in output order.
    sequence: usize,
    /// The batches the job's rows are in, every column.
    sources: Arc<[InputBatch]>,
    rows: RowsToMerge,
}

impl MergeJob {
    /// The merge's form of one source batch: `keys`, the batch's sort
    /// columns, followed by two columns that say which row each key is, the
    /// number `source` of the batch among its job's source batches and the
    /// key's row there. The merge compares the sort columns and carries the
    /// other two along.
    fn label_keys(keys: RecordBatch, source: u32) -> WriteResult<RecordBatch> {
        let mut fields: Vec<Arc<Field>> = keys.schema().fields().to_vec();
        fields.push(Arc::new(Field::new("source", DataType::UInt32, false)));
        fields.push(Arc::new(Field::new("row", DataType::UInt32, false)));
        let mut columns: Vec<ArrayRef> = keys.columns().to_vec();
        columns.push(Arc::new(UInt32Array::from(vec![source; keys.num_rows()])));
        columns.push(Arc::new(UInt32Array::from_iter_values(
            0..keys.num_rows() as u32,
        )));
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    /// For `keys`, labelled keys that are consecutive rows of one source
    /// batch: which rows of which batch they are, read from the labels of
    /// the first.
    fn read_consecutive_rows(keys: &RecordBatch) -> RowsToMerge {
        let column_count = keys.num_columns();
        let source = keys
            .column(column_count - 2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("labelled keys are followed by their source batch");
        let row = keys
            .column(column_count - 1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("labelled keys end with their row in the source batch");
        RowsToMerge::Consecutive {
            source: source.value(0),
            first_row: row.value(0),
            row_count: keys.num_rows(),
        }
    }

    fn row_count(&self) -> usize {
        match &self.rows {
            RowsToMerge::Slice(task) => task.row_count(),
            RowsToMerge::Consecutive { row_count, .. } => *row_count,
        }
    }

    /// Merge the job's sort columns and keep what the merge shows: which row
    /// of which source batch each output row is. The merged sort columns
    /// themselves are dropped. Runs on any worker, with that worker's
    /// `allocator`.
    fn merge_into_run(self, allocator: &mut Option<SlabAllocator>) -> UnaryResult<MergedRun> {
        let rows = match self.rows {
            RowsToMerge::Consecutive {
                source,
                first_row,
                row_count,
            } => MergedRows::Consecutive {
                source,
                first_row,
                row_count,
            },
            RowsToMerge::Slice(task) => {
                // Built on the first slice: a slab allocator takes a write
                // buffer from the running worker's memory context.
                let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
                let merged_keys = task
                    .merge_slice(allocator)
                    .map_err(WriteError::from)?
                    .into_batch();
                let column_count = merged_keys.num_columns();
                let sources = merged_keys
                    .column(column_count - 2)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .expect("merged keys are followed by their source batch");
                let rows = merged_keys
                    .column(column_count - 1)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .expect("merged keys end with their row in the source batch");
                MergedRows::Listed {
                    sources: sources.clone(),
                    rows: rows.clone(),
                }
            }
        };
        Ok(MergedRun {
            sequence: self.sequence,
            sources: self.sources,
            rows,
        })
    }
}

/// Which row of which source batch each row of a run is.
pub(super) enum MergedRows {
    /// Row `i` of the run is row `rows[i]` of source batch `sources[i]`.
    Listed {
        sources: UInt32Array,
        rows: UInt32Array,
    },
    /// The run is consecutive rows of one source batch.
    Consecutive {
        source: u32,
        first_row: u32,
        row_count: usize,
    },
}

/// The next rows of the output, in order, as rows of the batches they were
/// read in. Runs are numbered in output order.
pub(super) struct MergedRun {
    pub(super) sequence: usize,
    /// The batches the run's rows are in, every column.
    pub(super) sources: Arc<[InputBatch]>,
    pub(super) rows: MergedRows,
}

impl MergedRun {
    pub(super) fn row_count(&self) -> usize {
        match &self.rows {
            MergedRows::Listed { rows, .. } => rows.len(),
            MergedRows::Consecutive { row_count, .. } => *row_count,
        }
    }
}
