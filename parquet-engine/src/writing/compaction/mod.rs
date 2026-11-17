//! Merges a compaction's input files into one stream of encoded row groups,
//! holding only a few row groups in memory however large the files are.
//!
//! Every input file is already sorted by the table's sort columns. So the
//! merge only ever needs each file's current row group: the rows of the
//! file's later row groups come later in the output. The compaction reads
//! whole row groups, every column, one file's next row group whenever the
//! merge has used that file's rows up.
//!
//! ```text
//!                    ┌─────────────────────┐
//!   ┌───────────────▶│   fetch row groups  │  any worker
//!   │                └─────────────────────┘
//!   │                     │ fetched row groups, still compressed
//!   │                     ▼
//!   │       ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
//!   │         decompress & cut into ranges of rows     any worker
//!   │       └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
//!   │                     │ ranges of rows, as pages
//!   │                     ▼
//!   │                ┌─────────────────────┐
//!   │   ┌ ─ ─ ─ ─ ─ ▶│     DecodeGate      │  holds a range until the
//!   │                └─────────────────────┘  planner releases its rows
//!   │   │                 │
//!   │                     ▼
//!   │   │            ┌─────────────────────┐
//!   │                │       decode        │  any worker
//!   │   │            └─────────────────────┘
//!   │                     │ batches of rows
//!   │     releases        ▼
//!   │   │            ┌─────────────────────┐
//!   │    ─ ─ ─ ─ ─ ─ │    MergePlanner     │  holds each file's next rows;
//!   └────────────────└─────────────────────┘  releases a range while the
//!     requests for        │                   memory budget is not full
//!     row groups          │ merge jobs
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │  merge sort columns │  any worker
//!                    └─────────────────────┘
//!                         │ MergedRun
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │  RowGroupCollector  │
//!                    └─────────────────────┘
//!                         │ one job per column, or per slice of a
//!                         ▼ variant column
//!                    ┌─────────────────────┐
//!                    │       shredder      │  any worker; folds a variant
//!                    └─────────────────────┘  column back and shreds it anew
//!                         │ leaves, of a column or of a slice of one
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │  LeafSliceCombiner  │  joins a leaf's slices in row
//!                    └─────────────────────┘  order
//!                         │ leaves
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │       encoder       │  any worker
//!                    └─────────────────────┘
//!                         │ encoded leaves
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │  RowGroupAssembler  │  gives the rows' memory back;
//!                    └─────────────────────┘  reserves the encoded bytes
//!                         │ encoded row groups, in output order
//!                         ▼
//!                    ┌─────────────────────┐
//!                    │       writer        │  gives the encoded bytes'
//!                    └─────────────────────┘  memory back when they are stored
//! ```
//!
//! A solid box is one operator. The dashed box is the stages that
//! decompress row groups and cut them into ranges, drawn as one. The
//! operators not marked "any worker" run on one worker, the compaction
//! worker.
//!
//! 1. [`MergePlanner`] receives the rows of the input files, as batches
//!    whose last two columns say which rows of which row group they are
//!    ([`DecodedBatch`](input_batch::DecodedBatch)).
//!    It is also the one that asks for them: only it knows which file's
//!    rows the merge has used up. It does that in two steps. It asks for a
//!    file's next row group about [`READ_AHEAD_ROWS_PER_FILE`] rows ahead.
//!    That row group is fetched, decompressed and cut into ranges of rows,
//!    which wait, undecoded, at the `DecodeGate`. Then
//!    it releases a file's next ranges to be decoded only when the file
//!    has no rows ready to merge, which is when the merge is waiting for
//!    it, and only while the memory budget is not full. A row takes
//!    many times more memory decoded than as pages, and the merge needs
//!    only the next rows of each file. Rows that no row
//!    still to be read can sort before are safe to merge. The planner sends
//!    those out as merge jobs. A job holds the batches its rows are in and
//!    the merge of their sort columns that is still to be done, which any
//!    worker does. The result is a [`MergedRun`](merge::MergedRun): the next
//!    rows of the output, as rows of those batches.
//! 2. [`RowGroupCollector`] puts the runs in order and cuts them where
//!    output row groups end. It sends each complete row group out as one job
//!    per column. A job carries the column's part of every batch the row
//!    group takes rows from, and which row of which batch each output row is
//!    ([`GatherRows`]).
//! 3. The shredder, the encoder and the assembler are the ones every write
//!    uses. The shredder copies a column's rows into output order, a record
//!    batch at a time, on any worker. The rows are held as they were
//!    decoded, so a variant column still has each input file's shredding;
//!    the shredder folds the rows it takes back to plain variants and
//!    shreds them the way the output is shredded. That is the heavy part of
//!    a row group, so the collector sends a variant column as slices of
//!    [`VARIANT_COLUMN_SLICE_ROWS`] rows, one job each, and
//!    [`LeafSliceCombiner`] joins the slices' leaves in row order before
//!    they are encoded. The assembler hands the encoded row groups to the
//!    writer in output order.
//!
//! A table without sort columns has nothing to merge: its row groups are
//! read in turn and go out as they are. A row group made of consecutive rows
//! of its batches, one batch after the other, is sent on without the copy.
//!
//! One memory budget of [`MEMORY_BUDGET_BYTES`] bounds the rows that are in
//! memory decoded ([`MemoryReservation`]). The planner releases no range to
//! be decoded while the budget is full, and reserves a batch's size when
//! the batch arrives. The reservation is held by the batch and then by the
//! output row groups that take rows from it. When a row group is encoded,
//! its rows' reservations are given back, and it takes a small one for its
//! encoded bytes, which the writer gives back when they are stored. Rows
//! being decoded are not counted, so the budget can be passed by the
//! ranges being decoded. One range is also released although the budget is
//! full:
//! that of a file the merge cannot go on without, while no row group is
//! being encoded or written, since then nothing would give memory back. An
//! operator that stopped gets no input to start it again, so the
//! compaction worker is woken whenever memory is given back.
mod input_batch;
mod leaf_slice_combiner;
mod memory_budget;
mod merge;
mod row_group_collector;

use std::sync::Arc;

use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Topology, node_work_queue,
    return_to_worker_mpsc, to_single_worker_mpsc,
};

use super::assembler::{self, EncodedRowGroup};
use super::error::{WriteError, WriteResult};
use super::shredding::FileShredding;
use super::types::{ColumnChunkJob, EncodedLeafChunk, LeafChunkJob};
use super::{encoder, shredder};
use crate::ParquetTable;
use leaf_slice_combiner::LeafSliceCombiner;
use memory_budget::MemoryBudget;
use merge::{MergePlanner, MergedRun};
use row_group_collector::RowGroupCollector;

pub use memory_budget::MemoryReservation;
pub(super) use row_group_collector::GatherRows;

/// How many rows of one input file are read ahead of the ones released to be
/// decoded: fetched, decompressed and cut into ranges, so that the merge
/// does not wait for a read when it gets to them. Undecoded, a row takes a
/// fraction of what it takes decoded. A row group is asked for in one
/// request, which takes as long as the merge takes to go through most of a
/// row group, so a file's next row group is asked for while the one before
/// it is still whole.
const READ_AHEAD_ROWS_PER_FILE: usize = 16 * 16 * 1024;

/// How many ranges of rows of one input file are decoded at a time, when
/// the merge is waiting for rows of that file.
const RANGES_DECODED_AT_A_TIME_PER_FILE: usize = 2;

/// How many rows of an output row group's variant column one job folds back
/// and shreds. A row group's variant column is cut into slices of this many
/// rows, so that several workers shred it at once.
const VARIANT_COLUMN_SLICE_ROWS: usize = 16 * 1024;

/// How many bytes the rows of a compaction may hold decoded at a time, from
/// when they arrive decoded until the output row group they go to is
/// encoded. An encoded row group counts against it too, at its encoded
/// size, until it is written. The rows being decoded are not counted: at
/// most [`RANGES_DECODED_AT_A_TIME_PER_FILE`] ranges of each input file.
const MEMORY_BUDGET_BYTES: usize = 1 << 30;

/// A compaction of input files as a stream of encoded row groups in file order,
/// with what its writer needs to lay them out as files.
pub struct FileCompaction<OF: OperatorFactory<EncodedRowGroup>> {
    pub row_groups: OperatorSpec<EncodedRowGroup, OF>,
    /// The worker the compaction is planned on and its row groups are
    /// assembled on, where the writer runs too.
    pub worker: usize,
    /// How many rows the row groups hold in all.
    pub row_count: usize,
}

/// Merge the files of `table`, each sorted by `sort_column_names`, into
/// encoded row groups of `target_rows_per_group` rows, shredded as
/// `shredding` says. Fails for a table with deleted rows, which the compaction
/// does not read.
pub(super) fn compact_files_spec(
    dispatcher: &DataFlowDispatcher,
    table: Arc<ParquetTable>,
    shredding: Arc<FileShredding>,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
) -> WriteResult<FileCompaction<impl OperatorFactory<EncodedRowGroup> + 'static>> {
    if table
        .row_groups()
        .iter()
        .any(|row_group| row_group.deleted_rows.is_some())
    {
        return Err(WriteError::CompactionOfDeletedRows);
    }
    let worker_count = dispatcher.worker_count();
    let topology: Topology = dispatcher.topology();
    let compaction_worker = dispatcher.next_worker();
    let row_count: usize = table
        .row_groups()
        .iter()
        .map(|row_group| row_group.num_rows as usize)
        .sum();
    let memory_budget = MemoryBudget::new(dispatcher, MEMORY_BUDGET_BYTES, compaction_worker);

    let row_groups = MergePlanner::build_merged_runs_spec(
        dispatcher,
        &table,
        sort_column_names.clone(),
        &memory_budget,
        compaction_worker,
    )
    .chain(
        to_single_worker_mpsc::<MergedRun>(worker_count, compaction_worker)
            .into_iter()
            .collect(),
        RowGroupCollector::factories(
            topology,
            shredding,
            row_count,
            memory_budget,
            compaction_worker,
            target_rows_per_group,
            sort_column_names,
        ),
    )
    .chain(
        node_work_queue::<ColumnChunkJob>(topology)
            .into_iter()
            .collect(),
        shredder::factories(worker_count),
    )
    .chain(
        to_single_worker_mpsc::<LeafChunkJob>(worker_count, compaction_worker)
            .into_iter()
            .collect(),
        LeafSliceCombiner::factories(worker_count),
    )
    .chain(
        node_work_queue::<LeafChunkJob>(topology)
            .into_iter()
            .collect(),
        encoder::factories(worker_count),
    )
    .chain(
        return_to_worker_mpsc::<EncodedLeafChunk>(worker_count)
            .into_iter()
            .collect(),
        assembler::row_group_assembler_factories(worker_count),
    );
    Ok(FileCompaction {
        row_groups,
        worker: compaction_worker,
        row_count,
    })
}
