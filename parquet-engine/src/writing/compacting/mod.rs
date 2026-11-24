//! The new compaction: rewriting several sorted input files of a table as
//! fewer, larger sorted files while holding only what is about to be used.
//!
//! So far this is the read half. [`read_in_merge_order_spec`] reads the
//! inputs in the order of the output and produces the decoded rows in merge
//! order. Nothing is encoded or written yet.
//!
//! The compaction that tables run today is not in this module. It decodes
//! every input row group whole and merges the decoded batches (see
//! `encode_compaction_batches_spec` in the parent module).
//!
//! # The read in merge order
//!
//! The read fetches, decompresses and decodes with the operators a scan
//! uses. It adds two gates that hold their input back, and the compacter
//! that decides what is requested and what the gates let through.
//!
//! The dataflow, one box per operator. A solid arrow is the channel between
//! two operators, labelled with the type it carries. A dashed arrow is the
//! compacter putting requests into the request queue, or releasing rows at
//! a gate.
//!
//! ```text
//!                 ┌──────────────────────────┐
//!   ┌ ─ ─ ─ ─ ─ ─▶│      request queue       │  source of the dataflow
//!                 └──────────────────────────┘
//!   │                          │ QueryRowGroupMetadata:
//!                              ▼   one column of one input row group
//!   │             ┌──────────────────────────┐
//!                 │     RowGroupFetcher      │  any worker; reads the
//!   │             └──────────────────────────┘  column's chunk from disk
//!                              │ RowGroupBuffer
//!   │                          ▼
//!                 ┌──────────────────────────┐
//!   │             │         Indexer          │  any worker; splits the
//!                 └──────────────────────────┘  chunk into pages
//!   │                          │ CompressedPage
//!                              ▼
//!   │             ┌──────────────────────────┐
//!     ┌ ─ ─ ─ ─ ─▶│      DecompressGate      │  any worker; holds a page
//!   │             └──────────────────────────┘  until a row of it is released
//!     │                        │ CompressedPage
//!   │                          ▼
//!     │           ┌──────────────────────────┐
//!   │             │       Decompressor       │  any worker
//!     │           └──────────────────────────┘
//!   │                          │ DecompressedPage
//!     │                        ▼
//!   │             ┌──────────────────────────┐  any worker; makes the budget
//!     │           │    PageMemoryCounter     │  count the page until nothing
//!   │             └──────────────────────────┘  holds its bytes
//!     │                        │ DecompressedPage
//!   │                          ▼
//!   │             ┌──────────────────────────┐  the worker that fetched;
//!     │           │       RangeCutter        │  builds the dictionary, cuts
//!   │             └──────────────────────────┘  ranges of 16K rows
//!     │                        │ DecodeRange
//!   │                          ▼
//!     │           ┌──────────────────────────┐
//!   │ ├ ─ ─ ─ ─ ─▶│        DecodeGate        │  any worker; holds a range
//!     │           └──────────────────────────┘  until it is released
//!   │                          │ DecodeRange
//!     │                        ▼
//!   │             ┌──────────────────────────┐
//!     │           │         Decoder          │  any worker
//!   │             └──────────────────────────┘
//!     │                        │ RecordBatch, then DecodedValues
//!   │                          ▼
//!     │           ┌──────────────────────────┐
//!   └ ┴ ─ ─ ─ ─ ─ │    MergeSortCompacter    │  the compaction worker
//!                 └──────────────────────────┘
//!                              │ MergedBatchColumn:
//!                              │   per merged batch and column, the
//!                              │   decoded batches and which of their rows
//!                              ▼   each row of the merged batch is
//! ```
//!
//! Every column is read on its own. A *read* is one column of one input row
//! group. It is requested once and its chunk is fetched whole. Its pages
//! then wait at the `DecompressGate`, and are decompressed and decoded a
//! range of rows at a time as the compacter releases its rows. How many
//! columns are read at once is not decided up front: the compacter starts a
//! column's rows when the memory budget has room for them.
//!
//! Inside the compacter, each decision is made in one layer, from the top
//! down:
//!
//! - [`compacter`]: the `MergeSortCompacter` operator. It only calls the
//!   compaction read.
//! - [`compaction_read`]: decides which reads are requested and which rows
//!   are released, in six steps it repeats (its module doc lists them). It
//!   orders the rows of one output row group at a time and sends them on a
//!   *merged batch* at a time: the next
//!   [`pull_rows`](MergeSortSettings::pull_rows) rows of the output.
//!   - [`key_merge`] decides the order for it, from the files' sort
//!     columns. From the row groups' min/max statistics it finds the row
//!     group that starts first and those that overlap it, and merges those
//!     by their decoded keys.
//!   - [`output_row_group`] holds the order of one output row group, and
//!     how far each column of it has got.
//! - [`input_file`]: one input file with the reads of its columns. It is
//!   told which columns to request and which rows of them to release, and
//!   keeps the decoded values until they are sent on.
//! - [`read_requests`]: what the files share to send their requests and
//!   releases into the pipeline.
//! - [`gated_reads`], [`decompress_gate`], [`decode_gate`]: the two gates,
//!   and how the compacter releases rows to them.
//! - [`memory_budget`]: the count of every byte the read holds. What can
//!   outlive the compacter carries its count with it and gives it back
//!   when it is dropped: a decompressed page (see [`page_memory_counter`])
//!   and decoded values (see [`InputValues`]).
//!
//! [`settings`] holds the tunables, and [`output`] what the read produces.

mod compacter;
mod compaction_read;
mod decode_gate;
mod decoded_values;
mod decompress_gate;
mod gated_reads;
mod input_file;
mod key_merge;
mod memory_budget;
mod output;
mod output_row_group;
mod page_memory_counter;
mod read_requests;
mod settings;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection, RECORD_BATCH_SIZE,
    channel_input, return_to_worker_mpsc, shared_work_queue, stealable, stealable_fifo,
    to_single_worker_mpsc,
};

use crate::types::metadata::QueryRowGroupMetadata;
use crate::writing::error::WriteError;
use crate::{
    CompressedPage, DecodeRange, DecoderFactory, DecompressedPage, DecompressorFactory,
    IndexerFactory, RangeCutterFactory, RowGroupBuffer, RowGroupFetcherFactory, WorkerAllocator,
};
use compacter::MergeSortCompacter;
use compaction_read::CompactionRead;
use decode_gate::DecodeGate;
use decoded_values::DecodedValues;
use decompress_gate::DecompressGate;
use gated_reads::GatedReads;
use memory_budget::MemoryBudget;
use page_memory_counter::PageMemoryCounter;
use read_requests::ReadRequests;

pub use memory_budget::MemoryReservation;
pub use output::{InputValues, MergedBatchColumn, MergedRows};
pub use settings::MergeSortSettings;

/// Builds the dataflow that reads the files of `table`, each already sorted
/// by `sort_column_names`, in merge order, for output row groups of
/// `output_row_group_rows` rows. The [module](self) has the diagram of the
/// operators chained here.
///
/// The dataflow's output is one [`MergedBatchColumn`] per column of each
/// merged batch. Fails for a table that has deleted rows.
pub fn read_in_merge_order_spec(
    dispatcher: &DataFlowDispatcher,
    table: Arc<crate::ParquetTable>,
    sort_column_names: &[String],
    output_row_group_rows: usize,
    settings: MergeSortSettings,
) -> Result<
    OperatorSpec<MergedBatchColumn, impl OperatorFactory<MergedBatchColumn> + 'static>,
    dispatch::DataFlowError,
> {
    let into_dataflow_error = |error: WriteError| {
        dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into())
    };
    if table
        .row_groups()
        .iter()
        .any(|row_group| row_group.deleted_rows.is_some())
    {
        return Err(into_dataflow_error(WriteError::CompactionOfDeletedRows));
    }
    let worker_count = dispatcher.worker_count();
    let topology = dispatcher.topology();
    let compaction_worker = dispatcher.next_worker();

    let sort_columns: Vec<usize> = sort_column_names
        .iter()
        .map(|name| table.schema().index_of(name))
        .collect::<Result<_, _>>()
        .map_err(|error| into_dataflow_error(error.into()))?;
    let memory_budget = MemoryBudget::new(
        dispatcher.waker_set(),
        settings.memory_bytes,
        compaction_worker,
    );
    let row_group_count = table.row_groups().len();
    let column_count = table.schema().fields().len();
    let gated_reads = GatedReads::new(row_group_count, column_count, worker_count);
    // The queue has room for every read, one per column of every row group,
    // so a request is never refused.
    let (request_sender, requests) = channel_input::<QueryRowGroupMetadata>(
        dispatcher,
        (row_group_count * column_count).max(1),
        Box::new(|| {}),
    );
    let read_medium = table.get_read_medium();
    // Chunks are fetched ahead within a quarter of the budget, and within
    // what the workers may have asked of the files' medium at a time:
    // more than is delivered in the time one read takes would only wait.
    let fetch_ahead_bytes =
        (settings.memory_bytes / 4).min(read_medium.count_read_ahead_bytes(worker_count));
    let read_requests = ReadRequests::new(
        table.clone(),
        gated_reads.clone(),
        memory_budget.clone(),
        request_sender,
        fetch_ahead_bytes,
    );
    let compaction_read = CompactionRead::new(
        &table,
        sort_columns,
        settings,
        output_row_group_rows,
        read_requests,
    )
    .map_err(into_dataflow_error)?;

    // Every request names its own column, so the columns the fetchers and
    // the range cutters read from a row group that names none are never
    // used.
    let no_columns = Projection::columns([]);
    let shared_no_columns = Arc::new(no_columns.clone());
    // One allocator per worker for the dictionaries its range cutter
    // builds, and one for the batches its decoder emits: a dictionary lives
    // as long as its read, a batch until its rows are sent on.
    let create_worker_allocators = || -> Vec<Arc<WorkerAllocator>> {
        (0..worker_count)
            .map(|_| Arc::new(WorkerAllocator::new()))
            .collect()
    };
    Ok(requests
        .chain(
            stealable_fifo::<QueryRowGroupMetadata>(topology)
                .into_iter()
                .collect(),
            // Every request arrives on the fetcher's input. One that needs
            // a read goes to the second queue, which every worker's fetcher
            // takes from; nothing is sent to the first.
            shared_work_queue::<QueryRowGroupMetadata>(worker_count)
                .into_iter()
                .zip(shared_work_queue::<QueryRowGroupMetadata>(worker_count))
                .map(|(io_row_groups_channel, uncached_row_groups_channel)| {
                    RowGroupFetcherFactory::new(
                        shared_no_columns.clone(),
                        read_medium,
                        true,
                        io_row_groups_channel,
                        uncached_row_groups_channel,
                    )
                })
                .collect(),
        )
        .chain(
            stealable::<RowGroupBuffer>(topology).into_iter().collect(),
            (0..worker_count).map(|_| IndexerFactory::new()).collect(),
        )
        .chain(
            stealable::<CompressedPage>(topology).into_iter().collect(),
            DecompressGate::create_for_workers(&gated_reads, &memory_budget, worker_count),
        )
        .chain(
            stealable::<CompressedPage>(topology).into_iter().collect(),
            (0..worker_count)
                .map(|_| DecompressorFactory::new())
                .collect(),
        )
        .chain(
            stealable::<DecompressedPage>(topology)
                .into_iter()
                .collect(),
            PageMemoryCounter::create_for_workers(&memory_budget, worker_count),
        )
        .chain(
            return_to_worker_mpsc::<DecompressedPage>(worker_count)
                .into_iter()
                .collect(),
            create_worker_allocators()
                .into_iter()
                .map(|allocator| RangeCutterFactory {
                    projection: no_columns.clone(),
                    eq_predicates: Arc::new(Vec::new()),
                    // Row groups are claimed by request, not by an
                    // injector, so this count is never consulted.
                    outstanding_row_groups: Arc::new(AtomicUsize::new(0)),
                    allocator,
                })
                .collect(),
        )
        .chain(
            stealable_fifo::<DecodeRange>(topology)
                .into_iter()
                .collect(),
            DecodeGate::create_for_workers(&gated_reads, worker_count),
        )
        .chain(
            stealable_fifo::<DecodeRange>(topology)
                .into_iter()
                .collect(),
            // The decoder appends the columns that say which rows of which
            // row group a batch holds.
            create_worker_allocators()
                .into_iter()
                .map(|allocator| DecoderFactory {
                    batch_size: RECORD_BATCH_SIZE,
                    add_row_group_metadata: true,
                    allocator,
                })
                .collect(),
        )
        .map_each(move |batch: RecordBatch| Ok(DecodedValues::read_decoder_output(batch, &table)?))
        .chain(
            to_single_worker_mpsc::<DecodedValues>(worker_count, compaction_worker)
                .into_iter()
                .collect(),
            MergeSortCompacter::create_for_workers(
                compaction_read,
                worker_count,
                compaction_worker,
            ),
        ))
}
