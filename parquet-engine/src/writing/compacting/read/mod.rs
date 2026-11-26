//! The read half of a compaction: its input files, read in merge order.
//!
//! The read fetches, decompresses and decodes with the operators a scan
//! uses, but for the cutter. It adds a gate and a cutter that keep their
//! input back, and the compacter that decides what is requested and what
//! those two let through.
//!
//! The dataflow, one box per operator. A solid arrow is the channel between
//! two operators, labelled with the type it carries. A dashed arrow is the
//! compacter putting requests into the request queue, or releasing rows at
//! the gate and the cutter.
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
//!     ┌ ─ ─ ─ ─ ─▶│      DecompressGate      │  any worker; lets a page on
//!   │             └──────────────────────────┘  once a row of it is released
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
//!     │           ┌──────────────────────────┐  the worker that fetched;
//!   │ ├ ─ ─ ─ ─ ─▶│    ReleasedRowsCutter    │  builds the dictionary, cuts
//!     │           └──────────────────────────┘  the rows that are released
//!   │                          │ DecodeRange
//!     │                        ▼
//!   │             ┌──────────────────────────┐
//!     │           │         Decoder          │  the worker that fetched
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
//! go on from the `DecompressGate`, to be decompressed and decoded, as the
//! compacter releases its rows, the rows of each release in one go: a
//! release is of the rows the output takes next. A page that waits for
//! its rows is not held: the gate and the cutter keep where it lies, the
//! caches keep its bytes while nothing else needs their memory, and it is
//! read again if they did not (see [`pipeline`]). How many
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
//!   [`pull_rows`](super::MergeSortSettings::pull_rows) rows of the output.
//!   - [`row_order::key_merge`] decides the order for it, from the files' sort
//!     columns. From the row groups' min/max statistics it finds the row
//!     group that starts first and those that overlap it, and merges those
//!     by their decoded keys.
//!   - [`row_order::output_row_group`] holds the order of one output row group, and
//!     how far each column of it has got.
//! - [`input::input_file`]: one input file with the reads of its columns. It is
//!   told which columns to request and which rows of them to release, and
//!   keeps the decoded values until they are sent on.
//! - [`input::read_requests`]: what the files share to send their requests and
//!   releases into the pipeline.
//! - [`pipeline`]: the gate and the cutter,
//!   and how the compacter releases rows to them.
//! - [`memory_budget`](super::memory_budget): the count of every byte the read holds. What can
//!   outlive the compacter carries its count with it and gives it back
//!   when it is dropped: a decompressed page (see [`pipeline::page_memory_counter`])
//!   and decoded values (see [`InputValues`](output::InputValues)).
//!
//! [`settings`](super::settings) holds the tunables, and [`output`] what the read produces.

pub(crate) mod compacter;
pub(crate) mod compaction_read;
pub(crate) mod input;
pub(crate) mod output;
pub(crate) mod pipeline;
pub(crate) mod row_order;

use std::sync::Arc;

use arrow_array::RecordBatch;
use dispatch::{
    DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection, RECORD_BATCH_SIZE,
    SharedWorkQueueChannelFactory, SharedWorkQueueSender, channel_input, return_to_worker_mpsc,
    shared_work_queue, stealable, stealable_fifo, to_single_worker_mpsc,
};

use super::into_dataflow_error;
use super::memory_budget::MemoryBudget;
use super::settings::MergeSortSettings;
use crate::types::metadata::QueryRowGroupMetadata;
use crate::writing::error::WriteError;
use crate::{
    CompressedPage, DecodeRange, DecoderFactory, DecompressedPage, DecompressorFactory,
    IndexerFactory, RowGroupBuffer, RowGroupFetcherFactory, WorkerAllocator,
};
use compacter::MergeSortCompacter;
use compaction_read::CompactionRead;
use input::decoded_values::DecodedValues;
use input::read_requests::ReadRequests;
use output::MergedBatchColumn;
use pipeline::decompress_gate::DecompressGate;
use pipeline::gated_reads::GatedReads;
use pipeline::page_memory_counter::PageMemoryCounter;
use pipeline::released_rows_cutter::ReleasedRowsCutterFactory;

/// Builds the read half of the dataflow: it reads the files of `table`,
/// each already sorted by `sort_column_names`, in merge order, for output
/// row groups of `output_row_group_rows` rows, within `memory_budget`. The
/// [`MergeSortCompacter`] runs on `compaction_worker`.
///
/// The output is one [`MergedBatchColumn`] per column of each merged batch.
/// Fails for a table that has deleted rows.
pub(super) fn read_in_merge_order_spec(
    dispatcher: &DataFlowDispatcher,
    table: Arc<crate::ParquetTable>,
    sort_column_names: &[String],
    output_row_group_rows: usize,
    settings: MergeSortSettings,
    memory_budget: Arc<MemoryBudget>,
    compaction_worker: usize,
) -> Result<
    OperatorSpec<MergedBatchColumn, impl OperatorFactory<MergedBatchColumn> + 'static>,
    dispatch::DataFlowError,
> {
    if table
        .row_groups()
        .iter()
        .any(|row_group| row_group.deleted_rows.is_some())
    {
        return Err(into_dataflow_error(WriteError::CompactionOfDeletedRows));
    }
    let worker_count = dispatcher.worker_count();
    let topology = dispatcher.topology();

    let sort_columns: Vec<usize> = sort_column_names
        .iter()
        .map(|name| table.schema().index_of(name))
        .collect::<Result<_, _>>()
        .map_err(|error| into_dataflow_error(error.into()))?;
    let row_group_count = table.row_groups().len();
    let column_count = table.schema().fields().len();
    let gated_reads = GatedReads::new(row_group_count, column_count, worker_count, &sort_columns);
    // The queue has room for every read, one per column of every row group,
    // so a request is never refused.
    let (request_sender, requests) = channel_input::<QueryRowGroupMetadata>(
        dispatcher,
        (row_group_count * column_count).max(1),
        Box::new(|| {}),
    );
    let read_medium = table.get_read_medium();
    // Chunks are fetched ahead within a sixteenth of the budget, and within
    // what the workers may have asked of the files' medium at a time:
    // more than is delivered in the time one read takes would only wait.
    let fetch_ahead_bytes =
        (settings.memory_bytes / 16).min(read_medium.count_read_ahead_bytes(worker_count));
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

    // Every request names its own column, so the columns the fetchers
    // read from a row group that names none are never used.
    let shared_no_columns = Arc::new(Projection::columns([]));
    // One allocator per worker for the dictionaries its cutter
    // builds, and one for the batches its decoder emits: a dictionary lives
    // as long as its read, a batch until its rows are sent on.
    let create_worker_allocators = || -> Vec<Arc<WorkerAllocator>> {
        (0..worker_count)
            .map(|_| Arc::new(WorkerAllocator::new()))
            .collect()
    };
    // The queue of requests that need a read, which every worker's fetcher
    // takes from. The gates and the cutters put into it the requests for
    // the pages they ask for again.
    let requests_to_read: Vec<SharedWorkQueueChannelFactory<QueryRowGroupMetadata>> =
        shared_work_queue(worker_count).into_iter().collect();
    let build_page_request_senders = || -> Vec<SharedWorkQueueSender<QueryRowGroupMetadata>> {
        let channels = requests_to_read.iter();
        channels
            .map(SharedWorkQueueChannelFactory::build_sender)
            .collect()
    };
    let gate_page_request_senders = build_page_request_senders();
    let cutter_page_request_senders = build_page_request_senders();
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
                .zip(requests_to_read)
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
            DecompressGate::create_for_workers(
                &gated_reads,
                &memory_budget,
                gate_page_request_senders,
            ),
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
            ReleasedRowsCutterFactory::create_for_workers(
                &gated_reads,
                &memory_budget,
                create_worker_allocators(),
                cutter_page_request_senders,
            ),
        )
        .chain(
            // A range goes to the decoder of the worker that cut it, which
            // decodes each read as one stream: the ranges of a read are
            // short when many files are merged together, and a worker that
            // took one over would skip into its first page of every column
            // and keep a decoder and a page of the read of its own.
            return_to_worker_mpsc::<DecodeRange>(worker_count)
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
                    // A read's next rows are released when the output
                    // gets to them, which is long after the last when
                    // many files are merged together.
                    lets_go_of_pages_between_ranges: true,
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
