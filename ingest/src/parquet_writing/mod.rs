//! The streaming, page-parallel Parquet **write** pipeline — the write-side
//! mirror of dispatch's read pipeline (indexer → decompressor → decoder), built
//! from the same operator/channel toolkit and organised the same way: one
//! module per stage, message types in [`types`].
//!
//! A flush's buffered items flow through five work-stealing stages and finished
//! Parquet files stream out the far end (consumed via `execute()`), so nothing
//! waits for the whole flush and only a bounded amount sits in memory:
//!
//! 1. [`convert`] (`T → RecordBatch`) — each worker flattens the items it
//!    steals into Arrow batches.
//! 2. [`builder`] (`RecordBatch → RowGroupBatch`) — fill row groups, emitting
//!    them as they're ready; the straddling remainder is combined on worker 0.
//! 3. [`planner`] (`RowGroupBatch → PipePageJob`) — cut one row group into pages.
//! 4. [`encoder`] (`PipePageJob → PipeEncodedPage`) — PLAIN-encode + snappy, in
//!    parallel; pages route back to their row group's owner worker.
//! 5. [`assembler`] (`PipeEncodedPage → Vec<u8>`) — collect a row group's pages,
//!    assemble it, and emit a finished file every `target_row_groups`; leftover
//!    row groups are packed into the final file(s) on worker 0.
//!
//! Because stage 2 onward is batch-native, any `RecordBatch` dataflow can feed
//! the pipeline directly: [`encode`] chains stages 2–5 onto an existing spec
//! (e.g. a table scan), which is how compaction re-encodes a table without its
//! batches ever leaving the worker pool.

mod assembler;
mod builder;
mod convert;
mod encoder;
mod planner;
mod types;

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use dispatch::{
    DataFlowDispatcher, DataFlowHandle, OperatorFactory, OperatorSpec, RecordBatchFactoryBridge,
    RecordBatchOperatorSpec, return_to_worker_mpsc, stealable, values_input,
};
use thriftparquet::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};

use types::{PipeEncodedPage, PipePageJob, RowGroupBatch};

/// A buffered ingest item that flattens itself into one Arrow `RecordBatch`.
/// Conversion runs inside the pipeline (stage 1) on a worker, so the receive
/// path only ever buffers the cheap, unconverted item.
pub trait ToRecordBatch: Send + 'static {
    /// Output rows this item will produce — a cheap count (not the full
    /// conversion), used for the flush threshold.
    fn num_rows(&self) -> usize;
    /// Flatten into one batch, or `None` if it produced no rows.
    fn to_record_batch(self) -> Result<Option<arrow_array::RecordBatch>, ArrowError>;
}

/// Carry a stage's `String` error (encode / assembly) onto the pipeline's
/// `unary` error channel.
fn to_arrow(e: String) -> ArrowError {
    ArrowError::ComputeError(e)
}

/// Serialise a thrift value (a page header or the file footer) into `out`.
/// Shared by [`encoder`] (page headers) and [`assembler`] (the footer).
pub(super) fn write_thrift<T: WriteThrift>(value: &T, out: &mut Vec<u8>) -> Result<(), String> {
    let mut prot = ThriftCompactOutputProtocol::new(out);
    value
        .write_thrift(&mut prot)
        .map_err(|e| format!("thrift encode: {e}"))
}

/// Run the write pipeline for one flush's `items`, yielding finished Parquet
/// file byte buffers as they complete. The returned handle streams: iterate it
/// on a blocking thread and write each file as it arrives.
pub fn run<T: ToRecordBatch>(
    dispatcher: &DataFlowDispatcher,
    items: Vec<T>,
    target_rows: usize,
    target_row_groups: usize,
) -> DataFlowHandle<Vec<u8>> {
    let workers = dispatcher.worker_count();
    let batches = values_input(dispatcher, items).chain(
        stealable::<T>(workers).into_iter().collect(),
        convert::factories::<T>(workers),
    );
    encode_stages(batches, workers, target_rows, target_row_groups)
}

/// Chain the encode stages (row-group building onward) onto an existing
/// `RecordBatch` dataflow and run it, yielding finished Parquet file byte
/// buffers as they complete. Scan-to-Parquet (compaction) is one dataflow this
/// way: the batches a worker decodes feed the encoder without ever leaving the
/// pool.
pub fn encode(
    spec: RecordBatchOperatorSpec,
    target_rows: usize,
    target_row_groups: usize,
) -> DataFlowHandle<Vec<u8>> {
    let (dispatcher, heads) = spec.into_parts();
    let workers = heads.len();
    let batches = OperatorSpec::new(
        dispatcher,
        heads
            .into_iter()
            .map(RecordBatchFactoryBridge::new)
            .collect::<Vec<_>>(),
    );
    encode_stages(batches, workers, target_rows, target_row_groups)
}

/// Stages 2–5: cut `RecordBatch`es into row groups, plan pages, encode them in
/// parallel, and assemble finished files.
fn encode_stages<OF: OperatorFactory<RecordBatch> + Send + 'static>(
    batches: OperatorSpec<RecordBatch, OF>,
    workers: usize,
    target_rows: usize,
    target_row_groups: usize,
) -> DataFlowHandle<Vec<u8>> {
    let next_rg_id = Arc::new(AtomicU64::new(0));
    batches
        .chain(
            stealable::<RecordBatch>(workers).into_iter().collect(),
            builder::factories(target_rows, workers, next_rg_id),
        )
        .chain(
            stealable::<RowGroupBatch>(workers).into_iter().collect(),
            planner::factories(workers),
        )
        .chain(
            stealable::<PipePageJob>(workers).into_iter().collect(),
            encoder::factories(workers),
        )
        .chain(
            return_to_worker_mpsc::<PipeEncodedPage>(workers)
                .into_iter()
                .collect(),
            assembler::factories(target_row_groups, workers),
        )
        .execute()
}
