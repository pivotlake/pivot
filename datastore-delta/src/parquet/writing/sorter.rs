//! The write pipeline's sorter: put each file's rows in key order and deal
//! them into the column-chunk jobs the encoders stitch and encode.
//!
//! A parallel stage fed by the [`collector`](super::collector) over a
//! work-stealing channel. A [`FileJob::MergeSlice`] merges one planned slice
//! of a file's k-way merge (dispatch's `merge_multiway_slice`, one gather per
//! column) into its slot of the file's shared [`FileMerge`]; the worker
//! completing the file's last slice sees every merged chunk and deals the
//! file. A [`FileJob::Sorted`] file needs no merging and deals directly.
//!
//! Dealing a file is where its variant columns pick their
//! [`shredding`](super::shredding) (this is the one stage that sees a whole
//! file's rows at once) and where its rows cut into row groups:
//! `target_rows_per_group`-sized windows, each column of each window a gather
//! recipe over the chunk list, so no row data is copied here and the encode
//! workers stitch the batch-sized chunks into full row groups in parallel.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::{ArrayRef, RecordBatch};
use dispatch::memory::SlabAllocator;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult, merge_multiway_slice};

use super::error::WriteError;
use super::shredding;
use super::types::{ColumnChunkJob, FileJob, FilePlan, PartitionTag, RowGroupHeader};

pub(super) type SorterFactory = DefaultUnaryFactory<Sorter>;

pub(super) fn factories(worker_count: usize) -> Vec<SorterFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

#[derive(Default)]
pub(super) struct Sorter {
    /// The ring memory this worker's merged rows gather into, taken on the
    /// first slice rather than when the operator is built, so a worker that
    /// merges nothing holds no buffer.
    allocator: Option<SlabAllocator>,
}

impl Unary<FileJob, ColumnChunkJob> for Sorter {
    fn consume(
        &mut self,
        job: FileJob,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        match job {
            FileJob::Sorted(file) => {
                send_jobs_for_file(&file.plan, &file.chunks, file.rows, sender)
            }
            FileJob::MergeSlice { merge, slice } => {
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                let merged = merge_multiway_slice(
                    allocator,
                    &merge.sort_keys,
                    &merge.runs,
                    &merge.slices[slice],
                )
                .map_err(WriteError::from)?;
                merge.merged[slice]
                    .set(merged)
                    .unwrap_or_else(|_| unreachable!("each slice merges once"));
                if merge.slices_remaining.fetch_sub(1, Ordering::AcqRel) > 1 {
                    return Ok(());
                }
                // Last slice in: every slot is set (the counter is the
                // barrier), and the slots in index order are the file's rows
                // in key order.
                let chunks: Vec<RecordBatch> = merge
                    .merged
                    .iter()
                    .map(|slot| slot.get().expect("every slice merged").clone())
                    .collect();
                send_jobs_for_file(&merge.plan, &chunks, merge.rows, sender)
            }
        }
    }
}

/// Deal one file's sorted chunks as its column-chunk jobs: identity gathers
/// over the chunk list, one row group per `target_rows_per_group` window, one
/// job per column of each window. The file's shredding is only *planned*
/// here; every rewrite of row data, shredding included, belongs to the encode
/// workers.
fn send_jobs_for_file(
    plan: &FilePlan,
    chunks: &[RecordBatch],
    rows: usize,
    sender: &mut dyn Sender<ColumnChunkJob>,
) -> UnaryResult<()> {
    debug_assert!(rows > 0, "the collector only cuts files with rows");
    tracing::debug!(
        file_id = plan.file_id,
        rows,
        "dealing a sorted file into column-chunk jobs"
    );
    let file_shredding = shredding::plan_file_shredding(chunks)?;

    // A row group's column is materialized by the encoder into a single
    // 2MB ring slab (`concat_chunks`), and the widest fixed-width value —
    // a decimal128, or a view column's 16-byte view structs — fills it at
    // 128K rows. (A view column's string *bytes* are unbounded either
    // way: they copy into their own chain of data blocks, never into that
    // slab.) The production target is exactly 128K for this reason; a
    // larger one would panic in the encoder's slab allocation.
    debug_assert!(
        plan.target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
        "a row group's widest column must fit one slab"
    );
    let n_row_groups = rows.div_ceil(plan.target_rows_per_group);
    let tag = Arc::new(PartitionTag {
        file_id: plan.file_id,
        n_row_groups,
        partition: plan.partition.clone(),
    });

    // Walk the chunks once, dealing each row group its slice of them:
    // whole chunks where they fit, zero-copy slices at the window edges.
    // The rows stay untouched; the encode workers stitch each window.
    let mut rows_left = rows;
    let mut chunk_cursor = 0;
    let mut rows_taken_of_chunk = 0;
    for row_group in 0..n_row_groups {
        let mut rows_wanted = plan.target_rows_per_group.min(rows_left);
        rows_left -= rows_wanted;
        let mut window_chunks: Vec<RecordBatch> = Vec::new();
        while rows_wanted > 0 {
            let chunk = &chunks[chunk_cursor];
            let rows_available = chunk.num_rows() - rows_taken_of_chunk;
            let taken = rows_available.min(rows_wanted);
            window_chunks.push(if taken == chunk.num_rows() {
                chunk.clone()
            } else {
                chunk.slice(rows_taken_of_chunk, taken)
            });
            rows_taken_of_chunk += taken;
            rows_wanted -= taken;
            if rows_taken_of_chunk == chunk.num_rows() {
                chunk_cursor += 1;
                rows_taken_of_chunk = 0;
            }
        }

        let header = Arc::new(RowGroupHeader {
            row_group_id: plan.base_row_group_id + row_group as u64,
            dest_worker: plan.dest_worker,
            schema: file_shredding.schema.clone(),
            tag: tag.clone(),
        });
        for column in 0..file_shredding.schema.fields().len() {
            let column_chunks: Arc<[ArrayRef]> = window_chunks
                .iter()
                .map(|chunk| chunk.column(column).clone())
                .collect();
            sender.send(ColumnChunkJob {
                header: header.clone(),
                column,
                chunks: column_chunks,
                shredding: file_shredding.column_shredding[column].clone(),
            })?;
        }
    }
    Ok(())
}
