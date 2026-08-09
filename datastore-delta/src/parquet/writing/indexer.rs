//! The write pipeline's indexer: turns the ordered stream of record batches
//! into the column-chunk jobs of one Parquet file per partition — the write
//! mirror of the read pipeline's indexer, which walks a file's footer into
//! read jobs.
//!
//! Every batch of the stream funnels to one worker's indexer in the order it
//! was produced (the single-worker channel in [`super`]), each batch holding
//! rows of exactly one partition, partitions arriving grouped, and — for a
//! sorted table — each partition's rows already in key order. So the indexer
//! never sorts, gathers or copies anything: it walks the stream, collects each
//! partition's chunk references, and when a partition ends (its tuple changes,
//! or the stream finishes) emits that partition's file: row groups are
//! `target_rows_per_group`-sized windows over the partition's rows, each
//! column of each window a gather over the chunk list, so the
//! encode workers stitch the batch-sized chunks into full row groups in
//! parallel. One file per partition, whatever its size.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, SortField};
use dispatch::{Sender, Unary, UnaryFactory, UnaryResult};

use super::error::WriteResult;
use super::shredding;
use super::types::{ColumnChunkJob, PartitionTag, RowGroupHeader};
use crate::scalar_values_from_row;

/// Builds one worker's [`Indexer`]. Every worker gets one, but only the
/// worker the single-worker channel targets ever receives a batch.
pub(super) struct IndexerFactory {
    partition_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_file_bytes: usize,
    worker_count: usize,
}

pub(super) fn factories(
    partition_by: Arc<[String]>,
    target_rows_per_group: usize,
    target_file_bytes: usize,
    worker_count: usize,
) -> Vec<IndexerFactory> {
    (0..worker_count)
        .map(|_| IndexerFactory {
            partition_by: partition_by.clone(),
            target_rows_per_group,
            target_file_bytes,
            worker_count,
        })
        .collect()
}

impl UnaryFactory<RecordBatch, ColumnChunkJob> for IndexerFactory {
    type Unary = Indexer;

    fn build_unary(self) -> Indexer {
        Indexer {
            partition_by: self.partition_by,
            target_rows_per_group: self.target_rows_per_group,
            target_file_bytes: self.target_file_bytes,
            worker_count: self.worker_count,
            partition_tuple_converter: None,
            open_partition: None,
            next_file_id: 0,
            next_row_group_id: 0,
        }
    }
}

/// The default for [`Indexer`]'s file bound: how many bytes of rows a file
/// holds before the indexer cuts it and starts another.
///
/// A partition's rows are written out as they arrive rather than held until the
/// partition ends: an unpartitioned insert is one partition however large the
/// table is, so holding one would mean buffering the whole insert before
/// writing a byte of it.
///
/// The bound is bytes rather than rows because rows differ enormously in width
/// — a hundred-column row is orders of magnitude larger than a narrow one — and
/// what has to stay bounded is the memory a file's rows occupy while they wait,
/// not how many of them there are.
const TARGET_FILE_BYTES: usize = 128 * 1024 * 1024;

/// The partition whose chunks are still arriving: its identity and everything
/// collected so far.
struct OpenPartition {
    tuple: Option<OwnedRow>,
    chunks: Vec<RecordBatch>,
    rows: usize,
    /// What the chunks occupy in memory, which is what the file cut bounds.
    ///
    /// Two things are counted, and the larger decides: the values' own size, and
    /// the ring slabs holding them. A batch's column occupies a whole slab
    /// however little of it the column fills, so a hundred-column table spends a
    /// hundred slabs a batch — orders of magnitude more ring than the values
    /// measure, and the reason a wide insert exhausted it.
    bytes: usize,
}

pub(super) struct Indexer {
    partition_by: Arc<[String]>,
    target_rows_per_group: usize,
    /// Bytes of rows a file holds before it is cut; see [`TARGET_FILE_BYTES`].
    target_file_bytes: usize,
    worker_count: usize,
    /// Encodes a batch's partition columns into its comparable tuple; built
    /// from the first batch's schema.
    partition_tuple_converter: Option<RowConverter>,
    open_partition: Option<OpenPartition>,
    next_file_id: u64,
    next_row_group_id: u64,
}

impl Indexer {
    /// The partition tuple `batch` belongs to. The whole batch is one
    /// partition — the upstream ORDER BY split it — so only the first row is
    /// encoded and read.
    fn partition_tuple_of(&mut self, batch: &RecordBatch) -> WriteResult<Option<OwnedRow>> {
        if self.partition_by.is_empty() {
            return Ok(None);
        }
        let first_row_of_each_partition_column = self
            .partition_by
            .iter()
            .map(|name| Ok(batch.column(batch.schema().index_of(name)?).slice(0, 1)))
            .collect::<WriteResult<Vec<ArrayRef>>>()?;
        let converter = match &self.partition_tuple_converter {
            Some(converter) => converter,
            None => self.partition_tuple_converter.insert(RowConverter::new(
                first_row_of_each_partition_column
                    .iter()
                    .map(|column| SortField::new(column.data_type().clone()))
                    .collect(),
            )?),
        };
        #[cfg(debug_assertions)]
        {
            let every_row = self
                .partition_by
                .iter()
                .map(|name| Ok(batch.column(batch.schema().index_of(name)?).clone()))
                .collect::<WriteResult<Vec<ArrayRef>>>()?;
            let tuples = converter.convert_columns(&every_row)?;
            debug_assert!(
                (1..batch.num_rows()).all(|row| tuples.row(row) == tuples.row(0)),
                "a batch reaching the indexer holds one partition"
            );
        }
        let tuple = converter.convert_columns(&first_row_of_each_partition_column)?;
        Ok(Some(tuple.row(0).owned()))
    }

    /// Send a file's worth of a partition off as one file: identity-gather jobs
    /// over its chunk list, one row group per `target_rows_per_group` window. A
    /// partition holding more than [`TARGET_FILE_BYTES`] becomes several files,
    /// all carrying its partition values. The
    /// file's shredding is only *planned* here; every rewrite of row data,
    /// shredding included, belongs to the encode workers.
    fn send_jobs_for_partitioned_file(
        &mut self,
        partition: OpenPartition,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        if partition.rows == 0 {
            return Ok(());
        }
        // This is the only place that sees a whole file's rows, so it is
        // where the file's variant columns pick their shredding. The plan
        // widens the schema the footer will describe; the values are still
        // the plain pairs, rewritten per row group on the encode workers.
        let file_shredding = shredding::plan_file_shredding(&partition.chunks)?;
        let file_id = self.next_file_id;
        self.next_file_id += 1;
        // The assembler gathers a whole file on one worker; spread files
        // across workers round-robin by id.
        let dest_worker = (file_id as usize) % self.worker_count;
        let partition_values = match partition.tuple {
            Some(_) => Some(scalar_values_from_row(
                &partition.chunks[0],
                &self.partition_by,
                0,
            )?),
            None => None,
        };

        // A row group's column is materialized by the encoder into a single
        // 2MB ring slab (`concat_chunks`), and the widest fixed-width value —
        // a decimal128, or a view column's 16-byte view structs — fills it at
        // 128K rows. (A view column's string *bytes* are unbounded either
        // way: they copy into their own chain of data blocks, never into that
        // slab.) The production target is exactly 128K for this reason; a
        // larger one would panic in the encoder's slab allocation.
        debug_assert!(
            self.target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
            "a row group's widest column must fit one slab"
        );
        // Walk the chunks once, dealing each row group its slice of them:
        // whole chunks where they fit, zero-copy slices at the window edges.
        // The rows stay untouched; the encode workers stitch each window.
        let n_row_groups = partition.rows.div_ceil(self.target_rows_per_group);
        let mut rows_left = partition.rows;
        let mut chunk_cursor = 0;
        let mut rows_taken_of_chunk = 0;
        for _ in 0..n_row_groups {
            let mut rows_wanted = self.target_rows_per_group.min(rows_left);
            rows_left -= rows_wanted;
            let mut window_chunks: Vec<RecordBatch> = Vec::new();
            while rows_wanted > 0 {
                let chunk = &partition.chunks[chunk_cursor];
                let rows_available = chunk.num_rows() - rows_taken_of_chunk;
                let rows = rows_available.min(rows_wanted);
                window_chunks.push(if rows == chunk.num_rows() {
                    chunk.clone()
                } else {
                    chunk.slice(rows_taken_of_chunk, rows)
                });
                rows_taken_of_chunk += rows;
                rows_wanted -= rows;
                if rows_taken_of_chunk == chunk.num_rows() {
                    chunk_cursor += 1;
                    rows_taken_of_chunk = 0;
                }
            }

            let tag = Arc::new(PartitionTag {
                file_id,
                n_row_groups,
                partition: partition_values.clone(),
            });
            let header = Arc::new(RowGroupHeader {
                row_group_id: self.next_row_group_id,
                dest_worker,
                schema: file_shredding.schema.clone(),
                tag,
            });
            self.next_row_group_id += 1;
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
}

impl Unary<RecordBatch, ColumnChunkJob> for Indexer {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let tuple = self.partition_tuple_of(&batch)?;
        let partition_changed = self
            .open_partition
            .as_ref()
            .is_some_and(|open| open.tuple != tuple);
        if partition_changed {
            let finished = self
                .open_partition
                .take()
                .expect("the partition change was against an open partition");
            self.send_jobs_for_partitioned_file(finished, sender)?;
        }
        let open = self.open_partition.get_or_insert_with(|| OpenPartition {
            tuple,
            chunks: Vec::new(),
            rows: 0,
            bytes: 0,
        });
        open.rows += batch.num_rows();
        let values = batch
            .columns()
            .iter()
            .map(|column| column.get_array_memory_size())
            .sum::<usize>();
        let slabs = batch.num_columns() * dispatch::BUFFER_SIZE;
        open.bytes += values.max(slabs);
        open.chunks.push(batch);
        if open.bytes >= self.target_file_bytes {
            let full = self
                .open_partition
                .take()
                .expect("the partition was open a line ago");
            self.send_jobs_for_partitioned_file(full, sender)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<ColumnChunkJob>) -> UnaryResult<bool> {
        if let Some(last) = self.open_partition.take() {
            self.send_jobs_for_partitioned_file(last, sender)?;
        }
        Ok(true)
    }
}
