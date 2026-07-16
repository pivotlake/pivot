//! Partitions a flush's rows into output files: split by partition tuple,
//! accumulate each partition into file-worths, then cut each file into row groups
//! and pages.
//!
//! This is the pipeline's one and only pipeline-breaker. It fuses what would
//! otherwise be three separate stages — partition split + accumulate, row-group
//! cutting, and page planning — because none of them is heavy enough to deserve
//! its own work-stealing hop (they only reshape data; the one heavy step,
//! encoding, stays parallel downstream). So everything after this stage is a plain
//! parallel map ([`encoder`](super::encoder)) or a gather
//! ([`assembler`](super::assembler)).
//!
//! Each worker splits the batches it steals by partition tuple and buffers per
//! partition. When a partition reaches one file's worth of rows it cuts that file
//! mid-stream (full-size files, no shuffle, skew-immune). At finish each worker
//! ships its sub-file remainders to worker 0, which consolidates them per
//! partition and cuts one tail file each — so a partition straddling workers
//! leaves one tail file, not one per worker. With no `partition_by` the split is a
//! no-op (one group, key `None`), so the same stage drives plain writes too.
//!
//! The partition tuple is a one-row arrow-json [`Value`] (`{"svc":"a"}`), used
//! directly as the `HashMap` grouping key (`Value` is `Hash + Eq`) and recorded
//! as-is in the manifest.

use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender as StdSender, TryRecvError};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_ord::partition::partition;
use arrow_ord::sort::{SortColumn, lexsort_to_indices};
use arrow_schema::{Schema, SchemaRef};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use catalog::SortBounds;
use dispatch::{Consumer, Outputter, PipelineBreaker, Sender, UnaryFactory, UnaryResult};
use serde_json::Value;

mod json;
mod stats;

use json::row_object;
use stats::column_min_max;

use super::error::WriteResult;
use super::shredding;
use super::types::{ColumnChunkJob, PartitionTag, RowGroupHeader, RowGroupSortStats, SortColStat};

/// A partition tuple (a one-row arrow-json object), or `None` for an
/// unpartitioned write — used directly as the grouping key (`Value` is `Hash +
/// Eq`) and recorded in the manifest.
type PartitionKey = Option<Value>;

/// A leftover partition's rows, shipped to worker 0 at finish.
type Leftover = (PartitionKey, Vec<RecordBatch>);

/// Factory for one worker's [`Partitioner`]. Only worker 0 receives the
/// side-channel receiver.
pub(super) struct PartitionerFactory {
    partition_by: Arc<[String]>,
    /// One file's worth of rows; a partition flushes once it reaches this.
    file_rows: usize,
    builder: RowGroupBuilder,
    leftover_tx: StdSender<Leftover>,
    leftover_rx: Option<Receiver<Leftover>>,
}

impl UnaryFactory<RecordBatch, ColumnChunkJob> for PartitionerFactory {
    type Unary = PipelineBreaker<RecordBatch, ColumnChunkJob, Partitioner>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(Partitioner {
            partition_by: self.partition_by,
            file_rows: self.file_rows,
            builder: self.builder,
            buffered_by_partition: HashMap::new(),
            leftover_tx: self.leftover_tx,
            leftover_rx: self.leftover_rx,
        })
    }
}

/// One factory per worker, sharing the id counters and the worker-0 side channel.
pub(super) fn factories(
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    file_rows: usize,
    target_rows: usize,
    worker_count: usize,
) -> Vec<PartitionerFactory> {
    let builder = RowGroupBuilder {
        sort_by,
        target_rows,
        worker_count,
        next_file_id: Arc::new(AtomicU64::new(0)),
        next_row_group_id: Arc::new(AtomicU64::new(0)),
    };
    let (tx, rx) = mpsc::channel();
    let mut rx = Some(rx);
    (0..worker_count)
        .map(|_| PartitionerFactory {
            partition_by: partition_by.clone(),
            file_rows,
            builder: builder.clone(),
            leftover_tx: tx.clone(),
            leftover_rx: rx.take(),
        })
        .collect()
}

/// One partition's buffered rows on one worker, awaiting a file's worth.
#[derive(Default)]
struct Buffer {
    rows: usize,
    batches: Vec<RecordBatch>,
}

/// Per-worker consumer: split incoming batches by partition, buffer per partition,
/// and cut a file's worth into [`ColumnChunkJob`]s as soon as one accumulates.
pub(super) struct Partitioner {
    partition_by: Arc<[String]>,
    file_rows: usize,
    builder: RowGroupBuilder,
    /// Rows buffered for each partition, keyed by its tuple, awaiting a file.
    buffered_by_partition: HashMap<PartitionKey, Buffer>,
    leftover_tx: StdSender<Leftover>,
    leftover_rx: Option<Receiver<Leftover>>,
}

impl Partitioner {
    /// Split `batch` into one `(key, rows)` per distinct partition tuple. With no
    /// partition columns, the batch is one group with key `None`. Identity
    /// partitioning: lexsort by the partition columns so equal tuples are
    /// contiguous, then the `partition` kernel cuts the runs.
    fn split(&self, batch: RecordBatch) -> WriteResult<Vec<(PartitionKey, RecordBatch)>> {
        if self.partition_by.is_empty() {
            return Ok(vec![(None, batch)]);
        }
        let column = |name: &str| -> WriteResult<ArrayRef> {
            Ok(batch.column(batch.schema().index_of(name)?).clone())
        };

        let sort_cols = self
            .partition_by
            .iter()
            .map(|n| {
                Ok(SortColumn {
                    values: column(n)?,
                    options: None,
                })
            })
            .collect::<WriteResult<Vec<_>>>()?;
        let order = lexsort_to_indices(&sort_cols, None)?;
        let batch = take_record_batch(&batch, &order)?;

        let part_cols = self
            .partition_by
            .iter()
            .map(|n| Ok(batch.column(batch.schema().index_of(n)?).clone()))
            .collect::<WriteResult<Vec<ArrayRef>>>()?;
        partition(&part_cols)?
            .ranges()
            .into_iter()
            .map(|r| {
                let slice = batch.slice(r.start, r.end - r.start);
                let tuple = row_object(&slice, &self.partition_by, 0)?;
                Ok((Some(tuple), slice))
            })
            .collect()
    }
}

impl Consumer<RecordBatch, ColumnChunkJob> for Partitioner {
    type Outputter = TailOutputter;

    fn consume<S: Sender<ColumnChunkJob>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        // Fold any variant column back to its plain `{metadata, value}` pair
        // before buffering. Batches read back from files each carry the layout
        // their file chose, and only once they agree on a schema can they be
        // concatenated and re-cut into files that shred afresh. Ingest's batches
        // are already this shape, so this is a no-op for them.
        let batch = shredding::unshred_batch(batch)?;
        for (key, slice) in self.split(batch)? {
            let buffer = self.buffered_by_partition.entry(key.clone()).or_default();
            buffer.rows += slice.num_rows();
            buffer.batches.push(slice);
            if buffer.rows >= self.file_rows {
                let Buffer { batches, .. } = self.buffered_by_partition.remove(&key).unwrap();
                self.builder.emit_column_chunks(key, batches, sender)?;
            }
        }
        Ok(())
    }

    fn into_outputter(mut self) -> UnaryResult<Option<Self::Outputter>> {
        // Ship each partition's sub-file-worth remainder to worker 0.
        for (key, buffer) in self.buffered_by_partition.drain() {
            let _ = self.leftover_tx.send((key, buffer.batches));
        }
        Ok(self.leftover_rx.map(|rx| TailOutputter {
            rx,
            builder: self.builder,
            batches_by_partition: HashMap::new(),
        }))
    }
}

/// Worker 0's output phase: gather every worker's leftover partition rows and,
/// once all senders disconnect, cut one consolidated tail file per partition.
pub(super) struct TailOutputter {
    rx: Receiver<Leftover>,
    builder: RowGroupBuilder,
    /// Every worker's leftover batches, gathered per partition by its tuple.
    batches_by_partition: HashMap<PartitionKey, Vec<RecordBatch>>,
}

impl Outputter<ColumnChunkJob> for TailOutputter {
    fn output<S: Sender<ColumnChunkJob>>(&mut self, sender: &mut S) -> UnaryResult<bool> {
        match self.rx.try_recv() {
            Ok((key, batches)) => {
                self.batches_by_partition
                    .entry(key)
                    .or_default()
                    .extend(batches);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                for (key, batches) in mem::take(&mut self.batches_by_partition) {
                    self.builder.emit_column_chunks(key, batches, sender)?;
                }
                Ok(true)
            }
        }
    }
}

/// Cuts one `(partition key, file's worth of rows)` into the file's column-chunk
/// jobs: assign a `file_id`, cut row groups, and emit one [`ColumnChunkJob`] per
/// column — stamping the partition/sort/stats provenance into a shared
/// [`RowGroupHeader`] per row group. Cheap and serial; shared by the consume path
/// and the worker-0 tail. Cloneable — the id counters are shared atomics.
#[derive(Clone)]
struct RowGroupBuilder {
    sort_by: Arc<[String]>,
    target_rows: usize,
    worker_count: usize,
    next_file_id: Arc<AtomicU64>,
    next_row_group_id: Arc<AtomicU64>,
}

impl RowGroupBuilder {
    fn emit_column_chunks<S: Sender<ColumnChunkJob>>(
        &self,
        partition: PartitionKey,
        batches: Vec<RecordBatch>,
        sender: &mut S,
    ) -> UnaryResult<()> {
        let schema = batches[0].schema();
        let batch = concat_batches(&schema, &batches)?;
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        // This file's rows are all here, so this is where its variant columns
        // pick their shredding — from the rows the file actually got, and for
        // this file alone. It widens the schema, so read it back afterwards.
        let batch = shredding::shred_batch(batch)?;
        let schema = batch.schema();
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        // The assembler gathers a whole file on one worker, so every column chunk
        // of this file must name the same worker; `return_to_worker_mpsc` routes a
        // chunk to exactly the worker index its header carries. Spread files across
        // workers round-robin by id.
        let dest_worker = (file_id as usize) % self.worker_count;
        let n_row_groups = rows.div_ceil(self.target_rows);
        // The file's sort bounds span the whole file; each row group's own
        // sort stats are computed per slice below.
        let sort_bounds = self.file_sort_bounds(&batch, &schema)?;

        let mut offset = 0;
        while offset < rows {
            let len = self.target_rows.min(rows - offset);
            let slice = batch.slice(offset, len);
            offset += len;
            let tag = Arc::new(PartitionTag {
                file_id,
                n_row_groups,
                partition: partition.clone(),
                sort_bounds: sort_bounds.clone(),
                sort_stats: self.row_group_stats(&slice, &schema),
            });
            // Every column chunk of this row group shares one header.
            let header = Arc::new(RowGroupHeader {
                row_group_id: self.next_row_group_id.fetch_add(1, Ordering::Relaxed),
                dest_worker,
                schema: schema.clone(),
                tag,
            });
            for column in 0..slice.num_columns() {
                sender.send(ColumnChunkJob {
                    header: header.clone(),
                    column,
                    values: slice.column(column).clone(),
                })?;
            }
        }
        Ok(())
    }

    /// One row group's per-sort-column min/max (skipping columns whose type has
    /// no stats — they just get no footer statistics).
    fn row_group_stats(&self, slice: &RecordBatch, schema: &SchemaRef) -> RowGroupSortStats {
        let mut cols = Vec::new();
        for name in self.sort_by.iter() {
            let Ok(i) = schema.index_of(name) else {
                continue;
            };
            if let Some((min, max, null_count)) = column_min_max(slice.column(i)) {
                cols.push(SortColStat {
                    column: i,
                    min,
                    max,
                    null_count,
                });
            }
        }
        RowGroupSortStats { cols }
    }

    /// The file's sort-key bounds: each sort column's min/max over the whole file,
    /// as `{col: min}` / `{col: max}` arrow-json objects. `None` if there's no
    /// sort key or no sort column has stats.
    fn file_sort_bounds(
        &self,
        batch: &RecordBatch,
        schema: &SchemaRef,
    ) -> WriteResult<Option<SortBounds>> {
        if self.sort_by.is_empty() {
            return Ok(None);
        }
        let mut fields = Vec::new();
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        let mut included = Vec::new();
        for name in self.sort_by.iter() {
            let i = schema.index_of(name)?;
            let Some((min, max, _)) = column_min_max(batch.column(i)) else {
                continue;
            };
            fields.push(schema.field(i).as_ref().clone());
            mins.push(min);
            maxs.push(max);
            included.push(name.clone());
        }
        if included.is_empty() {
            return Ok(None);
        }
        let bound_schema = Arc::new(Schema::new(fields));
        let min_batch = RecordBatch::try_new(bound_schema.clone(), mins)?;
        let max_batch = RecordBatch::try_new(bound_schema, maxs)?;
        Ok(Some(SortBounds {
            min: row_object(&min_batch, &included, 0)?,
            max: row_object(&max_batch, &included, 0)?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    /// A partitioner with just a partition spec, for testing [`Partitioner::split`].
    fn partitioner(partition_by: &[&str]) -> Partitioner {
        let (leftover_tx, _rx) = mpsc::channel();
        Partitioner {
            partition_by: partition_by.iter().map(|s| s.to_string()).collect(),
            file_rows: usize::MAX,
            builder: RowGroupBuilder {
                sort_by: Arc::from([]),
                target_rows: usize::MAX,
                worker_count: 1,
                next_file_id: Arc::new(AtomicU64::new(0)),
                next_row_group_id: Arc::new(AtomicU64::new(0)),
            },
            buffered_by_partition: HashMap::new(),
            leftover_tx,
            leftover_rx: None,
        }
    }

    fn batch(services: &[&str], ts: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("service", DataType::Utf8, false),
            Field::new("ts", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(services.to_vec())),
                Arc::new(Int64Array::from(ts.to_vec())),
            ],
        )
        .unwrap()
    }

    /// Rows split into one group per partition value, grouped by that value.
    #[test]
    fn splits_by_partition_value() {
        let parts = partitioner(&["service"])
            .split(batch(&["b", "a", "b", "a"], &[4, 1, 3, 2]))
            .unwrap();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].0.as_ref().unwrap()["service"], "a");
        assert_eq!(parts[0].1.num_rows(), 2);
        assert_eq!(parts[1].0.as_ref().unwrap()["service"], "b");
        assert_eq!(parts[1].1.num_rows(), 2);
    }

    /// No partition columns: a single group with key `None`, batch unchanged.
    #[test]
    fn no_partition_is_one_group() {
        let parts = partitioner(&[]).split(batch(&["b", "a"], &[2, 1])).unwrap();

        assert_eq!(parts.len(), 1);
        assert!(parts[0].0.is_none());
        assert_eq!(parts[0].1.num_rows(), 2);
    }
}
