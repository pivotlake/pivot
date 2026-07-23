//! `row_number()` window: a pipeline breaker that gathers all input, sorts by
//! `[partition keys, order keys]`, and appends a per-partition `row_number`.
//!
//! Like [`order_by_limit`](super::order_by_limit), each worker accumulates its
//! rows and forwards them to a single collector (the worker holding the
//! receiver), which does the global sort + numbering. Scoped to
//! `row_number() OVER (PARTITION BY … ORDER BY …)` — the one window function a
//! JDBC client's catalog introspection (`DatabaseMetaData.getColumns`) uses.

mod factory;

use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, partition, take};
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SortOptions};

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::order_by_limit::OrderBy;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};
use crate::waker::waker_set;

pub use factory::WindowRowNumberFactory;

/// Consuming phase: accumulate this worker's input batches, then forward them to
/// the collector.
pub struct WindowRowNumber {
    /// Sort keys: the partition keys (ascending) followed by the order keys.
    order_by: Vec<OrderBy>,
    /// How many leading `order_by` entries are partition keys; the rest order
    /// rows within a partition.
    partition_key_count: usize,
    batches: Vec<RecordBatch>,
    sender: mpsc::Sender<RecordBatch>,
    /// Set on exactly one worker — the collector that emits the result.
    receiver: Option<Receiver<RecordBatch>>,
}

impl WindowRowNumber {
    pub(crate) fn new(
        order_by: Vec<OrderBy>,
        partition_key_count: usize,
        sender: mpsc::Sender<RecordBatch>,
        receiver: Option<Receiver<RecordBatch>>,
    ) -> Self {
        Self {
            order_by,
            partition_key_count,
            batches: Vec::new(),
            sender,
            receiver,
        }
    }
}

impl Consumer<RecordBatch, RecordBatch> for WindowRowNumber {
    type Outputter = WindowRowNumberOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        if batch.num_rows() > 0 {
            self.batches.push(batch);
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        let WindowRowNumber {
            order_by,
            partition_key_count,
            batches,
            sender,
            receiver,
        } = self;

        if !batches.is_empty() {
            let schema = batches[0].schema();
            let concatenated = concat_batches(&schema, &batches)?;
            sender.send(concatenated).expect("window collector dropped");
        }
        // Drop the sender before waking so the collector observes `Disconnected`
        // only once every worker has forwarded (see order_by_limit's note); the
        // collector may be parked on another node, so wake every node.
        drop(sender);
        waker_set().notify_all();

        Ok(receiver.map(|rx| WindowRowNumberOutputter {
            rx,
            batches: Vec::new(),
            order_by,
            partition_key_count,
        }))
    }
}

/// Output phase (the collector worker): gather every worker's rows, sort
/// globally, and append the `row_number`.
pub struct WindowRowNumberOutputter {
    rx: Receiver<RecordBatch>,
    batches: Vec<RecordBatch>,
    order_by: Vec<OrderBy>,
    partition_key_count: usize,
}

impl Outputter<RecordBatch> for WindowRowNumberOutputter {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        match self.rx.try_recv() {
            Ok(batch) => {
                self.batches.push(batch);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.batches.is_empty() {
                    let out = sort_and_number(
                        mem::take(&mut self.batches),
                        &self.order_by,
                        self.partition_key_count,
                    )?;
                    sender.send(out)?;
                }
                Ok(true)
            }
        }
    }
}

/// Concatenate, sort by `order_by`, and append a per-partition `row_number`
/// (partition boundaries taken over the leading `partition_key_count` keys).
fn sort_and_number(
    batches: Vec<RecordBatch>,
    order_by: &[OrderBy],
    partition_key_count: usize,
) -> unary::Result<RecordBatch> {
    let schema = batches[0].schema();
    let batch = concat_batches(&schema, &batches)?;

    let sorted = if order_by.is_empty() {
        batch
    } else {
        let sort_columns: Vec<SortColumn> = order_by
            .iter()
            .map(|o| SortColumn {
                values: batch.column(o.column_idx()).clone(),
                options: Some(SortOptions {
                    descending: o.descending(),
                    nulls_first: o.nulls_first(),
                }),
            })
            .collect();
        let indices = lexsort_to_indices(&sort_columns, None)?;
        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| take(c, &indices, None))
            .collect::<Result<_, _>>()?;
        RecordBatch::try_new(schema.clone(), columns)?
    };

    let row_number = row_number_column(&sorted, order_by, partition_key_count)?;

    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("row_number", DataType::Int64, false));
    let mut columns: Vec<ArrayRef> = sorted.columns().to_vec();
    columns.push(Arc::new(row_number));
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// The `row_number` for each (already sorted) row: `1..` within each run of
/// equal partition keys.
fn row_number_column(
    sorted: &RecordBatch,
    order_by: &[OrderBy],
    partition_key_count: usize,
) -> unary::Result<Int64Array> {
    let n = sorted.num_rows();
    let mut values = vec![0i64; n];
    if partition_key_count == 0 {
        // A single partition over the whole input: a plain 1..n sequence.
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as i64 + 1;
        }
        return Ok(Int64Array::from(values));
    }
    let partition_columns: Vec<ArrayRef> = order_by[..partition_key_count]
        .iter()
        .map(|o| sorted.column(o.column_idx()).clone())
        .collect();
    for Range { start, end } in partition(&partition_columns)?.ranges() {
        for (offset, i) in (start..end).enumerate() {
            values[i] = offset as i64 + 1;
        }
    }
    Ok(Int64Array::from(values))
}
