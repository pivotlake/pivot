//! Lays fetched rows out in sorted order, a stretch at a time.
//!
//! The sort ran over the keys alone and left its order with the materializer
//! as `(row group, row)` pairs. The fetched batches arrive carrying the
//! metadata columns that say which row group and rows each holds, roughly in
//! the order the rows are needed. This stage keeps them, and whenever the
//! next stretch of the order is wholly resident it hands the stretch out as a
//! [`GatherJob`], lets go of every row group the stretch finished with, and
//! publishes how far it got so the materializer requests the next row groups.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::{Array, RecordBatch, UInt32Array};
use arrow_schema::{Schema, SchemaRef};
use dispatch::{RECORD_BATCH_SIZE, Sender, Unary, UnaryFactory, UnaryResult};

use super::streaming::GatherJob;
use crate::OrderedFetch;
use crate::reading::record_batch_metadata::{global_row_group, row_index};

/// One fetched batch of a row group: its first row index, and the batch.
struct ResidentBatch {
    first_row: u32,
    batch: RecordBatch,
}

pub(super) struct ReorderFactory {
    fetch: OrderedFetch,
}

pub(super) fn factories(fetch: OrderedFetch, worker_count: usize) -> Vec<ReorderFactory> {
    (0..worker_count)
        .map(|_| ReorderFactory {
            fetch: fetch.clone(),
        })
        .collect()
}

impl UnaryFactory<RecordBatch, GatherJob> for ReorderFactory {
    type Unary = Reorder;

    fn build_unary(self) -> Reorder {
        Reorder {
            fetch: self.fetch,
            resident: HashMap::new(),
            rows_left: Vec::new(),
            cursor: 0,
            sequence: 0,
            schema: None,
        }
    }
}

pub(super) struct Reorder {
    fetch: OrderedFetch,
    /// The fetched batches of each row group still needed, by first row.
    resident: HashMap<u32, Vec<ResidentBatch>>,
    /// Per row group, how many of its rows the cursor has yet to pass.
    rows_left: Vec<usize>,
    /// The next row of the order to lay out, and the next stretch's number.
    cursor: usize,
    sequence: usize,
    /// The data columns' schema, without the metadata columns.
    schema: Option<SchemaRef>,
}

impl Reorder {
    /// Hand out every stretch of the order that is wholly resident.
    fn advance(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<()> {
        let order = self
            .fetch
            .order
            .get()
            .expect("the order is published before any row is fetched");
        if self.rows_left.is_empty() {
            let groups = order.iter().map(|&(group, _)| group).max().unwrap_or(0) as usize + 1;
            self.rows_left = vec![0; groups];
            for &(group, _) in order {
                self.rows_left[group as usize] += 1;
            }
        }
        while self.cursor < order.len() {
            let end = (self.cursor + RECORD_BATCH_SIZE).min(order.len());
            let stretch = &order[self.cursor..end];
            // Each row of the stretch as (source batch, row in it), with the
            // sources numbered as the gather will see them.
            let mut sources: Vec<(u32, usize)> = Vec::new();
            let mut source_of: HashMap<(u32, usize), u32> = HashMap::new();
            let mut mapping = Vec::with_capacity(stretch.len());
            for &(group, row) in stretch {
                let Some(batches) = self.resident.get(&group) else {
                    return Ok(());
                };
                let position = batches.partition_point(|batch| batch.first_row <= row);
                if position == 0 {
                    return Ok(());
                }
                let position = position - 1;
                let batch = &batches[position];
                if row >= batch.first_row + batch.batch.num_rows() as u32 {
                    return Ok(());
                }
                let source = *source_of.entry((group, position)).or_insert_with(|| {
                    sources.push((group, position));
                    sources.len() as u32 - 1
                });
                mapping.push((source, row - batch.first_row));
            }
            sender.send(GatherJob {
                sequence: self.sequence,
                file_rows: order.len(),
                schema: self
                    .schema
                    .clone()
                    .expect("a resident batch set the schema"),
                sources: sources
                    .iter()
                    .map(|&(group, position)| self.resident[&group][position].batch.clone())
                    .collect(),
                mapping,
            })?;
            self.sequence += 1;
            for &(group, _) in stretch {
                let left = &mut self.rows_left[group as usize];
                *left -= 1;
                if *left == 0 {
                    self.resident.remove(&group);
                }
            }
            self.cursor = end;
            self.fetch.consumed.store(self.cursor, Ordering::Release);
        }
        Ok(())
    }
}

impl Unary<RecordBatch, GatherJob> for Reorder {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<GatherJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let groups = global_row_group(&batch);
        let group = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("row group ids are unsigned")
            .value(groups.run_ends().get_start_physical_index());
        let first_row = row_index(&batch).value(0);
        if self.schema.is_none() {
            let data_columns = batch.num_columns() - 2;
            self.schema = Some(Arc::new(Schema::new(
                batch.schema().fields()[..data_columns].to_vec(),
            )));
        }
        let batches = self.resident.entry(group).or_default();
        let at = batches.partition_point(|resident| resident.first_row < first_row);
        batches.insert(at, ResidentBatch { first_row, batch });
        self.advance(sender)
    }

    fn finish(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<bool> {
        if self.schema.is_none() {
            return Ok(true);
        }
        self.advance(sender)?;
        let order_len = self.fetch.order.get().map_or(0, Vec::len);
        assert_eq!(
            self.cursor, order_len,
            "every row of the order was fetched before the file finished"
        );
        Ok(true)
    }
}
