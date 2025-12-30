use crate::io::OperationIOSubmitter;
use crate::{ConsumeContext, Operation, Output, PipelineBreaker};
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::RecordBatch;
use arrow_schema::SortOptions;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use tracing::{debug, trace};

pub struct OrderBy {
    column_idx: usize,
    descending: bool,
    nulls_first: bool,
}

impl OrderBy {
    pub fn new(column_idx: usize, descending: bool, nulls_first: bool) -> Self {
        Self {
            column_idx,
            descending,
            nulls_first,
        }
    }
}

pub struct OrderByLimit {
    limit: usize,
    batches: Vec<RecordBatch>,
    output_source: Box<dyn Output>,
    order_by: Vec<OrderBy>,
    sender: Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

fn get_top_k_from_batches(batches: Vec<RecordBatch>, order_by: &[OrderBy], k: usize) -> RecordBatch {
    if batches.is_empty() {
        panic!("get_top_k_streaming called with no batches");
    }

    let schema = batches[0].schema();
    let mut winners = Vec::with_capacity(batches.len());

    for batch in batches {
        let local_top = get_top_k_from_single(&batch, order_by, k);
        winners.push(local_top);
    }

    let final_pool = arrow::compute::concat_batches(&schema, &winners).unwrap();
    get_top_k_from_single(&final_pool, order_by, k)
}

fn get_top_k_from_single(batch: &RecordBatch, order_by: &[OrderBy], k: usize) -> RecordBatch {
    let sort_columns: Vec<SortColumn> = order_by
        .iter()
        .map(|ob| {
            let values = batch.column(ob.column_idx).clone();
            SortColumn {
                values,
                options: Some(SortOptions {
                    descending: ob.descending,
                    nulls_first: ob.nulls_first,
                }),
            }
        })
        .collect();

    let indices = lexsort_to_indices(&sort_columns, Some(k)).unwrap();

    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None).unwrap())
        .collect();

    unsafe { RecordBatch::new_unchecked(batch.schema(), columns, indices.len()) }
}

impl OrderByLimit {
    pub fn new(
        limit: usize,
        output_source: Box<dyn Output>,
        sender: Sender<RecordBatch>,
        receiver: Option<Receiver<RecordBatch>>,
        order_by: Vec<OrderBy>,
    ) -> Self {
        Self {
            limit,
            batches: vec![],
            output_source,
            sender,
            receiver,
            order_by,
        }
    }
}

impl Operation for OrderByLimit {
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> Option<RecordBatch> {
        self.batches.push(batch.clone());
        None
    }
}

impl PipelineBreaker for OrderByLimit {
    fn output(mut self: Box<Self>) {
        let start = Instant::now();
        if !self.batches.is_empty() {
            let local_top_k = get_top_k_from_batches(self.batches, &self.order_by, self.limit);
            debug!("Sending on {:?}", local_top_k.num_rows());

            let _ = self.sender.send(local_top_k);
        }

        if let Some(rx) = self.receiver {
            drop(self.sender);

            trace!("Receiving...");
            let global_top_k: Vec<RecordBatch> = rx.into_iter().collect();

            debug!("Winners {:?}", global_top_k.len());
            if !global_top_k.is_empty() {
                let final_batch = arrow::compute::concat_batches(&global_top_k[0].schema(), &global_top_k).unwrap();
                debug!("Outputting to source {:?}", final_batch.num_rows());
                self.output_source.write(final_batch);
            }

            debug!("Sort took {:?}", start.elapsed());
        }
        self.output_source.finish();
    }
}
