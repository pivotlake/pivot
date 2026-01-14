use crate::io::OperationIOSubmitter;
use crate::operations::{ConsumeContext, Operation, Output, PipelineBreaker};
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SortOptions};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, trace};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone)]
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
    top_k_per_batch: Vec<RecordBatch>,
    output_source: Box<dyn Output>,
    order_by: Vec<OrderBy>,
    sender: Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

fn get_top_k_from_top_ks(
    batches: Vec<RecordBatch>,
    order_by: &[OrderBy],
    k: usize,
) -> Result<RecordBatch> {
    if batches.is_empty() {
        panic!("get_top_k_streaming called with no batches");
    }

    let schema = batches[0].schema();
    let final_pool = arrow::compute::concat_batches(&schema, &batches)?;
    get_top_k_from_single(&final_pool, order_by, k)
}

fn get_top_k_from_single(
    batch: &RecordBatch,
    order_by: &[OrderBy],
    k: usize,
) -> Result<RecordBatch> {
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

    let indices = lexsort_to_indices(&sort_columns, Some(k))?;

    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(unsafe { RecordBatch::new_unchecked(batch.schema(), columns, indices.len()) })
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
            top_k_per_batch: vec![],
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
    ) -> super::Result<Option<RecordBatch>> {
        debug!("Received batch of length {:?}", batch.num_rows());
        self.top_k_per_batch
            .push(get_top_k_from_single(&batch, &self.order_by, self.limit)?);
        Ok(None)
    }
}

impl PipelineBreaker for OrderByLimit {
    fn finish(mut self: Box<Self>) -> super::Result<()> {
        let start = Instant::now();
        if !self.top_k_per_batch.is_empty() {
            let local_top_k =
                get_top_k_from_top_ks(self.top_k_per_batch, &self.order_by, self.limit)?;
            debug!("Sending on {:?}", local_top_k.num_rows());
            let _ = self.sender.send(local_top_k);
        }

        if let Some(rx) = self.receiver {
            drop(self.sender);

            trace!("Receiving...");
            // TODO: we should somehow return here, as we're blocking the entire worker
            let global_top_k: Vec<RecordBatch> = rx.into_iter().collect();

            if !global_top_k.is_empty() {
                let final_batch =
                    arrow::compute::concat_batches(&global_top_k[0].schema(), &global_top_k)
                        .map_err(Error::from)?;
                debug!("Outputting to source {:?}", final_batch.num_rows());
                self.output_source.write(get_top_k_from_top_ks(
                    global_top_k,
                    &self.order_by,
                    self.limit,
                )?);
            }

            debug!("Sort took {:?}", start.elapsed());
        }
        self.output_source.finish();
        Ok(())
    }
}
