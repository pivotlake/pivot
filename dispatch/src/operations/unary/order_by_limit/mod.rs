use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SortOptions};
use std::mem;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, TryRecvError};
use thiserror::Error;
use tracing::debug;

mod factory;
pub use factory::OrderByLimitFactory;

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

pub struct OrderByLimit {
    limit: usize,
    top_k_per_batch: Vec<RecordBatch>,
    order_by: Vec<OrderBy>,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl OrderByLimit {
    pub fn new(
        order_by: Vec<OrderBy>,
        limit: usize,
        sender: mpsc::Sender<RecordBatch>,
        receiver: Option<Receiver<RecordBatch>>,
    ) -> Self {
        Self {
            limit,
            top_k_per_batch: vec![],
            order_by,
            sender,
            receiver,
        }
    }
}

impl Consumer<RecordBatch, RecordBatch> for OrderByLimit {
    type Outputter = OrderByLimitOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        debug!("Received batch of length {:?}", batch.num_rows());
        self.top_k_per_batch
            .push(get_top_k_from_single(&batch, &self.order_by, self.limit)?);
        Ok(())
    }

    fn into_outputter(self) -> crate::operations::unary::Result<Option<Self::Outputter>> {
        if !self.top_k_per_batch.is_empty() {
            let local_top_k =
                get_top_k_from_top_ks(self.top_k_per_batch, &self.order_by, self.limit)?;
            debug!("Sending on {:?}", local_top_k.num_rows());
            let _ = self.sender.send(local_top_k);
        }

        Ok(self.receiver.map(|rx| OrderByLimitOutputter {
            rx,
            batches: vec![],
            order_by: self.order_by,
            limit: self.limit,
        }))
    }
}

pub struct OrderByLimitOutputter {
    rx: Receiver<RecordBatch>,
    batches: Vec<RecordBatch>,
    order_by: Vec<OrderBy>,
    limit: usize,
}

impl Outputter<RecordBatch> for OrderByLimitOutputter {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        match self.rx.try_recv() {
            Ok(c) => {
                self.batches.push(c);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.batches.is_empty() {
                    let final_batch =
                        arrow::compute::concat_batches(&self.batches[0].schema(), &self.batches)
                            .map_err(Error::from)?;
                    debug!("Outputting to source {:?}", final_batch.num_rows());
                    sender.send(get_top_k_from_top_ks(
                        mem::take(&mut self.batches),
                        &self.order_by,
                        self.limit,
                    )?)?;
                }
                Ok(true)
            }
        }
    }
}
