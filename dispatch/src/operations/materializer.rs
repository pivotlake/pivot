use crate::io::OperationIOSubmitter;
use crate::record_batch_metadata::{global_row_group, row_index};
use crate::table::{RowGroupMetadataHandle, Table};
use crate::{ConsumeContext, Operation};
use arrow::compute::filter_record_batch;
use arrow_array::{Array, RecordBatch, UInt32Array};
use arrow_schema::ArrowError;
use parquetd::Projection;
use std::mem;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Filter a record batch by a list of global indexes within the row group. Every `dispatch`
/// RecordBatch should have a row index column; the `global_indexes` are meant to match that.
fn filter_batch_by_global_indexes(
    batch: &RecordBatch,
    global_indexes: &[u32],
) -> Result<RecordBatch> {
    // 1. Get the batch indexes
    let batch_indexes = row_index(batch);

    let mut keep_indices = Vec::with_capacity(batch_indexes.len());

    // 2. Two-pointer intersection logic
    // Assumes both batch_indexes and global_filters are sorted
    let mut batch_ptr = 0;
    let mut filter_ptr = 0;

    while batch_ptr < batch_indexes.len() && filter_ptr < global_indexes.len() {
        let b_val = batch_indexes.value(batch_ptr);
        let f_val = global_indexes[filter_ptr];

        if b_val == f_val {
            keep_indices.push(batch_ptr as u32);
            batch_ptr += 1;
            filter_ptr += 1;
        } else if b_val < f_val {
            batch_ptr += 1;
        } else {
            filter_ptr += 1;
        }
    }

    // 3. Create the Boolean mask
    let mut mask = vec![false; batch.num_rows()];
    for idx in keep_indices {
        mask[idx as usize] = true;
    }
    let filter_mask = arrow::array::BooleanArray::from(mask);

    // 4. Apply the filter using arrow::compute::filter_record_batch
    Ok(filter_record_batch(batch, &filter_mask)?)
}

/// The `Materializer` materializes a new projection from the Parquet reader for rows published
/// to it.
///
/// Essentially, for every row group the materializer gets, it will send out a request to the parquet
/// reader. The parquet reader will return that entire row group, and the Materializer will filter
/// only the rows that match the rows the materializer originally got from its publisher.
///
/// This can be very useful when there is a filtering operation and later a large projection-
/// we can filter WITHOUT bringing all the data, and only later materialize data for the rows that
/// passed the filter
pub struct Materializer {
    projection: Option<Projection>,
    table: Arc<Table>,
}

impl Materializer {
    pub fn new(projection: Option<Projection>, table: Arc<Table>) -> Self {
        Self { projection, table }
    }

    /// Handle an incoming `RecordBatch` from our parent publisher.
    ///
    /// The idea here is to run through the group array (which is a `RunArray`), doing two things:
    /// 1. Sending one IO request per group to retrieve that entire `RowGroup` (with the given
    ///    projection) from disk
    /// 2. Save all indexes of Rows we're seeing for the given `RowGroup` so we can match later when
    ///    we retrieve the row group from disk
    fn handle_published_record_batch(
        &mut self,
        io_submitter: &mut OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> Result<()> {
        let groups = global_row_group(batch);

        let row_indexes = row_index(batch);

        let groups_run_values = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("RunArray values must be UInt32Array");

        let mut running_group = groups.get_physical_index(0);
        let mut running_indexes = vec![];

        for i in 0..batch.num_rows() {
            let logical_index = groups.get_physical_index(i);
            let group = if groups_run_values.is_valid(logical_index) {
                groups_run_values.value(logical_index)
            } else {
                panic!("unsupported")
            };

            let row_index = row_indexes.value(i);
            if running_group != group as usize {
                io_submitter.submit_operation_io(
                    RowGroupMetadataHandle::new(self.table.clone(), running_group),
                    self.projection.as_ref(),
                    Box::new((running_group, mem::take(&mut running_indexes))),
                );
            }

            running_group = group as usize;
            running_indexes.push(row_index);
        }

        io_submitter.submit_operation_io(
            RowGroupMetadataHandle::new(self.table.clone(), running_group),
            self.projection.as_ref(),
            Box::new((running_group, running_indexes)),
        );
        Ok(())
    }
}

impl Operation for Materializer {
    fn consume(
        &mut self,
        context: &ConsumeContext,
        mut io_submitter: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> super::Result<Option<RecordBatch>> {
        match context {
            ConsumeContext::Publisher => {
                self.handle_published_record_batch(&mut io_submitter, batch)?;
                Ok(None)
            }
            ConsumeContext::IORequest(value) => {
                let (_, indices) = value.downcast_ref::<(usize, Vec<u32>)>().unwrap();
                let batch = filter_batch_by_global_indexes(batch, indices)?;
                if batch.num_rows() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(batch))
                }
            }
        }
    }
}
