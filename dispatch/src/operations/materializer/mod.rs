mod factory;
pub use factory::*;

use crate::data_flow::WorkStatus;
use crate::io::cache::CACHE;
use crate::io::{IOLocation, IORequest};
use crate::operations::Operator;
use crate::operations::channels::{Receiver, Sender};
use crate::operations::unary::Unary;
use crate::record_batch_metadata::{global_row_group, row_index};
use crate::table::input::{RowGroupBuffer, calculate_projected_row_group_range};
use crate::table::{Projection, RowGroupMetadataHandle, Table};
use arrow::compute::filter_record_batch;
use arrow_array::{Array, RecordBatch, UInt32Array};
use arrow_schema::ArrowError;
use bytes::Bytes;
use std::mem;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use thiserror::Error;
use tracing::debug;

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

pub struct MaterializeJobGenerator {
    projection_to_materialize: Option<Projection>,
    table: Arc<Table>,
}

pub struct MaterializeRequest {
    filtered_indices: Vec<u32>,
    row_group_metadata_handle: RowGroupMetadataHandle,
    location: IOLocation,
}

impl MaterializeJobGenerator {
    pub fn new(projection_to_materialize: Option<Projection>, table: Arc<Table>) -> Self {
        Self {
            projection_to_materialize,
            table,
        }
    }
}

impl Unary<RecordBatch, MaterializeRequest> for MaterializeJobGenerator {
    /// Handle an incoming `RecordBatch` from our parent publisher.
    ///
    /// The idea here is to run through the group array (which is a `RunArray`), doing two things:
    /// 1. Sending one IO request per group to retrieve that entire `RowGroup` (with the given
    ///    projection) from disk
    /// 2. Save all indexes of Rows we're seeing for the given `RowGroup` so we can match later when
    ///    we retrieve the row group from disk
    fn consume<S: Sender<MaterializeRequest>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
        let groups = global_row_group(&batch);

        let row_indexes = row_index(&batch);

        let groups_run_values = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("RunArray values must be UInt32Array");

        let logical_index = groups.get_physical_index(0);
        let mut running_group = if groups_run_values.is_valid(logical_index) {
            groups_run_values.value(logical_index) as usize
        } else {
            panic!("unsupported")
        };

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
                let handle = RowGroupMetadataHandle::new(self.table.clone(), running_group);
                let metadata = handle.get();
                let parquet_row_group_metadata = metadata
                    .arrow_metadata
                    .metadata()
                    .row_group(metadata.row_group);
                let (offset, length) = calculate_projected_row_group_range(
                    parquet_row_group_metadata,
                    self.projection_to_materialize.as_ref(),
                );
                debug!(
                    "Sending indices {:?} for group {:?}",
                    running_indexes, running_group
                );
                let mut running_indices = mem::take(&mut running_indexes);
                running_indices.sort_unstable();
                sender.send(MaterializeRequest {
                    filtered_indices: running_indices,
                    row_group_metadata_handle: RowGroupMetadataHandle::new(
                        self.table.clone(),
                        running_group,
                    ),
                    location: IOLocation {
                        raw_fd: metadata.file.as_raw_fd(),
                        offset,
                        size: length,
                    },
                })?;
            }

            running_group = group as usize;
            running_indexes.push(row_index);
        }

        let handle = RowGroupMetadataHandle::new(self.table.clone(), running_group);
        let metadata = handle.get();
        let parquet_row_group_metadata = metadata
            .arrow_metadata
            .metadata()
            .row_group(metadata.row_group);
        let (offset, length) = calculate_projected_row_group_range(
            parquet_row_group_metadata,
            self.projection_to_materialize.as_ref(),
        );
        debug!(
            "Last Sending indices {:?} for group {:?}",
            running_indexes, running_group
        );
        let mut running_indices = mem::take(&mut running_indexes);
        running_indices.sort_unstable();
        sender.send(MaterializeRequest {
            filtered_indices: running_indices,
            row_group_metadata_handle: RowGroupMetadataHandle::new(
                self.table.clone(),
                running_group,
            ),
            location: IOLocation {
                raw_fd: metadata.file.as_raw_fd(),
                offset,
                size: length,
            },
        })?;

        Ok(())
    }
}

pub struct Materializer<R: Receiver<MaterializeRequest>, S: Sender<RowGroupBuffer>> {
    current_row_group: Option<MaterializeRequest>,
    waiting: usize,
    rx: R,
    tx: S,
}

impl<R: Receiver<MaterializeRequest>, S: Sender<RowGroupBuffer>> Materializer<R, S> {
    pub fn new(rx: R, tx: S) -> Self {
        Self {
            current_row_group: None,
            waiting: 0,
            rx,
            tx,
        }
    }

    fn set_current_row_group_if_not_set(&mut self) {
        if self.current_row_group.is_some() {
            return;
        }

        if let Some(fetch) = self.rx.try_recv() {
            self.current_row_group = Some(fetch);
        }
    }
}

impl<R: Receiver<MaterializeRequest>, S: Sender<RowGroupBuffer>> Operator for Materializer<R, S> {
    fn run_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
        self.set_current_row_group_if_not_set();

        if let Some(rg) = self.current_row_group.as_ref()
            && let Some(b) = CACHE.get(&rg.location)
        {
            let row_group_request = self.current_row_group.take().unwrap();
            self.tx.send(RowGroupBuffer {
                metadata: row_group_request.row_group_metadata_handle,
                bytes: b,
                base_offset: row_group_request.location.offset as u64,
                filtered_indices: Some(row_group_request.filtered_indices),
            })?;
            return Ok(WorkStatus::Ran);
        }
        Ok(WorkStatus::Pending)
    }
    fn next_io_request(&mut self) -> crate::operations::Result<Option<IORequest>> {
        self.set_current_row_group_if_not_set();

        if let Some(rg) = self.current_row_group.as_ref()
            && !CACHE.contains(&rg.location)
        {
            let row_group_request = self.current_row_group.take().unwrap();
            self.waiting += 1;
            return Ok(Some(IORequest {
                location: row_group_request.location.clone(),
                ctx: Box::new(row_group_request),
            }));
        }
        Ok(None)
    }

    fn process_disk_response(
        &mut self,
        buffer: Bytes,
        request: IORequest,
    ) -> crate::operations::Result<()> {
        let fetch = *request.ctx.downcast::<MaterializeRequest>().unwrap();
        self.waiting -= 1;
        debug!(
            "Received Disk Response {:?} {:?} {:?}",
            request.location,
            fetch.row_group_metadata_handle.index(),
            fetch.filtered_indices
        );
        Ok(self.tx.send(RowGroupBuffer {
            metadata: fetch.row_group_metadata_handle,
            bytes: buffer,
            base_offset: request.location.offset as u64,
            filtered_indices: Some(fetch.filtered_indices),
        })?)
    }

    fn try_finish(&mut self) -> crate::operations::Result<bool> {
        Ok(self.rx.is_empty() && self.waiting == 0)
    }

    fn try_steal_io_request(&mut self) -> crate::operations::Result<Option<IORequest>> {
        if self.current_row_group.is_none() {
            if let Some(r) = self.rx.steal() {
                self.current_row_group = Some(r);
                return self.next_io_request();
            }
        }
        Ok(None)
    }

    fn try_steal_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
        if self.current_row_group.is_none() {
            if let Some(r) = self.rx.steal() {
                self.current_row_group = Some(r);
                return self.run_cpu_work();
            }
        }
        Ok(WorkStatus::Pending)
    }
}
