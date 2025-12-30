use crate::io::OperationIOSubmitter;
use crate::table::{RowGroupMetadataHandle, Table};
use crate::{ConsumeContext, Operation};
use arrow::compute::filter_record_batch;
use arrow_array::{Array, RecordBatch, UInt32Array};
use parquetd::Projection;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use crate::record_batch_metadata::{global_row_group, row_index};

/// Filter a record batch by a list of global indexes within the row group. Every `dispatch`
/// RecordBatch should have a row index column; the `global_indexes` are meant to match that.
fn filter_batch_by_global_indexes(batch: &RecordBatch, global_indexes: &[u32]) -> RecordBatch {
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
    filter_record_batch(batch, &filter_mask).unwrap()
}

/// The `Materializer` materializes a new projection from the Parquet reader for rows published
/// to it.
///
/// Essentially, for every row group the materializer gets, it will send out a request to the parquet
/// reader. The parquet reader will return that entire row group, and the Materializer will filter
/// only the rows that match the rows the materializer originally got from its publisher.
pub struct Materializer {
    projection: Option<Projection>,
    row_groups_to_indices: HashMap<u32, Vec<u32>>,
    table: Arc<Table>,
}

impl Materializer {
    pub fn new(projection: Option<Projection>, table: Arc<Table>) -> Self {
        Self {
            projection,
            table,
            row_groups_to_indices: Default::default(),
        }
    }
}

impl Operation for Materializer {
    fn consume(
        &mut self,
        context: &ConsumeContext,
        mut io_submitter: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> Option<RecordBatch> {
        match context {
            ConsumeContext::Publisher => {
                let groups = global_row_group(batch);

                let row_indexes = row_index(batch);

                let run_values = groups
                    .values()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .expect("RunArray values must be UInt32Array");

                for i in 0..batch.num_rows() {
                    let logical_index = groups.get_physical_index(i);
                    let group = if run_values.is_valid(logical_index) {
                        Some(run_values.value(logical_index))
                    } else {
                        panic!("unsupported")
                    };

                    let row_index = row_indexes.value(i);
                    match self.row_groups_to_indices.entry(group.unwrap()) {
                        Entry::Occupied(mut e) => {
                            e.get_mut().push(row_index);
                        }
                        Entry::Vacant(e) => {
                            io_submitter
                                .submit_operation_io(
                                    RowGroupMetadataHandle::new(
                                        self.table.clone(),
                                        group.unwrap() as usize,
                                    ),
                                    self.projection.as_ref(),
                                    Box::new(group.unwrap()),
                                )
                                .unwrap();
                            e.insert(vec![row_index]);
                        }
                    }
                }
                None
            }
            ConsumeContext::IORequest(value) => {
                let row_group = value.downcast_ref::<u32>().unwrap();
                let batch = filter_batch_by_global_indexes(
                    batch,
                    self.row_groups_to_indices.get(row_group).unwrap(),
                );
                if batch.num_rows() == 0 {
                    None
                } else {
                    Some(batch)
                }
            }
        }
    }
}
