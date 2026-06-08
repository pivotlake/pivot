use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, RunArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::{Arc, LazyLock};

const GLOBAL_ROW_GROUP_OFFSET_FROM_END: usize = 2;

const ROW_OFFSET_FROM_END: usize = 1;

static GLOBAL_ROW_GROUP_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
    let run_ends = Field::new("run_ends", DataType::Int32, false);
    let values = Field::new("values", DataType::UInt32, true);
    let dt = DataType::RunEndEncoded(Arc::new(run_ends), Arc::new(values));
    Arc::new(Field::new(dispatch::ROW_GROUP_IDX_FIELD, dt, false))
});

static ROW_IDX_FIELD: LazyLock<Arc<Field>> =
    LazyLock::new(|| Arc::new(Field::new(dispatch::ROW_IDX_FIELD, DataType::UInt32, false)));

/// The global row group column for a particular RecordBatch. Note that the global row group column
/// is NOT the row_group within a parquet file, but across
/// the entire table
pub fn global_row_group(batch: &RecordBatch) -> &RunArray<Int32Type> {
    batch
        .column(batch.num_columns() - GLOBAL_ROW_GROUP_OFFSET_FROM_END)
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .expect("Metadata not set correctly")
}

/// Gets the array for the index of the row within the row group
pub fn row_index(batch: &RecordBatch) -> &UInt32Array {
    batch
        .column(batch.num_columns() - ROW_OFFSET_FROM_END)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("Metadata not set correctly")
}

/// Add row group metadata to the record batch, receiving a new RecordBatch with a column for global
/// row group, and another for index within the group.
///
/// Note that the global row group column is NOT the row_group within a parquet file, but across
/// the entire table
pub fn with_row_group_metadata(batch: RecordBatch, group: usize, offset: usize) -> RecordBatch {
    let (schema, mut columns, row_count) = batch.into_parts();

    // We create the row group as a single repeated value for the entire group
    let run_ends = Int32Array::from(vec![row_count as i32]);
    let values = UInt32Array::from(vec![group as u32]);
    let run_array =
        RunArray::<Int32Type>::try_new(&run_ends, &values).expect("Failed to create RunArray");

    // These are the indexes of the rows- they are unique across the entire RowGroup (this is why
    // offset must be supplied)
    let row_idxs = UInt32Array::from_iter_values(offset as u32..offset as u32 + row_count as u32);

    columns.push(Arc::new(run_array));
    columns.push(Arc::new(row_idxs));

    let new_schema = Arc::new(Schema::new(
        schema
            .fields()
            .to_vec()
            .into_iter()
            .chain(vec![
                Arc::clone(&GLOBAL_ROW_GROUP_FIELD),
                Arc::clone(&ROW_IDX_FIELD),
            ])
            .collect::<Vec<_>>(),
    ));

    unsafe { RecordBatch::new_unchecked(new_schema, columns, row_count) }
}
