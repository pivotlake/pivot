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

/// Walk a metadata-tagged batch's row-group runs, calling `visit(group,
/// logical_start, logical_end)` once per run. The row-group column is a
/// `RunArray` (consecutive same-group rows = one run) and the batch may be a
/// logical slice (e.g. a `LIMIT` above the scan), so the walk uses the
/// *logical* run bounds via `RunEndBuffer::sliced_values()` (run ends already
/// adjusted by the slice offset and capped at the slice length) and maps each
/// run to its physical group value from `get_start_physical_index()`. The
/// bounds index the batch's logical rows, so they address the (equally
/// logically indexed) `row_index` column and any per-row mask directly.
pub(crate) fn visit_row_group_runs(batch: &RecordBatch, mut visit: impl FnMut(u32, usize, usize)) {
    let groups = global_row_group(batch);
    let run_ends = groups.run_ends();
    let physical_start = run_ends.get_start_physical_index();
    let group_values = groups
        .values()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();

    let mut logical = 0usize;
    for (run_offset, logical_end) in run_ends.sliced_values().enumerate() {
        let logical_end = logical_end as usize;
        let group = group_values.value(physical_start + run_offset);
        visit(group, logical, logical_end);
        logical = logical_end;
    }
}

/// Add row group metadata to the record batch, receiving a new RecordBatch with a column for global
/// row group, and another for index within the group.
///
/// Note that the global row group column is NOT the row_group within a parquet file, but across
/// the entire table
pub fn with_row_group_metadata(batch: RecordBatch, group: usize, offset: usize) -> RecordBatch {
    // An unfiltered scan emits every row, so the row indices are the dense
    // range starting at the rows already emitted.
    let row_count = batch.num_rows();
    let row_idxs = UInt32Array::from_iter_values(offset as u32..offset as u32 + row_count as u32);
    attach_row_group_metadata(batch, group, row_idxs)
}

/// Like [`with_row_group_metadata`], but for a filtered scan: each emitted row's
/// index is its ORIGINAL position in the row group, taken from the surviving
/// `indices` slice (one entry per batch row, in emission order).
pub fn with_row_group_metadata_from_indices(
    batch: RecordBatch,
    group: usize,
    indices: &[u32],
) -> RecordBatch {
    assert_eq!(
        indices.len(),
        batch.num_rows(),
        "row-index slice must cover exactly the batch rows"
    );
    let row_idxs = UInt32Array::from_iter_values(indices.iter().copied());
    attach_row_group_metadata(batch, group, row_idxs)
}

fn attach_row_group_metadata(
    batch: RecordBatch,
    group: usize,
    row_idxs: UInt32Array,
) -> RecordBatch {
    let (schema, mut columns, row_count) = batch.into_parts();

    // We create the row group as a single repeated value for the entire group
    let run_ends = Int32Array::from(vec![row_count as i32]);
    let values = UInt32Array::from(vec![group as u32]);
    let run_array =
        RunArray::<Int32Type>::try_new(&run_ends, &values).expect("Failed to create RunArray");

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
