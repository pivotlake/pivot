//! Encode selected columns at one row of a `RecordBatch` into a one-row
//! arrow-json object — the shared primitive behind partition tuples
//! (`{"ServiceName":"svc-a"}`) and sort-key bounds (`{"Timestamp":100}`).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use serde_json::Value;

use crate::parquet::writing::error::WriteResult;

/// The values of `columns` at `row` of `batch`, as a JSON object keyed by column
/// name.
pub(super) fn row_object(
    batch: &RecordBatch,
    columns: &[String],
    row: usize,
) -> WriteResult<Value> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(columns.len());
    let mut values = Vec::with_capacity(columns.len());
    for name in columns {
        let i = schema.index_of(name)?;
        fields.push(schema.field(i).as_ref().clone());
        values.push(batch.column(i).slice(row, 1));
    }
    let one = RecordBatch::try_new(Arc::new(Schema::new(fields)), values)?;

    let mut buf = Vec::new();
    let mut writer = arrow_json::ArrayWriter::new(&mut buf);
    writer.write(&one)?;
    writer.finish()?;
    // `ArrayWriter` emits a JSON array of row objects; we wrote exactly one row.
    let rows: Value = serde_json::from_slice(&buf)?;
    Ok(rows
        .as_array()
        .and_then(|a| a.first())
        .cloned()
        .expect("arrow-json emits one object for a one-row batch"))
}
