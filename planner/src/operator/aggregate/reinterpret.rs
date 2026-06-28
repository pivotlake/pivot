//! One place to reinterpret selected columns of a batch to another arrow type.
//!
//! Used wherever pivot crosses the int/temporal boundary: the scan that
//! introduces `Date32`/`Timestamp`, the late-materialization fetch that does the
//! same for late-read columns, and the aggregate's int-in / temporal-out
//! boundary. The cast between an int and its same-width temporal type is
//! zero-copy (it shares the underlying buffer); only the per-column schema is
//! rebuilt.

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// Rebuild `batch`, replacing each column `i` whose `target(i, current_type)`
/// returns a different `DataType` with that column cast to the target. Columns
/// with a `None` target, or whose type already equals the target, pass through
/// untouched (no copy).
pub(crate) fn reinterpret_columns(
    batch: RecordBatch,
    target: impl Fn(usize, &DataType) -> Option<DataType>,
) -> RecordBatch {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for i in 0..batch.num_columns() {
        let col = batch.column(i);
        match target(i, col.data_type()) {
            Some(dt) if col.data_type() != &dt => {
                let field = schema.field(i);
                fields.push(Field::new(field.name(), dt.clone(), field.is_nullable()));
                columns.push(arrow::compute::cast(col, &dt).expect("reinterpret cast"));
            }
            _ => {
                fields.push(schema.field(i).clone());
                columns.push(col.clone());
            }
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("reinterpret schema")
}

/// The backing int a temporal arrow column drops to (`Date32 -> Int32`,
/// `Timestamp -> Int64`), or `None` for a non-temporal column. This is the
/// group-by boundary coercion: the only place a temporal column becomes an int,
/// so the row encoder and int reducers can compute on the day/second count.
pub(crate) fn temporal_to_int(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::Date32 => Some(DataType::Int32),
        DataType::Timestamp(_, _) => Some(DataType::Int64),
        _ => None,
    }
}
