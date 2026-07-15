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

/// Rebuild `batch` with the `Float64` columns at `columns` bit-punned to
/// `Int64` group keys. `-0.0` normalises to `0.0` (`v + 0.0`) so both bit
/// patterns land in one group; equal floats otherwise have equal bits.
pub(crate) fn pun_float_columns_to_bits(batch: RecordBatch, columns: &[usize]) -> RecordBatch {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};

    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut arrays = Vec::with_capacity(batch.num_columns());
    for i in 0..batch.num_columns() {
        let column = batch.column(i);
        if columns.contains(&i) {
            let floats = column.as_primitive::<Float64Type>();
            let bits: arrow_array::PrimitiveArray<Int64Type> =
                floats.unary(|v| (v + 0.0).to_bits() as i64);
            let field = schema.field(i);
            fields.push(Field::new(
                field.name(),
                DataType::Int64,
                field.is_nullable(),
            ));
            arrays.push(Arc::new(bits) as arrow_array::ArrayRef);
        } else {
            fields.push(schema.field(i).clone());
            arrays.push(column.clone());
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).expect("punned schema")
}

/// The inverse of [`pun_float_columns_to_bits`] on the aggregate's output: the
/// `Int64` key columns at `columns` become `Float64` again, bit for bit.
pub(crate) fn unpun_bits_to_float(batch: RecordBatch, columns: &[usize]) -> RecordBatch {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};

    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut arrays = Vec::with_capacity(batch.num_columns());
    for i in 0..batch.num_columns() {
        let column = batch.column(i);
        if columns.contains(&i) {
            let bits = column.as_primitive::<Int64Type>();
            let floats: arrow_array::PrimitiveArray<Float64Type> =
                bits.unary(|v| f64::from_bits(v as u64));
            let field = schema.field(i);
            fields.push(Field::new(
                field.name(),
                DataType::Float64,
                field.is_nullable(),
            ));
            arrays.push(Arc::new(floats) as arrow_array::ArrayRef);
        } else {
            fields.push(schema.field(i).clone());
            arrays.push(column.clone());
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).expect("unpunned schema")
}
