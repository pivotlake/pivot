//! One live data file of a Delta table as its table holds it: the facts
//! every format knows of a file, beside the entry the log recorded for it
//! and the row groups of its footer.
//!
//! Where a file stands on the table's sort key is read off the statistics
//! its entry holds whenever two files are compared ([`SortColumns`]), so a
//! record holds its bounds in no other form.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use parquet_engine::RowGroupMetadata;
use planner::catalog::Column;
use table_file_index::{BoundSide, FileFacts, FileIndex, FilesByPath, IndexedRecord, SortValue};

use super::partition_key::PartitionKeyEncoder;
use crate::Result;
use crate::file_entry::FileEntry;
use crate::table::TableFile;

/// One live data file of a Delta table version.
pub(crate) struct DeltaFile {
    facts: FileFacts,
    /// The file as the log recorded it: its partition tuple and its
    /// statistics.
    pub entry: FileEntry,
    /// The file's row groups, in the file's own order.
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
    /// The columns the file's table is sorted by, which the file's bounds
    /// on the sort key are read by: shared by the records one builder made.
    sort_columns: Arc<SortColumns>,
}

impl DeltaFile {
    /// The rows the file's row groups hold.
    pub(crate) fn count_rows(&self) -> u64 {
        self.row_groups
            .iter()
            .map(|row_group| row_group.num_rows.max(0) as u64)
            .sum()
    }
}

impl IndexedRecord for DeltaFile {
    fn facts(&self) -> &FileFacts {
        &self.facts
    }

    fn count_sort_columns(&self) -> usize {
        self.sort_columns.columns.len()
    }

    fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_> {
        let (name, column_type) = &self.sort_columns.columns[position];
        let Some(stats) = self.entry.stats.as_deref() else {
            return SortValue::Missing;
        };
        let bounds = match side {
            BoundSide::Lower => &stats.min_values,
            BoundSide::Upper => &stats.max_values,
        };
        match bounds.get(name) {
            Some(bound) => read_sort_value(bound, column_type),
            None => SortValue::Missing,
        }
    }

    /// Zero: nothing budgets a Delta record, so none is measured.
    fn estimate_heap_bytes(&self) -> u64 {
        0
    }
}

/// Files of a Delta table version, in the order a partition's files stand
/// in.
pub(crate) type DeltaIndex = FileIndex<DeltaFile>;

/// Files of a Delta table version by their paths, in no order.
pub(crate) type DeltaFilesByPath = FilesByPath<DeltaFile>;

/// The columns a table is sorted by, in sort order, each by its name and
/// the Arrow type its values have now, which a bound on it is read as a
/// value of.
struct SortColumns {
    columns: Vec<(String, DataType)>,
}

/// `bound`, what a file's statistics hold on one side of a column whose
/// values are now of `column_type`, as a value of that type.
///
/// A bound recorded in the column's type reads as it is. One recorded in
/// another type reads as the value a cast to the column's type gives it:
/// a string of any string type as its text, an integer of another width
/// when the column's type holds it, a float widened, a decimal of the
/// column's precision and scale kept in the other width, a timestamp under
/// another zone as its instant, and anything else by the cast itself. A
/// NULL, a bound that is not a single value, a bound the cast does not
/// read, and every bound of a column whose type has no [`SortValue`] are
/// missing.
fn read_sort_value<'f>(bound: &'f ArrayRef, column_type: &DataType) -> SortValue<'f> {
    if bound.len() != 1 || bound.is_null(0) {
        return SortValue::Missing;
    }
    if matches!(
        column_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        return read_string(bound).map_or(SortValue::Missing, SortValue::String);
    }
    let value = if casts_to_same_value(bound.data_type(), column_type) {
        read_native_value(bound.as_ref())
    } else {
        arrow_cast::cast(bound.as_ref(), column_type)
            .ok()
            .filter(|cast| !cast.is_null(0))
            .and_then(|cast| read_native_value(cast.as_ref()))
    };
    match value {
        Some(SortValue::Integer(value)) if !holds_integer(column_type, value) => SortValue::Missing,
        Some(value) => value,
        None => SortValue::Missing,
    }
}

/// The text of `bound`, a single string of any string type.
fn read_string(bound: &ArrayRef) -> Option<&str> {
    match bound.data_type() {
        DataType::Utf8 => Some(bound.as_string::<i32>().value(0)),
        DataType::LargeUtf8 => Some(bound.as_string::<i64>().value(0)),
        DataType::Utf8View => Some(bound.as_string_view().value(0)),
        _ => None,
    }
}

/// Whether a cast of a value of `bound_type` to `column_type` keeps the
/// value [`read_native_value`] reads of it, wherever `column_type` holds
/// that value.
fn casts_to_same_value(bound_type: &DataType, column_type: &DataType) -> bool {
    use DataType::{Decimal64, Decimal128, Float32, Float64, Timestamp};
    match (bound_type, column_type) {
        (Float32, Float64) => true,
        (
            Decimal64(bound_precision, bound_scale) | Decimal128(bound_precision, bound_scale),
            Decimal64(precision, scale) | Decimal128(precision, scale),
        ) => bound_precision == precision && bound_scale == scale,
        (Timestamp(TimeUnit::Microsecond, _), Timestamp(TimeUnit::Microsecond, _)) => true,
        _ => bound_type == column_type || (bound_type.is_integer() && column_type.is_integer()),
    }
}

/// The single value of `bound` as its own type has it; `None` for a type
/// with no [`SortValue`].
fn read_native_value(bound: &dyn Array) -> Option<SortValue<'static>> {
    let integer = |value: i128| Some(SortValue::Integer(value));
    match bound.data_type() {
        DataType::Boolean => Some(SortValue::Boolean(bound.as_boolean().value(0))),
        DataType::Int8 => integer(bound.as_primitive::<Int8Type>().value(0).into()),
        DataType::Int16 => integer(bound.as_primitive::<Int16Type>().value(0).into()),
        DataType::Int32 => integer(bound.as_primitive::<Int32Type>().value(0).into()),
        DataType::Int64 => integer(bound.as_primitive::<Int64Type>().value(0).into()),
        DataType::UInt8 => integer(bound.as_primitive::<UInt8Type>().value(0).into()),
        DataType::UInt16 => integer(bound.as_primitive::<UInt16Type>().value(0).into()),
        DataType::UInt32 => integer(bound.as_primitive::<UInt32Type>().value(0).into()),
        DataType::UInt64 => integer(bound.as_primitive::<UInt64Type>().value(0).into()),
        DataType::Date32 => integer(bound.as_primitive::<Date32Type>().value(0).into()),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let micros = bound.as_primitive::<TimestampMicrosecondType>().value(0);
            integer(micros.into())
        }
        DataType::Decimal64(_, _) => integer(bound.as_primitive::<Decimal64Type>().value(0).into()),
        DataType::Decimal128(_, _) => integer(bound.as_primitive::<Decimal128Type>().value(0)),
        DataType::Float32 => {
            let value = bound.as_primitive::<Float32Type>().value(0);
            Some(SortValue::from_single_precision(value))
        }
        DataType::Float64 => Some(SortValue::Float(
            bound.as_primitive::<Float64Type>().value(0),
        )),
        _ => None,
    }
}

/// Whether a column of `column_type` holds the integer `value`, or the
/// decimal whose unscaled value it is.
fn holds_integer(column_type: &DataType, value: i128) -> bool {
    let (lowest, highest): (i128, i128) = match column_type {
        DataType::Int8 => (i8::MIN.into(), i8::MAX.into()),
        DataType::Int16 => (i16::MIN.into(), i16::MAX.into()),
        DataType::Int32 => (i32::MIN.into(), i32::MAX.into()),
        DataType::Int64 | DataType::Decimal64(_, _) => (i64::MIN.into(), i64::MAX.into()),
        DataType::UInt8 => (0, u8::MAX.into()),
        DataType::UInt16 => (0, u16::MAX.into()),
        DataType::UInt32 => (0, u32::MAX.into()),
        DataType::UInt64 => (0, u64::MAX.into()),
        _ => return true,
    };
    (lowest..=highest).contains(&value)
}

/// Builds the records of one table's files: names each file's partition,
/// and has the records it builds share the columns the table is sorted by.
pub(crate) struct RecordBuilder {
    sort_columns: Arc<SortColumns>,
    partition_keys: PartitionKeyEncoder,
}

impl RecordBuilder {
    /// The builder for a table of `columns`, partitioned by `partition_by`
    /// and sorted by `sort_by`.
    pub(crate) fn new(
        columns: &[Column],
        partition_by: &[String],
        sort_by: &[String],
    ) -> Result<Self> {
        let sort_columns = sort_by
            .iter()
            .filter_map(|name| {
                let column = columns.iter().find(|column| &column.name == name)?;
                let arrow_type = planner::types::physical_arrow_type(&column.col_type);
                Some((name.clone(), arrow_type))
            })
            .collect();
        Ok(Self {
            sort_columns: Arc::new(SortColumns {
                columns: sort_columns,
            }),
            partition_keys: PartitionKeyEncoder::new(columns, partition_by)?,
        })
    }

    /// The record of `file`, which the commit standing at `commit_sequence`
    /// among the table's added, beside the bytes its partition is named by.
    pub(crate) fn build(
        &self,
        file: TableFile,
        commit_sequence: u64,
    ) -> Result<(Vec<u8>, DeltaFile)> {
        let TableFile { entry, row_groups } = file;
        let partition = self.partition_keys.encode(entry.partition.as_ref())?;
        let rows = entry
            .stats
            .as_deref()
            .and_then(|stats| stats.num_records)
            .and_then(|rows| u64::try_from(rows).ok());
        let record = DeltaFile {
            facts: FileFacts::new(entry.file.clone(), rows, commit_sequence),
            entry,
            row_groups,
            sort_columns: Arc::clone(&self.sort_columns),
        };
        Ok((partition, record))
    }
}
