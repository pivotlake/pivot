//! One live data file of a Delta table as its table holds it: the entry the
//! log recorded for it, the row groups of its footer, the partition it sits
//! in and where its commit stands among the table's.
//!
//! Where a file stands on the table's sort key is read off the statistics
//! its entry holds whenever two files are compared ([`SortColumns`]), so a
//! record holds its bounds in no other form.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use object_storage::FileRef;
use parquet_engine::RowGroupMetadata;
use planner::catalog::Column;

use super::partition_key::PartitionKeyEncoder;
use crate::Result;
use crate::compact::sort_key::{BoundSide, SortValue};
use crate::file_entry::FileEntry;
use crate::table::TableFile;
use crate::table_format::CompactionFile;

/// One live data file of a Delta table version.
pub(crate) struct DeltaFile {
    /// The file as the log recorded it: its path and size, its partition
    /// tuple and its statistics.
    pub entry: FileEntry,
    /// The file's row groups, in the file's own order.
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
    /// The key of the partition the file sits in.
    pub partition: Arc<[u8]>,
    /// Where the file's commit stands among the table's commits.
    commit_sequence: u64,
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

impl CompactionFile for DeltaFile {
    fn file(&self) -> &FileRef {
        &self.entry.file
    }

    /// The row count the log recorded in the file's statistics.
    fn row_count(&self) -> Option<u64> {
        let stats = self.entry.stats.as_deref()?;
        u64::try_from(stats.num_records?).ok()
    }

    fn commit_sequence(&self) -> u64 {
        self.commit_sequence
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

    /// The entry the log recorded, whole.
    fn build_entry(&self) -> FileEntry {
        self.entry.clone()
    }
}

/// A file as a version's set of files holds it: it stands by its path,
/// which it is looked up by and which its record alone holds.
#[derive(Clone)]
pub(crate) struct FileByPath(pub Arc<DeltaFile>);

impl FileByPath {
    fn read_path(&self) -> &str {
        self.0.entry.file.path.as_str()
    }
}

impl PartialEq for FileByPath {
    fn eq(&self, other: &Self) -> bool {
        self.read_path() == other.read_path()
    }
}

impl Eq for FileByPath {}

impl PartialOrd for FileByPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FileByPath {
    fn cmp(&self, other: &Self) -> Ordering {
        self.read_path().cmp(other.read_path())
    }
}

impl Borrow<str> for FileByPath {
    fn borrow(&self) -> &str {
        self.read_path()
    }
}

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
    /// among the table's added, with the key of its partition encoded.
    pub(crate) fn build(&self, file: TableFile, commit_sequence: u64) -> Result<DeltaFile> {
        let TableFile { entry, row_groups } = file;
        let partition = self.partition_keys.encode(entry.partition.as_ref())?;
        Ok(DeltaFile {
            entry,
            row_groups,
            partition: Arc::from(partition),
            commit_sequence,
            sort_columns: Arc::clone(&self.sort_columns),
        })
    }
}
