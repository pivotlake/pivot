//! A live data file as maintenance takes it: where it is and how large,
//! its row count, its commit and its bounds on the table's sort key, and
//! nothing else of its manifest entry.
//!
//! Maintenance holds one of these for every file of a table between its
//! rounds, whatever the cache keeps of the manifests they were read from,
//! so a record is small: it holds the bounds of the sort columns alone,
//! each as the value its manifest recorded.
//!
//! What reading a bound takes is resolved once for the records that share
//! it: the columns the table is sorted by, each with the type its values
//! have now (`SortColumns`).

use std::sync::Arc;

use iceberg::spec::{PrimitiveLiteral, Schema, TableMetadata};
use object_storage::FileRef;
use parquet_engine::FileStats;
use planner::catalog::Column;
use planner::types::{MAX_DECIMAL64_PRECISION, Type};

use crate::files::data_files::literal_to_scalar;
use crate::files::manifest_parts::BoundSide;

/// The bound of one file on one sort column, as a value of the column's
/// current type, borrowed from the file's record. It says what the bound
/// is and nothing of where it stands among others: whoever orders files by
/// their bounds has the rules for that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SortBound<'f> {
    /// No bound: the file's manifest records none for the column, or
    /// something that does not read as a value of the column's type.
    Missing,
    Boolean(bool),
    /// An integer of any width, a date as its days and a timestamp as its
    /// microseconds since the epoch, or the unscaled value of a decimal at
    /// its column's scale.
    Integer(i128),
    /// A float recorded in single precision, as it was recorded, whatever
    /// width its column has now.
    SinglePrecision(f32),
    /// A float recorded in double precision.
    DoublePrecision(f64),
    String(&'f str),
}

/// One live data file of a table version with its bounds on the columns
/// the table is sorted by.
pub struct SortBoundedFile {
    /// The file's path as its table names it, and its size.
    file: FileRef,
    /// The rows the file holds, deleted ones included.
    row_count: u64,
    /// Where the file's commit stands among the table's commits.
    commit_sequence: u64,
    /// The file's lower bound and then its upper bound on each sort column,
    /// in sort order: `None` for a bound the file's manifest does not
    /// record, or records under a schema that does not have the column as
    /// a primitive.
    bounds: Box<[Option<PrimitiveLiteral>]>,
    sort_columns: Arc<SortColumns>,
}

impl SortBoundedFile {
    /// `file`, of `row_count` rows, which the commit standing at
    /// `commit_sequence` among its table's added, bounded on the columns of
    /// `sort_columns` by `bounds`: its lower and then its upper bound on
    /// each, in sort order.
    pub(crate) fn new(
        file: FileRef,
        row_count: u64,
        commit_sequence: u64,
        bounds: Vec<Option<PrimitiveLiteral>>,
        sort_columns: &Arc<SortColumns>,
    ) -> Self {
        Self {
            file,
            row_count,
            commit_sequence,
            bounds: bounds.into_boxed_slice(),
            sort_columns: Arc::clone(sort_columns),
        }
    }

    /// The bound on `side` of the sort column at `position`.
    fn find_bound(&self, position: usize, side: BoundSide) -> Option<&PrimitiveLiteral> {
        let side_offset = match side {
            BoundSide::Lower => 0,
            BoundSide::Upper => 1,
        };
        self.bounds[2 * position + side_offset].as_ref()
    }

    /// The file's row count and its bounds on the sort columns, as the
    /// datastore keys statistics: by column name, as typed scalars of the
    /// column's Pivot type. A column is bounded only when both of its
    /// bounds read as values of that type. No null count is held.
    pub fn read_sort_key_stats(&self) -> FileStats {
        let mut stats = FileStats {
            num_records: i64::try_from(self.row_count).ok(),
            min_values: Default::default(),
            max_values: Default::default(),
            null_counts: Default::default(),
        };
        for position in 0..self.sort_columns.len() {
            let read = |side: BoundSide| {
                let bound = self.find_bound(position, side)?;
                self.sort_columns.read_scalar(position, bound)
            };
            if let (Some(min), Some(max)) = (read(BoundSide::Lower), read(BoundSide::Upper)) {
                let name = self.sort_columns.name(position);
                stats.min_values.insert(name.to_string(), min);
                stats.max_values.insert(name.to_string(), max);
            }
        }
        stats
    }

    /// The file's path as its table names it, and its size.
    pub fn file(&self) -> &FileRef {
        &self.file
    }

    /// The rows the file holds, deleted ones included.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Where the file's commit stands among the table's commits: a later
    /// commit's is not below an earlier one's.
    pub fn commit_sequence(&self) -> u64 {
        self.commit_sequence
    }

    /// How many columns the file's table is sorted by.
    pub fn count_sort_columns(&self) -> usize {
        self.sort_columns.len()
    }

    /// The file's bound on `side` of the sort column at `position`, which
    /// is below [`count_sort_columns`](Self::count_sort_columns), as a
    /// value of the column's current type. The same value for as long as
    /// the record lives.
    pub fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortBound<'_> {
        match self.find_bound(position, side) {
            Some(bound) => self.sort_columns.read_value(position, bound),
            None => SortBound::Missing,
        }
    }
}

/// The columns a table is sorted by, in sort order, with what reading a
/// file's bound on each takes, resolved once for the records that share
/// it.
pub(crate) struct SortColumns {
    columns: Vec<SortColumn>,
}

/// One column a table is sorted by.
struct SortColumn {
    name: String,
    field_id: i32,
    /// The Pivot type the column has now, which a bound on it is read as a
    /// value of.
    column_type: Type,
    /// The ids of the schemas the table lists that have the field as a
    /// primitive, in order: a file whose manifest was written under
    /// another has no bound on the column that reads.
    stored_in_schemas: Vec<i32>,
}

impl SortColumns {
    /// The sort columns of a table sorted by `sort_by`, columns among
    /// `columns` whose field ids `metadata`'s current schema gives.
    pub(crate) fn new(metadata: &TableMetadata, columns: &[Column], sort_by: &[String]) -> Self {
        let schema = metadata.current_schema();
        let columns = sort_by
            .iter()
            .filter_map(|name| {
                let field_id = schema.field_id_by_name(name)?;
                let column = columns.iter().find(|column| &column.name == name)?;
                let mut stored_in_schemas: Vec<i32> = metadata
                    .schemas_iter()
                    .filter(|stored| holds_primitive_field(stored, field_id))
                    .map(|stored| stored.schema_id())
                    .collect();
                stored_in_schemas.sort_unstable();
                Some(SortColumn {
                    name: name.clone(),
                    field_id,
                    column_type: column.col_type.clone(),
                    stored_in_schemas,
                })
            })
            .collect();
        Self { columns }
    }

    /// How many columns the table is sorted by.
    pub(crate) fn len(&self) -> usize {
        self.columns.len()
    }

    /// The ids of the fields the table is sorted by, in sort order.
    pub(crate) fn field_ids(&self) -> impl Iterator<Item = i32> {
        self.columns.iter().map(|column| column.field_id)
    }

    /// The name of the sort column at `position`.
    pub(crate) fn name(&self, position: usize) -> &str {
        &self.columns[position].name
    }

    /// Whether a manifest written under the schema `schema_id` bounds the
    /// sort column at `position` by values that read: the table lists the
    /// schema, and it has the column as a primitive.
    pub(crate) fn is_bounded_under(&self, position: usize, schema_id: i32) -> bool {
        let stored_in_schemas = &self.columns[position].stored_in_schemas;
        stored_in_schemas.binary_search(&schema_id).is_ok()
    }

    /// `bound`, what a manifest recorded for the sort column at `position`,
    /// as a value of the column's current type.
    pub(crate) fn read_value<'f>(
        &self,
        position: usize,
        bound: &'f PrimitiveLiteral,
    ) -> SortBound<'f> {
        read_sort_bound(bound, &self.columns[position].column_type)
    }

    /// `bound`, what a manifest recorded for the sort column at `position`,
    /// as the datastore holds a statistic: a typed scalar of the column's
    /// Pivot type; `None` for a bound that does not read as one.
    pub(crate) fn read_scalar(
        &self,
        position: usize,
        bound: &PrimitiveLiteral,
    ) -> Option<arrow_array::ArrayRef> {
        literal_to_scalar(bound, &self.columns[position].column_type)
            .map(|scalar| scalar.into_inner())
    }
}

/// `literal`, a bound a manifest recorded for a column whose type is now
/// `column_type`, as a value of that type. It reads for the pairings
/// [`literal_to_scalar`] reads, an integer only when the column's type
/// holds it, and for a bound recorded in the narrower type its column had
/// before it was widened, an `int` under a `long` column and a `float`
/// under a `double` one, which reads as its value. Any other bound is
/// missing.
fn read_sort_bound<'f>(literal: &'f PrimitiveLiteral, column_type: &Type) -> SortBound<'f> {
    let read_integer = |value: i64, lowest: i64, highest: i64| {
        if (lowest..=highest).contains(&value) {
            SortBound::Integer(i128::from(value))
        } else {
            SortBound::Missing
        }
    };
    match (literal, column_type) {
        (PrimitiveLiteral::Boolean(value), Type::Boolean) => SortBound::Boolean(*value),
        (PrimitiveLiteral::Int(value), Type::Int8) => {
            read_integer(i64::from(*value), i64::from(i8::MIN), i64::from(i8::MAX))
        }
        (PrimitiveLiteral::Int(value), Type::Int16) => {
            read_integer(i64::from(*value), i64::from(i16::MIN), i64::from(i16::MAX))
        }
        (PrimitiveLiteral::Int(value), Type::UInt8) => {
            read_integer(i64::from(*value), 0, i64::from(u8::MAX))
        }
        (PrimitiveLiteral::Int(value), Type::UInt16) => {
            read_integer(i64::from(*value), 0, i64::from(u16::MAX))
        }
        (PrimitiveLiteral::Int(value), Type::Int32 | Type::Date | Type::Int64) => {
            SortBound::Integer(i128::from(*value))
        }
        (PrimitiveLiteral::Long(value), Type::UInt32) => {
            read_integer(*value, 0, i64::from(u32::MAX))
        }
        (PrimitiveLiteral::Long(value), Type::UInt64) => read_integer(*value, 0, i64::MAX),
        (PrimitiveLiteral::Long(value), Type::Int64 | Type::Timestamp | Type::TimestampTz) => {
            SortBound::Integer(i128::from(*value))
        }
        (PrimitiveLiteral::Float(value), Type::Float32 | Type::Float64) => {
            SortBound::SinglePrecision(value.0)
        }
        (PrimitiveLiteral::Double(value), Type::Float64) => SortBound::DoublePrecision(value.0),
        (PrimitiveLiteral::String(value), Type::Utf8) => SortBound::String(value),
        (PrimitiveLiteral::Int128(unscaled), Type::Decimal { precision, .. }) => {
            let is_held = *precision > MAX_DECIMAL64_PRECISION || i64::try_from(*unscaled).is_ok();
            if is_held {
                SortBound::Integer(*unscaled)
            } else {
                SortBound::Missing
            }
        }
        _ => SortBound::Missing,
    }
}

/// Whether `schema` has the field `field_id` as a primitive, so a manifest
/// written under it records bounds for the field.
pub(crate) fn holds_primitive_field(schema: &Schema, field_id: i32) -> bool {
    schema
        .field_by_id(field_id)
        .is_some_and(|field| field.field_type.as_primitive_type().is_some())
}
