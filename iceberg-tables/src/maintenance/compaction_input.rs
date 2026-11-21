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

use std::ops::RangeInclusive;
use std::sync::Arc;

use iceberg::spec::{PrimitiveLiteral, TableMetadata};
use object_storage::FileRef;
use parquet_engine::{BoundSide, FileStats, SortValue};
use planner::catalog::Column;
use planner::types::{MAX_DECIMAL64_PRECISION, Type};

use crate::types::datum::literal_to_scalar;
use crate::types::schema::holds_primitive_field;

/// One live data file of a table version with its bounds on the columns
/// the table is sorted by. The record names no manifest: the commit that
/// removes the file finds it by partition and path in the version it
/// commits on (`Table::find_files`, which says why).
pub struct CompactionInput {
    /// The file's path as its table names it, and its size.
    pub(crate) file: FileRef,
    /// The rows the file holds, deleted ones included.
    pub(crate) row_count: u64,
    /// Where the file's commit stands among the table's commits.
    pub(crate) commit_sequence: u64,
    /// The file's lower bound and then its upper bound on each sort column,
    /// in sort order: `None` for a bound the file's manifest does not
    /// record, or records under a schema that does not have the column as
    /// a primitive.
    pub(crate) bounds: Box<[Option<PrimitiveLiteral>]>,
    pub(crate) sort_columns: Arc<SortColumns>,
}

impl CompactionInput {
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
    pub fn build_sort_key_stats(&self) -> FileStats {
        let mut stats = FileStats {
            num_records: i64::try_from(self.row_count).ok(),
            min_values: Default::default(),
            max_values: Default::default(),
            null_counts: Default::default(),
        };
        for (position, column) in self.sort_columns.columns.iter().enumerate() {
            let read = |side: BoundSide| {
                let bound = self.find_bound(position, side)?;
                let scalar = literal_to_scalar(bound, &column.column_type)?;
                Some(scalar.into_inner())
            };
            if let (Some(min), Some(max)) = (read(BoundSide::Lower), read(BoundSide::Upper)) {
                stats.min_values.insert(column.name.clone(), min);
                stats.max_values.insert(column.name.clone(), max);
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
        self.sort_columns.columns.len()
    }

    /// The file's bound on `side` of the sort column at `position`, which
    /// is below [`count_sort_columns`](Self::count_sort_columns), as a
    /// value of the column's current type. The same value for as long as
    /// the record lives.
    pub fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_> {
        match self.find_bound(position, side) {
            Some(bound) => self.sort_columns.columns[position].build_sort_value(bound),
            None => SortValue::Missing,
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

    /// The ids of the fields the table is sorted by, in sort order.
    pub(crate) fn field_ids(&self) -> impl Iterator<Item = i32> {
        self.columns.iter().map(|column| column.field_id)
    }

    /// Whether a manifest written under the schema `schema_id` bounds the
    /// sort column at `position` by values that read: the table lists the
    /// schema, and it has the column as a primitive.
    pub(crate) fn is_bounded_under(&self, position: usize, schema_id: i32) -> bool {
        let stored_in_schemas = &self.columns[position].stored_in_schemas;
        stored_in_schemas.binary_search(&schema_id).is_ok()
    }
}

impl SortColumn {
    /// The integers the column holds, when its type is one a manifest
    /// bounds by an `int` or a `long`: an integer type, a date or a
    /// timestamp. An unsigned 64-bit column is bounded by a `long`, which
    /// holds none of its values above the highest `long`.
    fn find_integer_range(&self) -> Option<RangeInclusive<i64>> {
        Some(match &self.column_type {
            Type::Int8 => i64::from(i8::MIN)..=i64::from(i8::MAX),
            Type::Int16 => i64::from(i16::MIN)..=i64::from(i16::MAX),
            Type::Int32 | Type::Date => i64::from(i32::MIN)..=i64::from(i32::MAX),
            Type::UInt8 => 0..=i64::from(u8::MAX),
            Type::UInt16 => 0..=i64::from(u16::MAX),
            Type::UInt32 => 0..=i64::from(u32::MAX),
            Type::UInt64 => 0..=i64::MAX,
            Type::Int64 | Type::Timestamp | Type::TimestampTz => i64::MIN..=i64::MAX,
            _ => return None,
        })
    }

    /// The compacter's value for the column of `literal`, a bound a
    /// manifest recorded for it: [`SortValue::Missing`] when the bound does
    /// not read as a value of the type the column has now. A bound reads
    /// for the pairings [`literal_to_scalar`] reads, an integer only when
    /// the column's type holds it: among them a bound recorded in the
    /// narrower type the column had before it was widened, an `int` under
    /// a `long` column and a `float` under a `double` one, which reads as
    /// its value. A float recorded in single precision is widened as every
    /// bound of that width is ([`SortValue::from_single_precision`]).
    fn build_sort_value<'f>(&self, literal: &'f PrimitiveLiteral) -> SortValue<'f> {
        let build_integer_value = |value: i64| {
            let is_held = self
                .find_integer_range()
                .is_some_and(|held| held.contains(&value));
            if is_held {
                SortValue::Integer(i128::from(value))
            } else {
                SortValue::Missing
            }
        };
        match (literal, &self.column_type) {
            (PrimitiveLiteral::Boolean(value), Type::Boolean) => SortValue::Boolean(*value),
            (
                PrimitiveLiteral::Int(value),
                Type::Int8
                | Type::Int16
                | Type::UInt8
                | Type::UInt16
                | Type::Int32
                | Type::Date
                | Type::Int64,
            ) => build_integer_value(i64::from(*value)),
            (
                PrimitiveLiteral::Long(value),
                Type::UInt32 | Type::UInt64 | Type::Int64 | Type::Timestamp | Type::TimestampTz,
            ) => build_integer_value(*value),
            (PrimitiveLiteral::Float(value), Type::Float32 | Type::Float64) => {
                SortValue::from_single_precision(value.0)
            }
            (PrimitiveLiteral::Double(value), Type::Float64) => SortValue::Float(value.0),
            (PrimitiveLiteral::String(value), Type::Utf8) => SortValue::String(value),
            (PrimitiveLiteral::Int128(unscaled), Type::Decimal { precision, .. }) => {
                let is_held =
                    *precision > MAX_DECIMAL64_PRECISION || i64::try_from(*unscaled).is_ok();
                if is_held {
                    SortValue::Integer(*unscaled)
                } else {
                    SortValue::Missing
                }
            }
            _ => SortValue::Missing,
        }
    }
}
