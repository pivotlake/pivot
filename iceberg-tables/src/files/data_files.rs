//! One data file as the datastore holds it, and as a manifest records it.
//!
//! The datastore keeps a file by its store identity, its partition tuple and
//! its column statistics as Pivot's typed Arrow scalars, the shape every
//! writer and the compaction planner work with. A manifest records the same
//! file by full URI with Iceberg values keyed by field id. Recording a file
//! here and reading it back from its table are inverses, so a file written,
//! read back and re-emitted by a later commit is recorded unchanged. A file is recorded under the table's
//! default partition spec and read back under the spec it was written
//! under, so a file of an earlier spec is read and never re-emitted.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Datum as ArrowDatum;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PartitionSpec,
    PrimitiveLiteral, PrimitiveType, Schema, Struct, TableMetadata, Transform,
};
use object_storage::FileRef;
use parquet_engine::{FileStats, PartitionValues};

use crate::metadata::layout::TableLocation;
use crate::types::datum::array_to_datum;
use crate::{Error, Result};

/// String bounds of a column outside the table's sort order are cut to this
/// many code points: such a bound only has to bracket the file's values, and
/// a long string would otherwise be recorded in full in every manifest entry
/// of the file.
const STRING_BOUND_CODE_POINTS: usize = 16;

/// One data file a commit adds: its store identity, the partition tuple a
/// partitioned INSERT stamps on it (`None` for an unpartitioned table; keyed
/// by partition field name, each value in the type the field's transform
/// produces), its statistics, and the sort order its rows are in. A commit
/// writes it under the table's default partition spec.
#[derive(Clone)]
pub struct AddedDataFile {
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    /// The file's statistics: its row count, and per column the bounds and
    /// null count. Written into the manifest, so a file that has none cannot
    /// be committed.
    pub stats: Option<Arc<FileStats>>,
    /// The id of the table sort order the file's rows are in, as its manifest
    /// entry records it: [`UNSORTED_ORDER_ID`](crate::types::schema::UNSORTED_ORDER_ID)
    /// for a file written in no order, `None` for one whose entry says
    /// nothing, which every reader takes as unsorted: a file adopted as it
    /// lay, whose order nobody knows. A file a writer of this process
    /// ordered gets its id from
    /// [`Table::find_sort_order_id_of_rows_sorted_by`](crate::Table::find_sort_order_id_of_rows_sorted_by).
    pub sort_order_id: Option<i32>,
}

/// The Iceberg primitive type of `schema`'s top-level field `name`, and its
/// id.
fn find_primitive_field(schema: &Schema, name: &str) -> Option<(i32, PrimitiveType)> {
    let field = schema.field_by_name(name)?;
    let primitive = field.field_type.as_primitive_type()?.clone();
    Some((field.id, primitive))
}

/// Per field of `partition_spec`, a partition spec of the table `metadata`
/// describes, in spec order, the Iceberg type of its values: the field's
/// transform's result over its source field in the current schema. The
/// error is the reason the spec's values have no type: a transform that
/// does not apply to its source field, or a field whose values are not
/// primitive.
pub(crate) fn read_partition_value_types(
    metadata: &TableMetadata,
    partition_spec: &PartitionSpec,
) -> std::result::Result<Vec<PrimitiveType>, String> {
    let partition_type = partition_spec
        .partition_type(metadata.current_schema())
        .map_err(|error| error.to_string())?;
    partition_type
        .fields()
        .iter()
        .map(|field| {
            field
                .field_type
                .as_primitive_type()
                .cloned()
                .ok_or_else(|| format!("partition field `{}` is not primitive", field.name))
        })
        .collect()
}

impl AddedDataFile {
    /// This entry as a manifest of the table `metadata` describes records
    /// it: its field ids from the current schema, its tuple laid out by the
    /// default partition spec, its path as a URI under `location`. Requires
    /// the entry's statistics: a manifest entry carries the file's row
    /// count, and a file without one cannot be committed.
    ///
    /// A column of the default sort order is bounded by the minimum and
    /// maximum the statistics hold, whole, however long: where a file lies
    /// on the sort key is read from them, and bounds cut to a prefix give
    /// the files whose values share it one range, so files that lie apart
    /// read as overlapping, as do the files a rewrite of them writes. A
    /// string bound of any other column is cut to
    /// `STRING_BOUND_CODE_POINTS` code points.
    pub fn create_data_file(
        &self,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<DataFile> {
        let schema = metadata.current_schema();
        let sort_column_ids: Vec<i32> = metadata
            .default_sort_order()
            .fields
            .iter()
            .map(|field| field.source_id)
            .collect();
        let stats = self.stats.as_ref().ok_or_else(|| Error::MissingFileStats {
            file: self.file.path.to_string(),
        })?;
        let record_count = stats
            .num_records
            .and_then(|count| u64::try_from(count).ok())
            .ok_or_else(|| Error::MissingFileStats {
                file: self.file.path.to_string(),
            })?;
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        let mut null_value_counts = HashMap::new();
        for (name, min) in &stats.min_values {
            let Some(max) = stats.max_values.get(name) else {
                continue;
            };
            let Some((field_id, primitive)) = find_primitive_field(schema, name) else {
                continue;
            };
            // A bound Iceberg cannot hold (a `UInt64` above `i64::MAX`, a NaN)
            // leaves the column unbounded for that file, which prunes nothing.
            let (Some(lower), Some(upper)) = (
                array_to_datum(min.as_ref(), &primitive),
                array_to_datum(max.as_ref(), &primitive),
            ) else {
                continue;
            };
            let (lower, upper) = if sort_column_ids.contains(&field_id) {
                (lower, upper)
            } else {
                // A cut maximum without a successor leaves the column
                // unbounded too.
                let Some(upper) = truncate_upper(upper) else {
                    continue;
                };
                (truncate_lower(lower), upper)
            };
            lower_bounds.insert(field_id, lower);
            upper_bounds.insert(field_id, upper);
        }
        for (name, nulls) in &stats.null_counts {
            if let (Some((field_id, _)), Ok(nulls)) =
                (find_primitive_field(schema, name), u64::try_from(*nulls))
            {
                null_value_counts.insert(field_id, nulls);
            }
        }
        let partition = self.build_partition_struct(metadata)?;
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(location.build_uri_for_path(&self.file.path)?)
            .file_format(DataFileFormat::Parquet)
            .partition(partition)
            .partition_spec_id(metadata.default_partition_spec().spec_id())
            .record_count(record_count)
            .file_size_in_bytes(self.file.size)
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds)
            .null_value_counts(null_value_counts);
        if let Some(sort_order_id) = self.sort_order_id {
            builder.sort_order_id(sort_order_id);
        }
        builder.build().map_err(|error| Error::InvalidDataFile {
            file: self.file.path.to_string(),
            message: error.to_string(),
        })
    }

    /// The entry's partition tuple laid out as the spec orders its fields,
    /// each value in the Iceberg type its field's transform produces. An
    /// identity field the tuple lacks is a NULL partition value (a column the
    /// file does not have reads as NULL); a derived field's value must be
    /// there. A partitioned table's entry must carry a tuple: an Iceberg
    /// tuple has no "unknown" value, and an all-null tuple would make every
    /// other reader prune the file wrongly.
    fn build_partition_struct(&self, metadata: &TableMetadata) -> Result<Struct> {
        let fields = metadata.default_partition_spec().fields();
        if fields.is_empty() {
            return Ok(Struct::empty());
        }
        let tuple = self
            .partition
            .as_ref()
            .ok_or_else(|| Error::PartitionNotConstant {
                file: self.file.path.to_string(),
            })?;
        let invalid = |message: String| Error::InvalidDataFile {
            file: self.file.path.to_string(),
            message,
        };
        let value_types = read_partition_value_types(metadata, metadata.default_partition_spec())
            .map_err(invalid)?;
        let mut values = Vec::with_capacity(fields.len());
        for (field, value_type) in fields.iter().zip(&value_types) {
            let value = match tuple.get(&field.name) {
                Some(scalar) => {
                    let (array, _) = ArrowDatum::get(scalar);
                    if array.is_null(0) {
                        None
                    } else {
                        let datum = array_to_datum(array, value_type).ok_or_else(|| {
                            invalid(format!(
                                "partition value of `{}` has no Iceberg form in `{value_type}`",
                                field.name
                            ))
                        })?;
                        Some(Literal::Primitive(datum.literal().clone()))
                    }
                }
                None if field.transform == Transform::Identity => None,
                None => {
                    return Err(invalid(format!(
                        "the file has no value for partition field `{}`",
                        field.name
                    )));
                }
            };
            values.push(value);
        }
        Ok(Struct::from_iter(values))
    }
}

/// A lower bound, cut to [`STRING_BOUND_CODE_POINTS`] when it is a string:
/// a prefix is at or below every value it was cut from.
fn truncate_lower(datum: Datum) -> Datum {
    match datum.literal() {
        PrimitiveLiteral::String(value) if value.chars().count() > STRING_BOUND_CODE_POINTS => {
            Datum::string(
                value
                    .chars()
                    .take(STRING_BOUND_CODE_POINTS)
                    .collect::<String>(),
            )
        }
        _ => datum,
    }
}

/// An upper bound, cut to [`STRING_BOUND_CODE_POINTS`] when it is a string:
/// the prefix with its last code point incremented is above every value it
/// was cut from. A prefix whose every code point is already the largest has
/// no such successor, which leaves the column unbounded above.
fn truncate_upper(datum: Datum) -> Option<Datum> {
    match datum.literal() {
        PrimitiveLiteral::String(value) if value.chars().count() > STRING_BOUND_CODE_POINTS => {
            let mut prefix: Vec<char> = value.chars().take(STRING_BOUND_CODE_POINTS).collect();
            while let Some(last) = prefix.pop() {
                if let Some(next) = ((last as u32 + 1)..=char::MAX as u32).find_map(char::from_u32)
                {
                    prefix.push(next);
                    return Some(Datum::string(prefix.into_iter().collect::<String>()));
                }
            }
            None
        }
        _ => Some(datum),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_string_bounds_are_cut_to_a_bracketing_prefix() {
        let long = "abcdefghijklmnopqrstuvwxyz";

        let lower = truncate_lower(Datum::string(long));
        let upper = truncate_upper(Datum::string(long)).unwrap();

        assert_eq!(lower, Datum::string("abcdefghijklmnop"));
        assert_eq!(upper, Datum::string("abcdefghijklmnoq"));
        assert_eq!(
            truncate_lower(Datum::string("short")),
            Datum::string("short")
        );
    }
}
