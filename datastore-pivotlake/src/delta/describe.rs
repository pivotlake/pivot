//! A Delta table as the cross-datastore catalog describes it: the columns
//! it declares with what they cost, and the files it holds, read from the
//! log entries and the footers the table holds.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Datum};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::DataType;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use object_storage::FileRef;
use parquet_engine::{FileStats, RowGroupMetadata};
use planner::catalog::SchemaQualifiedTableName;

use super::DeltaTable;
use crate::Result;
use crate::file_entry::PartitionValues;
use crate::table_format::Table;

impl DeltaTable {
    /// This version as the catalog describes it under `name` and `id`, its
    /// files in the order a walk visits them.
    pub(super) fn describe_as(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> Result<DatastoreTableMetadata> {
        let mut walked_files: Vec<FileRef> = Vec::new();
        self.walk_files(&mut |walked| {
            walked_files.push(walked.file.clone());
            ControlFlow::Continue(())
        })?;

        let column_positions: HashMap<&str, usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();
        let mut column_bytes = vec![(0, 0); self.columns.len()];
        let mut files = Vec::with_capacity(walked_files.len());
        let mut total_rows = 0;
        for file in &walked_files {
            let entry = self
                .find_entry(&file.path)?
                .expect("a walked file is live in the version walked");
            let row_groups = self
                .footers
                .get(&file.path)
                .expect("a file of this version has its footer held");
            total_rows += row_groups
                .iter()
                .map(|row_group| row_group.num_rows.max(0) as u64)
                .sum::<u64>();
            files.push(DatastoreFileMetadata {
                path: file.path.as_str().to_string(),
                bytes: file.size,
                bytes_uncompressed: add_file_bytes(
                    &mut column_bytes,
                    &column_positions,
                    row_groups,
                ),
                partition: self.format_partition(&entry.partition),
                min_max_stats: format_min_max_stats(entry.stats.as_deref()),
            });
        }

        let columns = self
            .columns
            .iter()
            .zip(column_bytes)
            .enumerate()
            .map(
                |(position, (column, (bytes, bytes_uncompressed)))| DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by.contains(&column.name),
                    is_sort_key: self.sort_by.contains(&column.name),
                },
            )
            .collect();
        Ok(DatastoreTableMetadata {
            name,
            id,
            columns,
            sort_by: self.sort_by.clone(),
            partition_by: self.partition_by.clone(),
            total_rows,
            bytes: files.iter().map(|file| file.bytes).sum(),
            bytes_uncompressed: files.iter().map(|file| file.bytes_uncompressed).sum(),
            files,
        })
    }

    /// A file's partition as the catalog reports it: `column=value` pairs in
    /// the table's partition order, comma-separated. Empty for an
    /// unpartitioned table, and for a file adopted before the table was
    /// partitioned, which carries no tuple (or none for some of the columns).
    fn format_partition(&self, partition: &Option<PartitionValues>) -> String {
        let Some(values) = partition else {
            return String::new();
        };
        self.partition_by
            .iter()
            .filter_map(|column| {
                let (array, _) = values.get(column)?.get();
                let formatter = ArrayFormatter::try_new(array, &FormatOptions::default()).ok()?;
                Some(format!("{column}={}", formatter.value(0)))
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Add to `totals`, per declared column (found by name in
/// `column_positions`), what the column costs in `row_groups`, those of one
/// committed file: the bytes it occupies in storage, and what they hold
/// decoded; and return what the whole file holds decoded. Chunks are per
/// leaf, so a nested column is charged for every leaf it spans; a file that
/// does not carry a column contributes nothing to it.
fn add_file_bytes(
    totals: &mut [(u64, u64)],
    column_positions: &HashMap<&str, usize>,
    row_groups: &[Arc<RowGroupMetadata>],
) -> u64 {
    let mut file_bytes_uncompressed = 0;
    for row_group in row_groups {
        let mut leaf = 0;
        for field in row_group.schema.fields() {
            let leaves = parquet_engine::leaf_count(field);
            if let Some(&position) = column_positions.get(field.name().as_str()) {
                for chunk in &row_group.columns[leaf..leaf + leaves] {
                    totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                    totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
                }
            }
            leaf += leaves;
        }
        file_bytes_uncompressed += row_group
            .columns
            .iter()
            .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
            .sum::<u64>();
    }
    file_bytes_uncompressed
}

/// Render a file's typed column bounds as a stable JSON object. Each column is
/// one field containing its `min` and `max`; values stay JSON numbers/booleans
/// where their Arrow type has that shape, while strings and temporal values are
/// quoted. A column whose pair cannot be rendered is omitted.
fn format_min_max_stats(stats: Option<&FileStats>) -> String {
    let Some(stats) = stats else {
        return "{}".to_string();
    };
    let mut names: Vec<_> = stats.min_values.keys().collect();
    names.sort_unstable();

    let fields = names
        .into_iter()
        .filter_map(|name| {
            let min = format_bound(stats.min_values.get(name)?)?;
            let max = format_bound(stats.max_values.get(name)?)?;
            let name = serde_json::to_string(name).expect("a Rust string is valid JSON");
            Some(format!(r#"{name}:{{"min":{min},"max":{max}}}"#))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{fields}}}")
}

/// Render the single value held by a file statistic as one JSON value.
fn format_bound(array: &ArrayRef) -> Option<String> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let formatter = ArrayFormatter::try_new(array.as_ref(), &FormatOptions::default()).ok()?;
    let rendered = formatter.value(0).to_string();
    match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => serde_json::from_str::<serde_json::Value>(&rendered)
            .ok()
            .map(|_| rendered),
        _ => serde_json::to_string(&rendered).ok(),
    }
}
