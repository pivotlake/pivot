//! The external `read_parquet(path)` table function.
//!
//! Binding resolves the store, expands the path pattern, and loads every
//! footer once. The result is a regular read-only `BoundTable`, so planning and
//! execution use the same captured files and schema as an ordinary catalog
//! table scan.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::TableFilter;
use planner::types::{Type, type_from_physical};

use crate::{
    ParquetTable, PushedPredicate, is_variant_field, materialize, prune_parquet,
    table_input_with_filter_and_eq_predicates,
};
use object_storage::{
    DataFile, ExternalStoreFactory, FileRef, ObjectPath, ObjectStore, StoreScheme, local_path,
};

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("invalid parquet path `{path}`: {message}")]
    InvalidPath { path: String, message: String },
    #[error("wildcards are not supported in a bucket name: `{0}`")]
    WildcardBucket(String),
    #[error("no files found matching `{0}`")]
    NoFiles(String),
    #[error("parquet file `{0}` contains no row groups")]
    NoRowGroups(String),
    #[error("opening external parquet store: {0}")]
    Store(#[from] object_storage::StoreError),
    #[error("loading external parquet metadata: {0}")]
    Metadata(String),
    #[error("parquet file `{file}` has schema {actual:?}, expected {expected:?} from `{first}`")]
    SchemaMismatch {
        first: String,
        expected: Vec<Column>,
        file: String,
        actual: Vec<Column>,
    },
    #[error("parquet column `{column}` has unsupported type {data_type}")]
    UnsupportedColumn {
        column: String,
        data_type: arrow_schema::DataType,
    },
}

/// Bind one external Parquet invocation into the same [`BoundTable`] interface
/// used by catalog tables. The caller supplies query execution and credential
/// policy; this engine owns path expansion, footer loading, and the scan state.
pub fn bind_read_parquet(
    dispatcher: &DataFlowDispatcher,
    store_factory: &dyn ExternalStoreFactory,
    location: &str,
) -> CatalogResult<Box<dyn BoundTable>> {
    let table = bind_external_table(dispatcher, store_factory, location)
        .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
    Ok(Box::new(table))
}

#[derive(Debug, Clone)]
struct ExternalParquetTable {
    location: String,
    columns: Vec<Column>,
    parquet: Arc<ParquetTable>,
    predicates: Vec<PushedPredicate>,
}

impl BoundTable for ExternalParquetTable {
    fn table_reference(&self) -> TableReference {
        TableReference {
            datastore: planner::DEFAULT_DATASTORE_NAME.to_string(),
            schema: planner::DEFAULT_SCHEMA_NAME.to_string(),
            table: self.location.clone(),
        }
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: self.location.clone(),
            version: "0".to_string(),
        }
    }

    fn is_plan_cacheable(&self) -> bool {
        false
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let equality_predicates = crate::equality_predicates(&self.predicates);
        let parquet = Arc::new(prune_parquet(&self.parquet, &self.predicates));
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            crate::row_group_filter_from(dynamic_filters),
            None,
            Arc::new(equality_predicates),
        ))
    }

    fn applies_variant_extracts(&self) -> bool {
        true
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Row-group metadata stores positions in this statically-pruned view,
        // so the fetch must rebuild exactly the view used by compile_scan.
        // Dynamic filters only skip groups within that stable view and do not
        // renumber them.
        Ok(materialize(
            input,
            Arc::new(prune_parquet(&self.parquet, &self.predicates)),
            projection,
        ))
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        self.predicates.extend(PushedPredicate::from_filter(filter));
        // Statistics and dictionaries may skip work, but the SQL filter stays
        // above the scan and remains responsible for query correctness.
        Ok(false)
    }

    fn row_count(&self) -> Option<i64> {
        if !self.predicates.is_empty() {
            return None;
        }
        Some(self.parquet.total_rows())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.parquet.total_rows() as u64)
    }
}

fn bind_external_table(
    dispatcher: &DataFlowDispatcher,
    store_factory: &dyn ExternalStoreFactory,
    location: &str,
) -> Result<ExternalParquetTable, Error> {
    let selection = ParquetLocationPattern::parse(location)?;
    let store = store_factory.open(&selection.store_root)?;
    let files = selection.find_files(store.as_ref(), location)?;
    let (columns, parquet) = load_strict_table(dispatcher, files)?;
    Ok(ExternalParquetTable {
        location: location.to_string(),
        columns,
        parquet: Arc::new(parquet),
        predicates: Vec::new(),
    })
}

struct ParquetLocationPattern {
    store_root: String,
    path_pattern: String,
}

impl ParquetLocationPattern {
    fn parse(location: &str) -> Result<Self, Error> {
        match StoreScheme::of(location)? {
            StoreScheme::S3 | StoreScheme::Gcs => Self::parse_remote(location),
            StoreScheme::Local => Self::parse_local(location),
        }
    }

    /// Split an S3 or GCS location into the longest literal store root and the
    /// relative pattern to expand beneath it. For example,
    /// `s3://bucket/events/*/hello/*.parquet` becomes store root
    /// `s3://bucket/events` and pattern `*/hello/*.parquet`. The store root is
    /// also the location used to select credentials.
    fn parse_remote(location: &str) -> Result<Self, Error> {
        let (_, rest) = location
            .split_once("://")
            .expect("a remote store location has a URI scheme");
        let (bucket, path_in_bucket) = rest.split_once('/').ok_or_else(|| Error::InvalidPath {
            path: location.to_string(),
            message: "expected an object path after the bucket".to_string(),
        })?;
        if bucket.is_empty() {
            return Err(Error::InvalidPath {
                path: location.to_string(),
                message: "bucket must not be empty".to_string(),
            });
        }
        if bucket.contains('*') {
            return Err(Error::WildcardBucket(location.to_string()));
        }
        if path_in_bucket.is_empty() || path_in_bucket.ends_with('/') {
            return Err(Error::InvalidPath {
                path: location.to_string(),
                message: "expected a file name or pattern".to_string(),
            });
        }
        let first_wildcard = location.find('*').unwrap_or(location.len());
        let root_end = location[..first_wildcard]
            .rfind('/')
            .expect("a remote object path follows its bucket");
        Ok(Self {
            store_root: location[..root_end].to_string(),
            path_pattern: location[root_end + 1..].to_string(),
        })
    }

    fn parse_local(location: &str) -> Result<Self, Error> {
        let path = Path::new(local_path(location));
        path.file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| Error::InvalidPath {
                path: location.to_string(),
                message: "expected a file name or pattern".to_string(),
            })?;
        let components: Vec<_> = path.components().collect();
        let pattern_start = components
            .iter()
            .position(|component| component.as_os_str().to_string_lossy().contains('*'))
            .unwrap_or(components.len() - 1);
        let mut store_root = PathBuf::new();
        for component in &components[..pattern_start] {
            store_root.push(component.as_os_str());
        }
        if store_root.as_os_str().is_empty() {
            store_root.push(".");
        }
        let path_pattern = components[pattern_start..]
            .iter()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        Ok(Self {
            store_root: store_root.to_string_lossy().into_owned(),
            path_pattern,
        })
    }

    fn find_files(
        &self,
        store: &dyn ObjectStore,
        original_location: &str,
    ) -> Result<Vec<DataFile>, Error> {
        let mut segments: Vec<_> = self.path_pattern.split('/').collect();
        let filename_pattern = segments
            .pop()
            .expect("a parquet location has a final file pattern");
        let mut directories = vec![ObjectPath::default()];
        for segment in segments {
            if !segment.contains('*') {
                directories = directories
                    .into_iter()
                    .map(|directory| directory.join(segment))
                    .collect();
                continue;
            }
            let mut matched = Vec::new();
            for directory in directories {
                for prefix in store
                    .list_with_name_prefix(&directory, literal_prefix(segment))?
                    .prefixes
                {
                    if wildcard_segment_matches(segment, prefix.as_str()) {
                        matched.push(directory.join(prefix.as_str()));
                    }
                }
            }
            directories = matched;
        }

        let mut files = Vec::new();
        for directory in directories {
            for object in store
                .list_with_name_prefix(&directory, literal_prefix(filename_pattern))?
                .objects
            {
                if !wildcard_segment_matches(filename_pattern, object.file.path.as_str()) {
                    continue;
                }
                let path = directory.join(object.file.path.as_str());
                let source = store.source(&path)?;
                files.push(DataFile {
                    file: FileRef {
                        path,
                        size: object.file.size,
                    },
                    source,
                });
            }
        }
        files.sort_by(|left, right| left.file.path.as_str().cmp(right.file.path.as_str()));
        if files.is_empty() {
            return Err(Error::NoFiles(original_location.to_string()));
        }
        Ok(files)
    }
}

fn load_strict_table(
    dispatcher: &DataFlowDispatcher,
    files: Vec<DataFile>,
) -> Result<(Vec<Column>, ParquetTable), Error> {
    let file_order: Vec<ObjectPath> = files.iter().map(|file| file.file.path.clone()).collect();
    let loaded = crate::load_file_row_groups(dispatcher, &files, Arc::from([]))
        .map_err(|error| Error::Metadata(error.to_string()))?;
    let mut loaded_by_path: HashMap<_, _> = loaded
        .into_iter()
        .map(|file| (file.file.path.clone(), file))
        .collect();

    let first_path = file_order[0].to_string();
    let first = loaded_by_path
        .get(&file_order[0])
        .expect("metadata loading returns every requested file");
    let first_schema = &first
        .row_groups
        .first()
        .ok_or_else(|| Error::NoRowGroups(first_path.clone()))?
        .schema;
    let expected = columns_from_schema(first_schema)?;
    let mut row_groups = Vec::new();
    for path in file_order {
        let loaded = loaded_by_path
            .remove(&path)
            .expect("metadata loading returns every requested file");
        let file_path = path.to_string();
        let file_schema = &loaded
            .row_groups
            .first()
            .ok_or_else(|| Error::NoRowGroups(file_path.clone()))?
            .schema;
        let actual = columns_from_schema(file_schema)?;
        if actual != expected {
            return Err(Error::SchemaMismatch {
                first: first_path,
                expected,
                file: file_path,
                actual,
            });
        }
        row_groups.extend(loaded.row_groups);
    }
    Ok((expected, ParquetTable::new(row_groups)))
}

fn columns_from_schema(schema: &arrow_schema::Schema) -> Result<Vec<Column>, Error> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let col_type = if is_variant_field(field) {
                Some(Type::Variant)
            } else {
                type_from_physical(field.data_type())
            }
            .ok_or_else(|| Error::UnsupportedColumn {
                column: field.name().clone(),
                data_type: field.data_type().clone(),
            })?;
            Ok(Column {
                name: field.name().clone(),
                col_type,
            })
        })
        .collect()
}

/// The literal head of a segment pattern, up to its first `*`. Listing with
/// this as the name prefix lets the store narrow the listing server-side; a
/// literal segment (no `*`) narrows it to exactly the named child.
fn literal_prefix(pattern: &str) -> &str {
    &pattern[..pattern.find('*').unwrap_or(pattern.len())]
}

/// Match one path segment. `*` consumes any number of bytes; every other
/// byte is literal. UTF-8 remains safe because successful literal comparisons
/// advance both strings over the same encoded bytes.
fn wildcard_segment_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let (mut pattern_index, mut value_index) = (0, 0);
    let (mut last_star, mut retry_value) = (None, 0);
    while value_index < value.len() {
        if pattern.get(pattern_index) == Some(&value[value_index]) {
            pattern_index += 1;
            value_index += 1;
        } else if pattern.get(pattern_index) == Some(&b'*') {
            last_star = Some(pattern_index);
            pattern_index += 1;
            retry_value = value_index;
        } else if let Some(star) = last_star {
            retry_value += 1;
            value_index = retry_value;
            pattern_index = star + 1;
        } else {
            return false;
        }
    }
    while pattern.get(pattern_index) == Some(&b'*') {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_remote_parquet_location_patterns() {
        for (location, store_root, path_pattern) in [
            (
                "s3://bucket/events/part-*.parquet",
                "s3://bucket/events",
                "part-*.parquet",
            ),
            (
                "s3a://bucket/events/*/hello/*.parquet",
                "s3a://bucket/events",
                "*/hello/*.parquet",
            ),
            (
                "gs://bucket/events/*/hello/*.parquet",
                "gs://bucket/events",
                "*/hello/*.parquet",
            ),
        ] {
            let pattern = ParquetLocationPattern::parse(location).unwrap();
            assert_eq!(pattern.store_root, store_root);
            assert_eq!(pattern.path_pattern, path_pattern);
        }
    }

    #[test]
    fn literal_prefix_stops_at_the_first_star() {
        assert_eq!(literal_prefix("part-*.parquet"), "part-");
        assert_eq!(literal_prefix("*.parquet"), "");
        assert_eq!(literal_prefix("data.parquet"), "data.parquet");
    }

    #[test]
    fn star_matches_a_path_segment() {
        assert!(wildcard_segment_matches("20*", "2026"));
        assert!(!wildcard_segment_matches("20*", "hello"));
    }
}
