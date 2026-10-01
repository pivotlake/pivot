//! One Iceberg table at one metadata location.
//!
//! Loading reads the schema and the current snapshot's manifest list. The
//! manifests are read later, when a query needs them. A query with a filter
//! prunes in three steps, each reading only what the step before kept:
//! manifests by the partition summaries in the manifest list, files by the
//! partition values and column bounds in their manifest, row groups by the
//! statistics in their footer. The first two steps lay their data out as a
//! [`Bounds`] layer and run the same evaluator over it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array};
use arrow_cast::display::array_value_to_string;
use arrow_schema::{DataType, Field};
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FieldSummary, FormatVersion, Manifest,
    ManifestContentType, ManifestFile, ManifestList, NestedField, PrimitiveLiteral, SchemaRef,
    TableMetadataRef, Transform,
};
use iceberg::table::Table;
use object_storage::{ExternalStoreFactory, load_objects};
use parquet_engine::{
    BoundKey, Bounds, ColumnResolution, ParquetTable, PushedPredicate, Range, RangePredicate,
    RowGroupMetadata, TableColumns, load_file_row_groups, with_nan_free_columns,
};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::types::{Type, physical_arrow_type};

use crate::columns::table_columns;
use crate::partition::{manifest_list_ranges, partition_value_ranges, project_predicates};
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::values::build_array;
use crate::{Error, Result};

/// How many files' footers `describe` reads at a time, which bounds how much
/// footer metadata, and how many open files, it holds at once.
const FILES_PER_DESCRIBE_FOOTER_READ: usize = 1024;

/// A table as of one metadata location.
pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    /// The table's UUID, the same across every metadata version.
    pub uuid: String,
    /// The metadata file this was loaded from: the current one when the
    /// table was indexed. A table has one metadata file per commit, so this
    /// names the version read.
    pub metadata_location: String,
    /// The metadata in that file: the schema, and the partition spec of each
    /// manifest.
    metadata: TableMetadataRef,
    columns: TableColumns,
    /// The columns the default spec partitions by identity, in spec order. A
    /// bucket or day partition is not a column value, so it is not listed.
    partition_by: Vec<String>,
    /// The columns the default sort order sorts on directly, in order.
    sort_by: Vec<String>,
    /// The current snapshot's manifests, in manifest list order. Empty for a
    /// table without a snapshot.
    manifest_list: Vec<ManifestFile>,
    /// Each manifest's partition summaries as a layer to prune: one slot per
    /// manifest, in list order.
    manifest_bounds: Bounds,
    /// The manifests read so far, by path. Each is read at most once per
    /// load, and only when a query needs it: a filtered scan reads the ones
    /// its predicates keep, a whole-table statistic reads them all.
    manifests: Mutex<HashMap<String, Arc<LoadedManifest>>>,
    /// The store the table's files are read through.
    store: TableStore,
    /// The dispatcher the reads run on, kept so a binding can read when it
    /// compiles or materializes.
    dispatcher: DataFlowDispatcher,
}

/// One manifest's live data files and what the manifest recorded about them.
/// No footer is read.
struct LoadedManifest {
    files: Vec<FileManifestEntry>,
    /// Each file's row count.
    record_counts: Int64Array,
    /// What the manifest recorded per file: each column's bounds and null
    /// count, and each partition field's value. One slot per file, in
    /// manifest order.
    bounds: Bounds,
}

impl LoadedManifest {
    fn total_rows(&self) -> i64 {
        self.record_counts.values().iter().sum()
    }
}

/// One data file as its manifest entry describes it. The footer is read only
/// when a scan needs the file.
struct FileManifestEntry {
    /// The file's full location.
    location: String,
    size: u64,
    /// The file's partition as `column=value` pairs in spec order, comma
    /// separated. Empty for an unpartitioned table.
    partition: String,
}

impl LoadedTable {
    /// Load `table` as the catalog last returned it: its schema, and its
    /// current snapshot's manifest list of `manifest_list_size` bytes (learned
    /// when the table was indexed), read through `store`. Manifests and
    /// footers are read later, per query.
    pub(crate) fn load(
        name: &SchemaQualifiedTableName,
        table: &Table,
        store: TableStore,
        manifest_list_size: Option<u64>,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let metadata = table.metadata_ref();
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: table_name.clone(),
            })?
            .to_string();
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable { table: table_name });
        }
        let schema = metadata.current_schema();
        let columns = table_columns(&table_name, schema)?;
        let partition_by = identity_columns(
            schema,
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );

        let manifest_list = match metadata.current_snapshot() {
            Some(snapshot) => {
                let size = manifest_list_size.expect(
                    "a table with a current snapshot is indexed with its manifest list's size",
                );
                let list = load_manifest_list(
                    &table_name,
                    snapshot.manifest_list(),
                    size,
                    metadata.format_version(),
                    &store,
                    dispatcher,
                )?;
                data_manifests(&table_name, list)?
            }
            None => Vec::new(),
        };

        let summaries: Vec<(i32, Option<&[FieldSummary]>)> = manifest_list
            .iter()
            .map(|entry| (entry.partition_spec_id, entry.partitions.as_deref()))
            .collect();
        let mut manifest_bounds = Bounds::new(manifest_list.len());
        for (key, range) in manifest_list_ranges(
            schema,
            |spec_id| {
                metadata
                    .partition_spec_by_id(spec_id)
                    .map(|spec| spec.as_ref())
            },
            &summaries,
        ) {
            manifest_bounds.insert(key, range);
        }

        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            metadata,
            columns,
            partition_by,
            sort_by,
            manifest_list,
            manifest_bounds,
            manifests: Mutex::new(HashMap::new()),
            store,
            dispatcher: dispatcher.clone(),
        })
    }

    /// The table a scan reads: the files `predicates` cannot rule out, with
    /// their row groups read from the footers. A manifest whose partition
    /// summaries rule the predicates out is not read. A file whose partition
    /// values or column bounds rule them out is not opened. Row groups keep
    /// manifest list order, then file order, so a row reference finds the
    /// same row in a rebuilt table. When no file can match, the empty table
    /// still carries the schema.
    pub(crate) fn fetch_pruned_parquet(
        &self,
        predicates: &[PushedPredicate],
    ) -> Result<ParquetTable> {
        let predicates = self.range_predicates(predicates)?;
        let kept_manifests: Vec<&ManifestFile> = self
            .manifest_list
            .iter()
            .zip(self.manifest_bounds.kept(&predicates))
            .filter_map(|(entry, kept)| kept.then_some(entry))
            .collect();

        let manifests = self.load_manifests(&kept_manifests)?;
        let mut files = Vec::new();
        for manifest in &manifests {
            let kept = manifest.bounds.kept(&predicates);
            files.extend(
                kept.into_iter()
                    .enumerate()
                    .filter_map(|(file, kept)| kept.then_some((manifest.as_ref(), file))),
            );
        }
        tracing::debug!(
            table = %self.name,
            manifests_read = manifests.len(),
            manifests = self.manifest_list.len(),
            files_kept = files.len(),
            "pruned by manifests"
        );
        if files.is_empty() {
            return Ok(ParquetTable::empty(self.declared_schema()));
        }
        let row_groups = self
            .read_row_groups(&files)?
            .into_iter()
            .flatten()
            .collect();
        Ok(ParquetTable::new(row_groups))
    }

    /// The manifests `entries` name, in that order. The ones not read yet
    /// are fetched in one dataflow and parsed, so each manifest is read at
    /// most once per load.
    fn load_manifests(&self, entries: &[&ManifestFile]) -> Result<Vec<Arc<LoadedManifest>>> {
        let mut manifests = self.manifests.lock().unwrap();
        let unread: Vec<&ManifestFile> = entries
            .iter()
            .copied()
            .filter(|entry| !manifests.contains_key(&entry.manifest_path))
            .collect();
        let to_fetch = unread
            .iter()
            .map(|entry| {
                self.store
                    .data_file(&entry.manifest_path, entry.manifest_length.max(0) as u64)
            })
            .collect::<Result<Vec<_>>>()?;
        for object in load_objects(&self.dispatcher, &to_fetch)? {
            let path = object.file.path.as_str().to_string();
            let manifest = Manifest::parse_avro(&object.bytes).map_err(|source| {
                Error::MalformedMetadataObject {
                    table: self.name.to_string(),
                    path: path.clone(),
                    source: Box::new(source),
                }
            })?;
            let loaded = self.load_manifest_files(&path, &manifest)?;
            manifests.insert(path, Arc::new(loaded));
        }
        Ok(entries
            .iter()
            .map(|entry| manifests[&entry.manifest_path].clone())
            .collect())
    }

    /// Every manifest of the current snapshot, in manifest list order.
    fn load_all_manifests(&self) -> Result<Vec<Arc<LoadedManifest>>> {
        let entries: Vec<&ManifestFile> = self.manifest_list.iter().collect();
        self.load_manifests(&entries)
    }

    /// The live data files of `manifest`, at `path`, and what it recorded
    /// about each column and each partition field.
    fn load_manifest_files(&self, path: &str, manifest: &Manifest) -> Result<LoadedManifest> {
        let schema = self.metadata.current_schema();
        let spec = manifest.metadata().partition_spec();
        let mut files = Vec::new();
        let mut data_files: Vec<&DataFile> = Vec::new();
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            // A data manifest only ever lists data files; anything else here
            // is a manifest this reader does not understand.
            if data_file.content_type() != DataContentType::Data {
                return Err(Error::DeleteFiles {
                    table: self.name.to_string(),
                    manifest: path.to_string(),
                });
            }
            if data_file.file_format() != DataFileFormat::Parquet {
                return Err(Error::NonParquetFile {
                    table: self.name.to_string(),
                    file: data_file.file_path().to_string(),
                    format: data_file.file_format().to_string(),
                });
            }
            files.push(FileManifestEntry {
                partition: spec
                    .partition_to_path(data_file.partition(), schema.clone())
                    .replace('/', ","),
                location: data_file.file_path().to_string(),
                size: data_file.file_size_in_bytes(),
            });
            data_files.push(data_file);
        }
        let mut bounds = Bounds::new(data_files.len());
        for (position, (column, &field_id)) in
            self.columns().iter().zip(self.field_ids()).enumerate()
        {
            // A bound is only useful with its pair: a file missing either, or
            // holding a truncated one, gets neither.
            let pairs: Vec<Option<(&Datum, &Datum)>> = data_files
                .iter()
                .map(|file| {
                    let lower = file.lower_bounds().get(&field_id)?;
                    let upper = file.upper_bounds().get(&field_id)?;
                    (!is_truncated(lower) && !is_truncated(upper)).then_some((lower, upper))
                })
                .collect();
            // A column that is NULL in every row matches no comparison.
            let all_null = BooleanArray::from_iter(data_files.iter().map(|file| {
                Some(
                    file.null_value_counts()
                        .get(&field_id)
                        .is_some_and(|nulls| *nulls == file.record_count()),
                )
            }));
            // A float column is NaN-free only where the manifest counted zero.
            let floating = matches!(column.col_type, Type::Float32 | Type::Float64);
            let may_hold_nan =
                BooleanArray::from_iter(data_files.iter().map(|file| {
                    Some(floating && file.nan_value_counts().get(&field_id) != Some(&0))
                }));
            bounds.insert(
                BoundKey::Column(position),
                Range {
                    min: build_array(
                        &column.col_type,
                        pairs
                            .iter()
                            .map(|pair| pair.map(|(lower, _)| lower.literal())),
                    ),
                    max: build_array(
                        &column.col_type,
                        pairs
                            .iter()
                            .map(|pair| pair.map(|(_, upper)| upper.literal())),
                    ),
                    all_null,
                    may_hold_nan,
                },
            );
        }
        for (key, range) in partition_value_ranges(schema, spec, &data_files) {
            bounds.insert(key, range);
        }
        Ok(LoadedManifest {
            files,
            record_counts: Int64Array::from_iter_values(
                data_files.iter().map(|file| file.record_count() as i64),
            ),
            bounds,
        })
    }

    /// The predicates every layer is pruned by: each pushed predicate on its
    /// column, plus its projections onto the partition fields built from that
    /// column in every spec the table has used.
    fn range_predicates(&self, predicates: &[PushedPredicate]) -> Result<Vec<RangePredicate>> {
        // A predicate on a path inside a variant has no bound at any layer.
        let plain: Vec<&PushedPredicate> = predicates
            .iter()
            .filter(|predicate| predicate.path.is_empty())
            .collect();
        let mut ranged: Vec<RangePredicate> = plain
            .iter()
            .map(|predicate| RangePredicate {
                key: BoundKey::Column(predicate.column_idx),
                compare: predicate.compare_type,
                constants: vec![predicate.value.clone()],
            })
            .collect();
        let schema = self.metadata.current_schema();
        let sourced: Vec<(&NestedField, &PushedPredicate)> = plain
            .iter()
            .map(|predicate| {
                let source = schema
                    .field_by_id(self.field_ids()[predicate.column_idx])
                    .expect("a declared column is a field of the schema")
                    .as_ref();
                (source, *predicate)
            })
            .collect();
        ranged.extend(project_predicates(
            &self.name.to_string(),
            schema,
            self.metadata
                .partition_specs_iter()
                .map(|spec| spec.as_ref()),
            &sourced,
        )?);
        Ok(ranged)
    }

    /// Each column's Iceberg field id, in declared order. A manifest keys its
    /// statistics by field id.
    fn field_ids(&self) -> &[i32] {
        let ColumnResolution::ByFieldId(field_ids) = self.columns.resolution() else {
            panic!("an Iceberg table's columns resolve by field id");
        };
        field_ids
    }

    /// The row groups of `files` (each a manifest and a file position in it),
    /// read from the footers in one dataflow, one list per file in the given
    /// order. The order matters: a late materialize rebuilds the scan's table
    /// and finds rows by row-group position. A footer has no NaN count, so a
    /// float column the manifest counted zero NaNs in is marked NaN-free on
    /// the file's row groups, and they prune by their bounds alone.
    fn read_row_groups(
        &self,
        files: &[(&LoadedManifest, usize)],
    ) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let to_fetch = files
            .iter()
            .map(|(manifest, file)| {
                let entry = &manifest.files[*file];
                self.store.data_file(&entry.location, entry.size)
            })
            .collect::<Result<Vec<_>>>()?;
        let loaded = load_file_row_groups(&self.dispatcher, &to_fetch, self.columns.clone())?;
        Ok(files
            .iter()
            .zip(loaded)
            .map(|((manifest, file), loaded)| {
                let nan_free: Vec<usize> = self
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|(position, column)| {
                        matches!(column.col_type, Type::Float32 | Type::Float64)
                            && manifest
                                .bounds
                                .range(BoundKey::Column(*position))
                                .is_some_and(|range| !range.may_hold_nan.value(*file))
                    })
                    .map(|(position, _)| position)
                    .collect();
                with_nan_free_columns(loaded.row_groups, &nan_free)
            })
            .collect())
    }

    /// File `file`'s bounds from `manifest`, as the catalog reports them: a
    /// JSON object keyed by column name, each with `min` and `max`. Numbers
    /// and booleans are bare, everything else a string. A column without a
    /// bound is left out.
    fn min_max_stats(&self, manifest: &LoadedManifest, file: usize) -> String {
        let quote = |text: &str| serde_json::to_string(text).expect("a Rust string is valid JSON");
        let render = |bound: &ArrayRef| -> Option<String> {
            let text = array_value_to_string(bound, file).ok()?;
            let bare = bound.data_type().is_numeric() || *bound.data_type() == DataType::Boolean;
            Some(if bare { text } else { quote(&text) })
        };
        let fields: Vec<String> = self
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(position, column)| {
                let bounds = manifest
                    .bounds
                    .range(BoundKey::Column(position))
                    .expect("a manifest bounds every declared column");
                (bounds.min.is_valid(file) && bounds.max.is_valid(file)).then_some((column, bounds))
            })
            .filter_map(|(column, bounds)| {
                Some(format!(
                    "{}:{{\"min\":{},\"max\":{}}}",
                    quote(&column.name),
                    render(&bounds.min)?,
                    render(&bounds.max)?
                ))
            })
            .collect();
        format!("{{{}}}", fields.join(","))
    }

    /// The declared Arrow schema: each column's name, physical type and
    /// nullability. What a scan with no files reports.
    fn declared_schema(&self) -> arrow_schema::SchemaRef {
        let fields: Vec<Field> = self
            .columns()
            .iter()
            .enumerate()
            .map(|(position, column)| {
                Field::new(
                    &column.name,
                    physical_arrow_type(&column.col_type),
                    self.column_may_hold_nulls(position),
                )
            })
            .collect();
        Arc::new(arrow_schema::Schema::new(fields))
    }

    pub(crate) fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    /// Whether `column` can hold NULL: whether the schema declares it
    /// optional. A required column has no NULLs and cannot be added once
    /// files exist, so every file holds it.
    pub(crate) fn column_may_hold_nulls(&self, column: usize) -> bool {
        !self
            .metadata
            .current_schema()
            .field_by_id(self.field_ids()[column])
            .expect("a declared column is a field of the schema")
            .required
    }

    pub(crate) fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid.clone(),
            version: self.metadata_location.clone(),
        }
    }

    /// The row count, summed from the manifests. No footer is read.
    pub(crate) fn total_rows(&self) -> Result<i64> {
        Ok(self
            .load_all_manifests()?
            .iter()
            .map(|manifest| manifest.total_rows())
            .sum())
    }

    /// The row count as the manifest list states it, without reading a
    /// manifest. A manifest whose list entry has no row counts (a v1 writer
    /// may omit them) is read.
    pub(crate) fn estimated_rows(&self) -> Result<i64> {
        let mut rows = 0i64;
        let mut uncounted = Vec::new();
        for entry in &self.manifest_list {
            match (entry.added_rows_count, entry.existing_rows_count) {
                (Some(added), Some(existing)) => rows += (added + existing) as i64,
                _ => uncounted.push(entry),
            }
        }
        for manifest in self.load_manifests(&uncounted)? {
            rows += manifest.total_rows();
        }
        Ok(rows)
    }

    /// The table as the system catalog describes it. Reads every manifest
    /// and every footer, since column byte sizes come from footers. Footers
    /// are read a chunk of files at a time and dropped once their sizes are
    /// summed, so a large table never has every footer in memory at once.
    pub(crate) fn describe(&self) -> Result<DatastoreTableMetadata> {
        let manifests = self.load_all_manifests()?;
        let all_files: Vec<(&LoadedManifest, usize)> = manifests
            .iter()
            .flat_map(|manifest| {
                (0..manifest.files.len()).map(move |file| (manifest.as_ref(), file))
            })
            .collect();
        let mut column_bytes = vec![(0u64, 0u64); self.columns().len()];
        let mut bytes_uncompressed = 0u64;
        let mut files = Vec::with_capacity(all_files.len());
        for chunk in all_files.chunks(FILES_PER_DESCRIBE_FOOTER_READ) {
            let per_file = self.read_row_groups(chunk)?;
            for ((manifest, position), groups) in chunk.iter().zip(&per_file) {
                let file = &manifest.files[*position];
                let mut file_uncompressed = 0u64;
                for group in groups {
                    for (position, column) in group.columns.iter().enumerate() {
                        let uncompressed = column.total_uncompressed_size.max(0) as u64;
                        column_bytes[position].0 += column.total_compressed_size.max(0) as u64;
                        column_bytes[position].1 += uncompressed;
                        file_uncompressed += uncompressed;
                    }
                }
                bytes_uncompressed += file_uncompressed;
                files.push(DatastoreFileMetadata {
                    path: file.location.clone(),
                    bytes: file.size,
                    bytes_uncompressed: file_uncompressed,
                    partition: file.partition.clone(),
                    min_max_stats: self.min_max_stats(manifest, *position),
                });
            }
        }

        let columns = self
            .columns()
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let (bytes, bytes_uncompressed) = column_bytes[position];
                DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by.contains(&column.name),
                    is_sort_key: self.sort_by.contains(&column.name),
                }
            })
            .collect();
        Ok(DatastoreTableMetadata {
            name: self.name.clone(),
            id: self.uuid.clone(),
            columns,
            sort_by: self.sort_by.clone(),
            partition_by: self.partition_by.clone(),
            total_rows: manifests
                .iter()
                .map(|manifest| manifest.total_rows().max(0) as u64)
                .sum(),
            bytes: files.iter().map(|file| file.bytes).sum(),
            bytes_uncompressed,
            files,
        })
    }
}

/// Whether a bound is the sentinel a manifest writes for a value outside the
/// type's range. Not a value to compare with.
fn is_truncated(bound: &Datum) -> bool {
    matches!(
        bound.literal(),
        PrimitiveLiteral::AboveMax | PrimitiveLiteral::BelowMin
    )
}

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source id and transform) use by identity, in field order. A bucket or
/// day field is not a column value and is left out.
fn identity_columns<'a>(
    schema: &SchemaRef,
    fields: impl Iterator<Item = (i32, &'a Transform)>,
) -> Vec<String> {
    fields
        .filter(|(_, transform)| **transform == Transform::Identity)
        .filter_map(|(source_id, _)| schema.name_by_field_id(source_id))
        .map(str::to_string)
        .collect()
}

/// The store `table`'s files are read through: opened with the credentials
/// the catalog vended, or through `store_factory` when it vended none. Opened
/// at index time rather than per query, since it may ask the bucket for its
/// region, and again at every refresh, so vended credentials never outlive
/// their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::StagedTable {
            table: table_name.clone(),
        })?;
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    TableStore::open(&table_name, metadata_location, store_factory, vended)
}

/// The size of `table`'s current manifest list, or `None` without a
/// snapshot. The snapshot names the list without a length, and a ring read
/// needs one, so the store is asked once at index time rather than per query.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store: &TableStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let size = store
        .object_size(path)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
}

/// The manifest list at `path`, `size` bytes, read through the ring and
/// parsed.
fn load_manifest_list(
    table: &str,
    path: &str,
    size: u64,
    format_version: FormatVersion,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> Result<ManifestList> {
    let bytes = load_objects(dispatcher, &[store.data_file(path, size)?])?
        .pop()
        .expect("load_objects returns one object per file")
        .bytes;
    let list = ManifestList::parse_with_version(&bytes, format_version).map_err(|source| {
        Error::MalformedMetadataObject {
            table: table.to_string(),
            path: path.to_string(),
            source: Box::new(source),
        }
    })?;
    Ok(list)
}

/// The data manifests `list` names, in order. A delete manifest that still
/// lists live delete files means rows this reader would return were deleted,
/// so the table is refused. One that only records removed delete files is
/// skipped.
fn data_manifests(table: &str, list: ManifestList) -> Result<Vec<ManifestFile>> {
    let mut manifests = Vec::new();
    for entry in list.consume_entries() {
        if entry.content == ManifestContentType::Deletes {
            if entry.has_added_files() || entry.has_existing_files() {
                return Err(Error::DeleteFiles {
                    table: table.to_string(),
                    manifest: entry.manifest_path,
                });
            }
            continue;
        }
        manifests.push(entry);
    }
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type as IcebergType};

    use super::*;

    #[test]
    fn only_identity_fields_are_columns_in_field_order() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", IcebergType::Primitive(PrimitiveType::Long))
                        .into(),
                    NestedField::required(
                        2,
                        "region",
                        IcebergType::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    NestedField::required(3, "day", IcebergType::Primitive(PrimitiveType::Date))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let fields = [
            (2, Transform::Identity),
            (1, Transform::Bucket(16)),
            (3, Transform::Day),
            (1, Transform::Identity),
        ];

        let columns = identity_columns(
            &schema,
            fields
                .iter()
                .map(|(source_id, transform)| (*source_id, transform)),
        );

        assert_eq!(columns, ["region", "id"]);
    }
}
