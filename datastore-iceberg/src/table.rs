//! One Iceberg table at one metadata location: its schema and its current
//! snapshot's manifest list, with the manifests read as queries need them. A
//! query with a filter prunes in three stages, each on what the stage before
//! it read: manifests by the partition summaries the manifest list carries,
//! then the files of the manifests it read by their partition values and
//! column bounds, then the row groups of the files it kept by their footer
//! statistics. A selective scan never reads a manifest, or fetches a footer,
//! it will not take rows from.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, Int64Array, Scalar};
use arrow_cast::display::array_value_to_string;
use arrow_ord::sort::{SortOptions, sort_to_indices};
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, Manifest,
    ManifestContentType, ManifestFile, ManifestList, PartitionSpec, PrimitiveLiteral, SchemaRef,
    TableMetadataRef, Transform,
};
use iceberg::table::Table;
use object_storage::{ExternalStoreFactory, load_objects};
use parquet_engine::{
    ColumnResolution, ParquetTable, PushedPredicate, RowGroupMetadata, TableColumns,
    bounds_exclude, load_file_row_groups,
};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::types::physical_arrow_type;

use crate::columns::table_columns;
use crate::partition::{PartitionPredicate, partition_value_type, project_predicates};
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::values::build_array;
use crate::{Error, Result};

/// How many files' footers `describe` reads in one dataflow: what bounds the
/// row-group metadata held, and the files open, at any moment while a table
/// of many files is described.
const FILES_PER_DESCRIBE_FOOTER_READ: usize = 1024;

/// A table as of one metadata location.
pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    /// The table's UUID: its identity across every metadata version.
    pub uuid: String,
    /// The metadata file this load was built from: the one the catalog
    /// pointed at as current when the table was indexed. A table keeps a
    /// metadata file per commit, so this location names the version read.
    pub metadata_location: String,
    /// The table's metadata as of that file: its schema, and the partition
    /// spec each manifest was written under.
    metadata: TableMetadataRef,
    columns: TableColumns,
    /// The identity-partitioned columns of the default partition spec, in
    /// spec order. A transformed partition (a bucket, a day) is not a column
    /// value, so it is not listed.
    partition_by: Vec<String>,
    /// The columns of the default sort order sorted on directly, in order.
    sort_by: Vec<String>,
    /// The current snapshot's manifests, as its manifest list names them and
    /// in its order. Empty for a table without a snapshot.
    manifest_list: Vec<ManifestFile>,
    /// The manifests read so far, by path. A manifest is read at most once
    /// per load, and only when a query needs it: a filtered scan reads the
    /// ones its predicates cannot rule out, a whole-table statistic reads
    /// them all.
    manifests: Mutex<HashMap<String, Arc<LoadedManifest>>>,
    /// The store the table's files are read through, kept so a query can read
    /// the manifests and footers it needs when it compiles.
    store: TableStore,
    /// The dispatcher those reads run on. Held so the read can happen when a
    /// binding compiles or materializes, past the point one is passed in.
    dispatcher: DataFlowDispatcher,
}

/// One manifest's live data files, each as its entry describes it, with what
/// the manifest recorded of them. No footer is read.
struct LoadedManifest {
    files: Vec<FileManifestEntry>,
    file_stats: FileStats,
}

/// One data file of the current snapshot, as its manifest entry describes it.
/// The footer is not read until a scan needs this file.
struct FileManifestEntry {
    /// The file's full location, as the manifest names it.
    location: String,
    size: u64,
    /// The partition the file belongs to, as `column=value` pairs in spec
    /// order, comma-separated. Empty for an unpartitioned table.
    partition: String,
}

/// What a manifest recorded of its files, one column at a time: every array
/// has one slot per file, in the manifest's file order. A predicate then
/// prunes every file in one comparison, and a table of many files holds a few
/// words per column per file rather than a scalar.
struct FileStats {
    /// Each file's row count.
    record_counts: Int64Array,
    /// Per declared column, in declared order.
    columns: Vec<ColumnFileStats>,
    /// Each file's value of each field of the manifest's partition spec, in
    /// spec order, typed as the field's transform yields. All null for a field
    /// whose values Pivot does not compare.
    partition_values: Vec<ArrayRef>,
}

impl FileStats {
    /// Which files `predicates` cannot rule out, by their column bounds, nor
    /// `partition_predicates` by their partition values: `true` at each file
    /// to keep. Each predicate is one comparison over every file. A
    /// variant-path predicate has no file-level bound, and a file the manifest
    /// gave no bound for cannot be ruled out, so both keep the file.
    fn files_kept(
        &self,
        predicates: &[PushedPredicate],
        partition_predicates: &[PartitionPredicate],
    ) -> Vec<bool> {
        let mut pruned = vec![false; self.record_counts.len()];
        for predicate in predicates
            .iter()
            .filter(|predicate| predicate.path.is_empty())
        {
            let Some(column) = self.columns.get(predicate.column_idx) else {
                continue;
            };
            let excluded = bounds_exclude(
                &column.min,
                &column.max,
                predicate.compare_type,
                &predicate.value,
            )
            .ok()
            .flatten();
            for (file, is_pruned) in pruned.iter_mut().enumerate() {
                // Every ordinary comparison against NULL is unknown, so a file
                // whose column is entirely NULL satisfies nothing.
                let all_null = column.null_counts.is_valid(file)
                    && column.null_counts.value(file) == self.record_counts.value(file);
                let proven = excluded
                    .as_ref()
                    .is_some_and(|mask| mask.is_valid(file) && mask.value(file));
                *is_pruned |= all_null || proven;
            }
        }
        for predicate in partition_predicates {
            let excluded = predicate.files_excluded(&self.partition_values[predicate.field]);
            for (file, is_pruned) in pruned.iter_mut().enumerate() {
                *is_pruned |= excluded.value(file);
            }
        }
        pruned.into_iter().map(|pruned| !pruned).collect()
    }

    fn total_rows(&self) -> i64 {
        self.record_counts.values().iter().sum()
    }
}

/// One declared column's manifest statistics across the files.
struct ColumnFileStats {
    /// The column's lower bound in each file, in the column's physical Arrow
    /// type so a predicate's constant compares against it directly. Null
    /// where the manifest recorded no bound, a truncated one, or one of a type
    /// Pivot cannot compare; `max` is null in the same slots.
    min: ArrayRef,
    /// The column's upper bound in each file, on the same terms.
    max: ArrayRef,
    /// The column's NULL count in each file; null where the manifest recorded
    /// none.
    null_counts: Int64Array,
}

impl LoadedTable {
    /// Load `table` as the catalog last returned it: its schema and the
    /// current snapshot's manifest list (of `manifest_list_size` bytes,
    /// learned when the table was indexed), read through `store`. Manifests
    /// and footers are read later, per query, as its predicates leave them.
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

        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            metadata,
            columns,
            partition_by,
            sort_by,
            manifest_list,
            manifests: Mutex::new(HashMap::new()),
            store,
            dispatcher: dispatcher.clone(),
        })
    }

    /// The table a scan reads: the files `predicates` cannot prune, with
    /// their row groups read from the footers. The manifests whose partition
    /// summaries rule the predicates out are not read; of the rest, a file
    /// is kept unless its partition values or column bounds rule them out;
    /// only the kept files' footers are fetched. Row groups keep manifest
    /// list order, then file order, so a row reference addresses a stable
    /// position. When no file can match, an empty table still carries the
    /// schema, so the scan shapes an empty result with the right columns.
    pub(crate) fn fetch_pruned_parquet(
        &self,
        predicates: &[PushedPredicate],
    ) -> Result<ParquetTable> {
        // A projection depends on the spec alone, and manifests share specs.
        let mut projections: HashMap<i32, Vec<PartitionPredicate>> = HashMap::new();
        let mut kept_manifests = Vec::new();
        for entry in &self.manifest_list {
            let spec = self.partition_spec_of(entry)?;
            let partition_predicates = match projections.entry(spec.spec_id()) {
                Entry::Occupied(projected) => projected.into_mut(),
                Entry::Vacant(slot) => slot.insert(project_predicates(
                    &self.name.to_string(),
                    self.metadata.current_schema(),
                    spec,
                    self.field_ids(),
                    predicates,
                )?),
            };
            let may_match = match &entry.partitions {
                Some(summaries) => partition_predicates.iter().all(|predicate| {
                    summaries
                        .get(predicate.field)
                        .is_none_or(|summary| predicate.manifest_may_match(summary))
                }),
                None => true,
            };
            if may_match {
                kept_manifests.push(entry);
            }
        }

        let manifests = self.load_manifests(&kept_manifests)?;
        let mut files = Vec::new();
        for (entry, manifest) in kept_manifests.iter().zip(&manifests) {
            let kept = manifest
                .file_stats
                .files_kept(predicates, &projections[&entry.partition_spec_id]);
            files.extend(
                manifest
                    .files
                    .iter()
                    .zip(kept)
                    .filter_map(|(file, kept)| kept.then_some(file)),
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

    /// The manifests `entries` name, in that order: those not yet read are
    /// read through the ring in one dataflow, parsed and kept, so a manifest
    /// is read at most once per load.
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

    /// The live data files of `manifest`, at `path`, with what it recorded of
    /// every declared column and of every field of its partition spec.
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
        let columns = self
            .columns()
            .iter()
            .zip(self.field_ids())
            .map(|(column, &field_id)| {
                // A bound prunes only with its pair, so a file missing either,
                // or holding a truncated one, gets no bound on either side.
                let bounds: Vec<Option<(&Datum, &Datum)>> = data_files
                    .iter()
                    .map(|file| {
                        let lower = file.lower_bounds().get(&field_id)?;
                        let upper = file.upper_bounds().get(&field_id)?;
                        (!is_truncated(lower) && !is_truncated(upper)).then_some((lower, upper))
                    })
                    .collect();
                ColumnFileStats {
                    min: build_array(
                        &column.col_type,
                        bounds
                            .iter()
                            .map(|pair| pair.map(|(lower, _)| lower.literal())),
                    ),
                    max: build_array(
                        &column.col_type,
                        bounds
                            .iter()
                            .map(|pair| pair.map(|(_, upper)| upper.literal())),
                    ),
                    null_counts: Int64Array::from_iter(data_files.iter().map(|file| {
                        file.null_value_counts()
                            .get(&field_id)
                            .map(|count| *count as i64)
                    })),
                }
            })
            .collect();
        let partition_values =
            spec.fields()
                .iter()
                .enumerate()
                .map(|(position, field)| {
                    let values = data_files.iter().map(|file| {
                        match file.partition().fields().get(position) {
                            Some(Some(Literal::Primitive(value))) => Some(value),
                            _ => None,
                        }
                    });
                    match partition_value_type(schema, field) {
                        Some((_, value_type)) => build_array(&value_type, values),
                        None => arrow_array::new_null_array(&DataType::Null, data_files.len()),
                    }
                })
                .collect();
        Ok(LoadedManifest {
            files,
            file_stats: FileStats {
                record_counts: Int64Array::from_iter_values(
                    data_files.iter().map(|file| file.record_count() as i64),
                ),
                columns,
                partition_values,
            },
        })
    }

    /// The partition spec `entry` was written under.
    fn partition_spec_of(&self, entry: &ManifestFile) -> Result<&PartitionSpec> {
        self.metadata
            .partition_spec_by_id(entry.partition_spec_id)
            .map(|spec| spec.as_ref())
            .ok_or_else(|| Error::UnknownPartitionSpec {
                table: self.name.to_string(),
                manifest: entry.manifest_path.clone(),
                spec_id: entry.partition_spec_id,
            })
    }

    /// The Iceberg field id of each declared column, in declared order. A
    /// manifest keys its bounds and counts by field id, which an Iceberg
    /// table's columns always resolve by.
    fn field_ids(&self) -> &[i32] {
        let ColumnResolution::ByFieldId(field_ids) = self.columns.resolution() else {
            panic!("an Iceberg table's columns resolve by field id");
        };
        field_ids
    }

    /// The row groups of `files`, read from their footers in one dataflow and
    /// returned in the given order, one list per file. The order matters: a
    /// late materialize rebuilds the scan's table and finds rows by row-group
    /// position, so both builds must list the row groups identically.
    fn read_row_groups(
        &self,
        files: &[&FileManifestEntry],
    ) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let to_fetch = files
            .iter()
            .map(|file| self.store.data_file(&file.location, file.size))
            .collect::<Result<Vec<_>>>()?;
        Ok(
            load_file_row_groups(&self.dispatcher, &to_fetch, self.columns.clone())?
                .into_iter()
                .map(|loaded| loaded.row_groups)
                .collect(),
        )
    }

    /// The bounds a manifest recorded of file `file` of `file_stats`, as the
    /// catalog reports them: a JSON object keyed by column name, each holding
    /// the column's `min` and `max`. Numbers and booleans are JSON scalars,
    /// everything else a string. A column the file has no bound for is left
    /// out.
    fn min_max_stats(&self, file_stats: &FileStats, file: usize) -> String {
        let quote = |text: &str| serde_json::to_string(text).expect("a Rust string is valid JSON");
        let render = |bound: &ArrayRef| -> Option<String> {
            let text = array_value_to_string(bound, file).ok()?;
            let bare = bound.data_type().is_numeric() || *bound.data_type() == DataType::Boolean;
            Some(if bare { text } else { quote(&text) })
        };
        let fields: Vec<String> = self
            .columns()
            .iter()
            .zip(&file_stats.columns)
            .filter(|(_, bounds)| bounds.min.is_valid(file) && bounds.max.is_valid(file))
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

    /// The table's declared Arrow schema: each column's name and physical type
    /// with its nullability. The shape a fully-pruned scan reports, matching
    /// the schema a file's row groups reconcile to.
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

    /// Whether `column` can hold a NULL: whether the schema declares it
    /// optional. A required column is written without NULLs, and cannot be
    /// added to a table with files that predate it, so its files all hold it.
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

    /// The table's row count, summed from its manifests, without reading a
    /// footer.
    pub(crate) fn total_rows(&self) -> Result<i64> {
        Ok(self
            .load_all_manifests()?
            .iter()
            .map(|manifest| manifest.file_stats.total_rows())
            .sum())
    }

    /// The table's row count as the manifest list states it, without
    /// reading a manifest. A manifest whose entry does not count its rows (a
    /// v1 writer need not) is read for them.
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
            rows += manifest.file_stats.total_rows();
        }
        Ok(rows)
    }

    /// A column's value range across the whole table, from the manifest bounds
    /// alone. `None` when any file recorded no bound for it, since the range
    /// would then be unknown.
    pub(crate) fn column_min_max(
        &self,
        column: usize,
    ) -> Result<Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)>> {
        let manifests = self.load_all_manifests()?;
        let bound_of = |side: fn(&ColumnFileStats) -> &ArrayRef| {
            let per_manifest: Vec<&dyn Array> = manifests
                .iter()
                .map(|manifest| side(&manifest.file_stats.columns[column]).as_ref())
                .collect();
            concat(&per_manifest).expect("one column's bounds share a type")
        };
        let (min, max) = (bound_of(|stats| &stats.min), bound_of(|stats| &stats.max));
        if min.is_empty() || min.null_count() > 0 || max.null_count() > 0 {
            return Ok(None);
        }
        let least = sort_to_indices(min.as_ref(), None, Some(1)).expect("a comparable type");
        let greatest = sort_to_indices(
            max.as_ref(),
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            Some(1),
        )
        .expect("a comparable type");
        Ok(Some((
            Scalar::new(min.slice(least.value(0) as usize, 1)),
            Scalar::new(max.slice(greatest.value(0) as usize, 1)),
        )))
    }

    /// This table as the cross-datastore catalog describes it. Reads every
    /// manifest and every file's footer, since per-column byte sizes come
    /// from the footers; this is the system-catalog path, not a query scan.
    /// The footers are read a chunk of files at a time, each chunk's sizes
    /// added to running totals and its row-group metadata dropped before the
    /// next is read, so a table of many files never holds every footer's
    /// metadata, or every file open, at once.
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
            let entries: Vec<&FileManifestEntry> = chunk
                .iter()
                .map(|(manifest, file)| &manifest.files[*file])
                .collect();
            let per_file = self.read_row_groups(&entries)?;
            for (((manifest, position), file), groups) in chunk.iter().zip(entries).zip(&per_file) {
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
                    min_max_stats: self.min_max_stats(&manifest.file_stats, *position),
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
                .map(|manifest| manifest.file_stats.total_rows().max(0) as u64)
                .sum(),
            bytes: files.iter().map(|file| file.bytes).sum(),
            bytes_uncompressed,
            files,
        })
    }
}

/// Whether a bound is the sentinel a manifest writes for a value it could not
/// represent, above or below the type's range: not a value to compare with.
fn is_truncated(bound: &Datum) -> bool {
    matches!(
        bound.literal(),
        PrimitiveLiteral::AboveMax | PrimitiveLiteral::BelowMin
    )
}

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source field id and transform) take by identity, in field order. A
/// transformed field (a bucket, a day) is not a column value, so it is left out.
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
/// the catalog vended for the table, or through `store_factory` when it
/// vended none. Opened when the table is indexed rather than when a query loads
/// it, since opening a store may ask the bucket for its region; and opened
/// again at every refresh, so vended credentials never outlive their session.
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

/// The size of `table`'s current manifest list, or `None` when the table has
/// no current snapshot. The snapshot names its manifest list without a
/// length, and a ring read needs one, so the store is asked once, when the
/// table is indexed, rather than on every query over it.
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

/// The snapshot's manifest list at `path`, of `size` bytes, read through the
/// ring and parsed.
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

/// The data manifests `list` names, in its order, once it is known to carry
/// nothing this reader would misread: a delete manifest that still lists live
/// delete files means rows this reader would return were deleted. One that
/// only records the removal of its delete files carries nothing to apply.
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
    use arrow_array::new_null_array;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type as IcebergType};
    use planner::expression::CompareType;
    use planner::types::Type;

    use super::*;
    use crate::partition::tests::pushed_predicate;

    /// The one pushed predicate `column <compare> constant` yields.
    fn predicate(column: usize, compare: CompareType, constant: i64) -> PushedPredicate {
        pushed_predicate(
            column,
            Type::Int64,
            compare,
            Arc::new(Int64Array::from(vec![constant])),
        )
    }

    /// The statistics of files of `record_counts` rows whose column 0 has
    /// the given `[min, max]` bounds and NULL counts, and whose column 1 has
    /// no bound in any file.
    fn stats(
        bounds: &[Option<(i64, i64)>],
        null_counts: &[Option<i64>],
        record_counts: &[i64],
    ) -> FileStats {
        let files = record_counts.len();
        FileStats {
            record_counts: Int64Array::from(record_counts.to_vec()),
            columns: vec![
                ColumnFileStats {
                    min: Arc::new(Int64Array::from_iter(
                        bounds.iter().map(|bound| bound.map(|(min, _)| min)),
                    )),
                    max: Arc::new(Int64Array::from_iter(
                        bounds.iter().map(|bound| bound.map(|(_, max)| max)),
                    )),
                    null_counts: Int64Array::from(null_counts.to_vec()),
                },
                ColumnFileStats {
                    min: new_null_array(&DataType::Int64, files),
                    max: new_null_array(&DataType::Int64, files),
                    null_counts: Int64Array::from(vec![None; files]),
                },
            ],
            partition_values: Vec::new(),
        }
    }

    #[test]
    fn a_file_is_kept_unless_its_bounds_prove_no_row_matches() {
        let stats = stats(
            &[Some((3, 6)), Some((10, 20)), None],
            &[Some(0), Some(0), None],
            &[4, 4, 4],
        );

        assert_eq!(
            stats.files_kept(&[predicate(0, CompareType::Greater, 6)], &[]),
            [false, true, true]
        );
        assert_eq!(
            stats.files_kept(&[predicate(0, CompareType::Equal, 2)], &[]),
            [false, false, true]
        );
        assert_eq!(
            stats.files_kept(&[predicate(0, CompareType::Equal, 4)], &[]),
            [true, false, true]
        );
        // A column the manifest gave no bound for cannot prune.
        assert_eq!(
            stats.files_kept(&[predicate(1, CompareType::Equal, 0)], &[]),
            [true, true, true]
        );
    }

    #[test]
    fn a_file_whose_column_is_entirely_null_satisfies_no_comparison() {
        let stats = stats(&[Some((3, 6)), Some((3, 6))], &[Some(4), None], &[4, 4]);

        assert_eq!(
            stats.files_kept(&[predicate(0, CompareType::Equal, 4)], &[]),
            [false, true]
        );
    }

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
