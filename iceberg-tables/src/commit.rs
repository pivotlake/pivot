//! Committing to a table: a new snapshot's manifests and manifest list, the
//! metadata document holding it, and the one step that makes it current,
//! which the table's [`TableCatalog`] performs.
//!
//! A commit from one version writes its Avro objects under new names, then
//! asks the catalog to publish the new metadata as the successor of the
//! version it was built on. Exactly one of several writers from the same
//! base succeeds; the rest learn they lost, withdraw their objects, reload,
//! and try again on top of the winner.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use iceberg::spec::{
    FormatVersion, MAIN_BRANCH, ManifestFile, ManifestStatus, Operation, Snapshot,
    SnapshotSummaryCollector, Summary, TableMetadata, TableMetadataBuilder,
};
use object_storage::{ObjectPath, StoreError};
use planner::catalog::Column;
use uuid::Uuid;

use crate::catalog::{CommitChanges, TableCatalog};
use crate::data_files::{
    DataFileEntry, EntryContext, ManifestOrigin, normalized_entry, to_data_file,
};
use crate::layout::{TableLocation, manifest_file_name, manifest_list_file_name};
use crate::load::identity_columns;
use crate::manifest_writer::{
    EntryWrite, LoadedManifest, MIN_MANIFESTS_TO_MERGE, ManifestCache, WrittenManifest,
    plan_manifests, write_manifest, write_manifest_list,
};
use crate::retention::{
    DEFAULT_PREVIOUS_VERSIONS, DEFAULT_SNAPSHOT_RETENTION, PREVIOUS_VERSIONS_PROPERTY,
    SNAPSHOT_RETENTION_PROPERTY, expired_snapshot_ids,
};
use crate::schema::{
    LOGICAL_TYPES_PROPERTY, NAME_MAPPING_PROPERTY, build_partition_spec, build_schema,
    build_sort_order, logical_types_property, name_mapping_property,
};
use crate::state::{CurrentMetadata, TableState};
use crate::{Error, Result};

/// How many times a store write that failed with an IO error is retried
/// before the commit gives up, and how many times a writer that lost the
/// race rebuilds its commit on the winner before giving up.
pub const COMMIT_RETRIES: usize = 10;
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// What `CREATE TABLE` asks a new table to be.
pub struct TableCreation<'a> {
    pub columns: &'a [Column],
    pub partition_by: &'a [String],
    pub sort_by: &'a [String],
    /// Files the table adopts as its initial data, each with its statistics
    /// and, for a partitioned table, its constant partition tuple.
    pub adopted: Vec<DataFileEntry>,
}

/// The metadata document of a new table at `location` with the shape
/// `creation` asks for, before any version of it is published: format
/// version 3, every field optional, the table's retention properties set,
/// and the Pivot types Iceberg has no spelling for recorded.
pub fn new_table_metadata(
    location: &TableLocation,
    table_uuid: Uuid,
    creation: &TableCreation<'_>,
) -> Result<TableMetadata> {
    let schema = build_schema(creation.columns)?;
    let partition_spec = build_partition_spec(&schema, creation.partition_by)?;
    let sort_order = build_sort_order(&schema, creation.sort_by)?;
    let built = TableMetadataBuilder::new(
        schema,
        partition_spec,
        sort_order,
        location.table_uri(),
        FormatVersion::V3,
        HashMap::new(),
    )
    .map_err(Error::spec)?
    .assign_uuid(table_uuid)
    .build()
    .map_err(Error::spec)?
    .metadata;
    // The properties key on field ids, so they are taken from the schema as
    // built, after the builder's own id assignment.
    let mut properties = HashMap::from([
        (
            SNAPSHOT_RETENTION_PROPERTY.to_string(),
            DEFAULT_SNAPSHOT_RETENTION.as_millis().to_string(),
        ),
        (
            PREVIOUS_VERSIONS_PROPERTY.to_string(),
            DEFAULT_PREVIOUS_VERSIONS.to_string(),
        ),
    ]);
    if let Some(logical_types) = logical_types_property(built.current_schema(), creation.columns) {
        properties.insert(LOGICAL_TYPES_PROPERTY.to_string(), logical_types);
    }
    if !creation.adopted.is_empty() {
        properties.insert(
            NAME_MAPPING_PROPERTY.to_string(),
            name_mapping_property(built.current_schema()),
        );
    }
    Ok(built
        .into_builder(None)
        .set_properties(properties)
        .map_err(Error::spec)?
        .build()
        .map_err(Error::spec)?
        .metadata)
}

/// Create the table at `location` through `catalog`: publish its first
/// version, holding the adopted files as the first snapshot when there are
/// any. The version is created, never replaced: a location that already has
/// one is not this table's to take over.
pub fn initialize_table(
    catalog: &dyn TableCatalog,
    location: &TableLocation,
    table_name: &str,
    table_uuid: Uuid,
    creation: TableCreation<'_>,
    now_ms: i64,
) -> Result<TableState> {
    let metadata = new_table_metadata(location, table_uuid, &creation)?;
    // The document has no location until the catalog publishes it, and no
    // base: its first version starts no metadata log.
    let unpublished = TableState {
        location: location.clone(),
        current: Arc::new(CurrentMetadata {
            metadata_location: String::new(),
            metadata: Arc::new(metadata),
        }),
        columns: creation.columns.to_vec(),
        partition_by: creation.partition_by.to_vec(),
        sort_by: creation.sort_by.to_vec(),
        manifest_list: Vec::new(),
        manifest_list_size: None,
        manifests: ManifestCache::new(),
        entries: Vec::new(),
    };
    let prepared = if creation.adopted.is_empty() {
        PreparedCommit::metadata_only(
            &unpublished,
            Arc::clone(&unpublished.current.metadata),
            Vec::new(),
        )
    } else {
        prepare_commit(
            &unpublished,
            catalog,
            None,
            &[],
            creation.adopted,
            true,
            now_ms,
        )?
    };
    match land(catalog, &unpublished, prepared, None)? {
        Some(state) => Ok(state),
        None => Err(Error::TableAlreadyInitialized {
            table: table_name.to_string(),
            location: location.table_uri(),
        }),
    }
}

/// Commit file changes on top of `state`: remove `removed` (each a live file
/// of the state) and add `added`, as one snapshot at the next version.
/// `data_change` is true for a logical change (INSERT) and false for a
/// rearrangement (compaction). `Ok(None)` means another writer landed the
/// next version first; the caller reloads and retries.
pub fn commit_file_changes(
    state: &TableState,
    catalog: &dyn TableCatalog,
    removed: &[DataFileEntry],
    added: Vec<DataFileEntry>,
    data_change: bool,
    now_ms: i64,
) -> Result<Option<TableState>> {
    let prepared = prepare_commit(
        state,
        catalog,
        Some(&state.current),
        removed,
        added,
        data_change,
        now_ms,
    )?;
    land(catalog, state, prepared, Some(&state.current))
}

/// Commit the expiry of every snapshot that has outlived the retention
/// window as of `now_ms`, with no change to the current snapshot. `Ok(None)`
/// when nothing has expired, or when another writer landed the next version
/// first (that writer expired them itself).
pub fn expire_snapshots(
    state: &TableState,
    catalog: &dyn TableCatalog,
    now_ms: i64,
) -> Result<Option<TableState>> {
    let expired = expired_snapshot_ids(&state.current.metadata, now_ms)?;
    if expired.is_empty() {
        return Ok(None);
    }
    let metadata = state
        .current
        .metadata
        .as_ref()
        .clone()
        .into_builder(Some(state.current.metadata_location.clone()))
        .remove_snapshots(&expired)
        .build()
        .map_err(Error::spec)?
        .metadata;
    let prepared = PreparedCommit::metadata_only(state, Arc::new(metadata), expired);
    land(catalog, state, prepared, Some(&state.current))
}

/// A commit built in memory and not yet written: the metadata document to
/// publish, what it changes, the Avro objects it names, and the state the
/// table is in once it is published.
struct PreparedCommit {
    metadata: Arc<TableMetadata>,
    changes: CommitChanges,
    /// The manifests and manifest list, by key, to store before publishing.
    objects: Vec<(ObjectPath, Vec<u8>)>,
    manifest_list: Vec<ManifestFile>,
    manifest_list_size: Option<u64>,
    manifests: ManifestCache,
    entries: Vec<DataFileEntry>,
}

impl PreparedCommit {
    /// A commit that changes the metadata document alone and keeps the
    /// snapshot, expiring `expired`.
    fn metadata_only(state: &TableState, metadata: Arc<TableMetadata>, expired: Vec<i64>) -> Self {
        Self {
            metadata,
            changes: CommitChanges {
                snapshot: None,
                expired,
            },
            objects: Vec::new(),
            manifest_list: state.manifest_list.clone(),
            manifest_list_size: state.manifest_list_size,
            manifests: state.manifests.clone(),
            entries: state.entries.clone(),
        }
    }
}

/// Build the snapshot that removes `removed` and adds `added` on top of
/// `state`: its manifests, its manifest list, and the metadata holding it.
/// `base` is the published document the commit succeeds, `None` for a
/// table's first version. Every snapshot that has expired by `now_ms` is
/// dropped when the catalog leaves expiry to its writers.
fn prepare_commit(
    state: &TableState,
    catalog: &dyn TableCatalog,
    base: Option<&CurrentMetadata>,
    removed: &[DataFileEntry],
    added: Vec<DataFileEntry>,
    data_change: bool,
    now_ms: i64,
) -> Result<PreparedCommit> {
    let metadata = &state.current.metadata;
    let format_version = metadata.format_version();
    let schema = metadata.current_schema();
    let partition_spec = metadata.default_partition_spec();
    let ctx = EntryContext {
        schema,
        partition_spec,
        columns: &state.columns,
        location: &state.location,
        format_version,
    };
    // The added entries are kept as their manifest will record them, so the
    // state holds nothing that only the writer's memory owns.
    let added = added
        .iter()
        .map(|entry| normalized_entry(entry, &ctx))
        .collect::<Result<Vec<_>>>()?;
    let plan = plan_manifests(
        &state.ordered_manifests(),
        removed,
        added,
        MIN_MANIFESTS_TO_MERGE,
    )?;

    let snapshot_id = fresh_snapshot_id(metadata);
    let sequence_number = metadata.next_sequence_number();
    let timestamp_ms = commit_timestamp_ms(metadata, now_ms);
    let commit = Uuid::new_v4();
    let mut summary = SnapshotSummaryCollector::default();
    for entry in removed {
        summary.remove_file(
            &to_data_file(entry, &ctx)?,
            schema.clone(),
            partition_spec.clone(),
        );
    }
    for entry in &plan.added {
        summary.add_file(
            &to_data_file(entry, &ctx)?,
            schema.clone(),
            partition_spec.clone(),
        );
    }

    // New manifests first, then the carried ones: a reader that stops early
    // sees what this commit changed.
    let mut written: Vec<WrittenManifest> = Vec::new();
    let mut manifest_index = 0;
    let mut write = |entries: Vec<EntryWrite>, written: &mut Vec<WrittenManifest>| -> Result<()> {
        let name = manifest_file_name(commit, manifest_index);
        manifest_index += 1;
        written.push(write_manifest(
            &state.location.metadata_uri(&name),
            format_version,
            snapshot_id,
            schema,
            partition_spec,
            entries,
            &ctx,
        )?);
        Ok(())
    };
    if !plan.added.is_empty() {
        let entries = plan
            .added
            .into_iter()
            .map(|entry| EntryWrite {
                entry,
                status: ManifestStatus::Added,
            })
            .collect();
        write(entries, &mut written)?;
    }
    if let Some(rewritten) = plan.rewritten {
        write(rewritten, &mut written)?;
    }
    if let Some(merged) = plan.merged {
        let entries = merged
            .into_iter()
            .map(|entry| EntryWrite {
                entry,
                status: ManifestStatus::Existing,
            })
            .collect();
        write(entries, &mut written)?;
    }

    let mut manifest_list: Vec<ManifestFile> = written
        .iter()
        .map(|manifest| manifest.file.clone())
        .collect();
    manifest_list.extend(plan.carried.iter().cloned());
    let list_name = manifest_list_file_name(snapshot_id, commit);
    let list_uri = state.location.metadata_uri(&list_name);
    // Row lineage is a format version 3 feature: a version 2 table assigns
    // no row ids and its snapshot records no row range.
    let first_row_id = (format_version >= FormatVersion::V3).then(|| metadata.next_row_id());
    let list = write_manifest_list(
        &list_uri,
        format_version,
        snapshot_id,
        metadata.current_snapshot_id(),
        sequence_number,
        first_row_id,
        manifest_list.clone(),
    )?;

    // What the list writer assigned, stamped onto the manifests the table
    // holds and onto the entries they record.
    let mut manifests = ManifestCache::new();
    let mut entries = Vec::new();
    let mut objects = Vec::new();
    for (position, manifest) in written.into_iter().enumerate() {
        let WrittenManifest {
            mut file,
            bytes,
            live_entries,
        } = manifest;
        file.sequence_number = sequence_number;
        if file.min_sequence_number < 0 {
            file.min_sequence_number = sequence_number;
        }
        file.first_row_id = list.first_row_ids[position];
        let mut next_row_id = file.first_row_id.map(|first| first as i64);
        let stamped: Vec<DataFileEntry> = live_entries
            .into_iter()
            .map(|mut entry| {
                entry.origin = Some(match entry.origin.take() {
                    Some(mut origin) => {
                        origin.manifest_path = file.manifest_path.clone();
                        origin
                    }
                    None => {
                        let assigned = next_row_id;
                        if let (Some(next), Some(stats)) = (next_row_id.as_mut(), &entry.stats) {
                            *next += stats.num_records.unwrap_or(0);
                        }
                        ManifestOrigin {
                            snapshot_id,
                            sequence_number,
                            file_sequence_number: sequence_number,
                            first_row_id: assigned,
                            manifest_path: file.manifest_path.clone(),
                        }
                    }
                });
                entry
            })
            .collect();
        manifest_list[position] = file.clone();
        entries.extend(stamped.iter().cloned());
        let key = state
            .location
            .metadata_key(&metadata_object_name(&file.manifest_path));
        objects.push((key, bytes));
        manifests.insert(
            file.manifest_path.clone(),
            Arc::new(LoadedManifest {
                file,
                entries: stamped,
            }),
        );
    }
    for carried in &plan.carried {
        let loaded = Arc::clone(
            state
                .manifests
                .get(&carried.manifest_path)
                .expect("a carried manifest is one the state holds"),
        );
        entries.extend(loaded.entries.iter().cloned());
        manifests.insert(carried.manifest_path.clone(), loaded);
    }
    let manifest_list_size = Some(list.bytes.len() as u64);
    objects.push((state.location.metadata_key(&list_name), list.bytes));

    let operation = if removed.is_empty() {
        Operation::Append
    } else if data_change {
        Operation::Overwrite
    } else {
        Operation::Replace
    };
    let mut properties = summary.build();
    properties.insert("total-data-files".to_string(), entries.len().to_string());
    properties.insert(
        "total-records".to_string(),
        entries
            .iter()
            .filter_map(|entry| entry.stats.as_ref()?.num_records)
            .sum::<i64>()
            .to_string(),
    );
    properties.insert(
        "total-files-size".to_string(),
        entries
            .iter()
            .map(|entry| entry.file.size)
            .sum::<u64>()
            .to_string(),
    );
    let snapshot = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        .with_parent_snapshot_id(metadata.current_snapshot_id())
        .with_sequence_number(sequence_number)
        .with_timestamp_ms(timestamp_ms)
        .with_manifest_list(list_uri)
        .with_summary(Summary {
            operation,
            additional_properties: properties,
        })
        .with_schema_id(metadata.current_schema_id());
    let snapshot = match (first_row_id, list.next_row_id) {
        (Some(first), Some(next)) => snapshot.with_row_range(first, next - first).build(),
        _ => snapshot.build(),
    };

    let expired = if catalog.expires_snapshots() {
        expired_snapshot_ids(metadata, now_ms)?
    } else {
        Vec::new()
    };
    let metadata = metadata
        .as_ref()
        .clone()
        .into_builder(base.map(|base| base.metadata_location.clone()))
        .set_branch_snapshot(snapshot.clone(), MAIN_BRANCH)
        .map_err(Error::spec)?
        .remove_snapshots(&expired)
        .build()
        .map_err(Error::spec)?
        .metadata;

    Ok(PreparedCommit {
        metadata: Arc::new(metadata),
        changes: CommitChanges {
            snapshot: Some(snapshot),
            expired,
        },
        objects,
        manifest_list,
        manifest_list_size,
        manifests,
        entries,
    })
}

/// The file name a metadata object's URI ends in.
fn metadata_object_name(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}

/// Store `prepared`'s objects and publish its metadata through `catalog` as
/// the successor of `base`. `Ok(None)` when another writer published first.
fn land(
    catalog: &dyn TableCatalog,
    state: &TableState,
    prepared: PreparedCommit,
    base: Option<&CurrentMetadata>,
) -> Result<Option<TableState>> {
    let location = &state.location;
    write_objects(location, &prepared.objects)?;
    let Some(current) = catalog.publish(base, Arc::clone(&prepared.metadata), &prepared.changes)?
    else {
        withdraw(location, &prepared.objects);
        return Ok(None);
    };
    let metadata = &current.metadata;
    Ok(Some(TableState {
        location: location.clone(),
        partition_by: identity_columns(
            metadata.current_schema(),
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        ),
        sort_by: identity_columns(
            metadata.current_schema(),
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        ),
        current: Arc::new(current),
        columns: state.columns.clone(),
        manifest_list: prepared.manifest_list,
        manifest_list_size: prepared.manifest_list_size,
        manifests: prepared.manifests,
        entries: prepared.entries,
    }))
}

/// Store every object of a commit, retrying a transient failure.
fn write_objects(location: &TableLocation, objects: &[(ObjectPath, Vec<u8>)]) -> Result<()> {
    let store = location.store();
    for (key, bytes) in objects {
        retry_io(|| store.put(key, bytes))?;
    }
    Ok(())
}

/// Delete the objects of a commit that did not land. Best effort: an object
/// left behind is an orphan a sweep reclaims.
fn withdraw(location: &TableLocation, objects: &[(ObjectPath, Vec<u8>)]) {
    for (key, _) in objects {
        if let Err(error) = location.store().delete(key) {
            tracing::warn!(key = %key, error = %error, "deleting a withdrawn commit's object failed");
        }
    }
}

/// Run `write` until it succeeds, retrying an IO or HTTP failure up to
/// [`COMMIT_RETRIES`] times with a jittered backoff.
pub fn retry_io<T>(mut write: impl FnMut() -> object_storage::Result<T>) -> Result<T> {
    let mut last_error = None;
    for attempt in 0..COMMIT_RETRIES {
        match write() {
            Ok(value) => return Ok(value),
            Err(error @ (StoreError::Io { .. } | StoreError::Http(_))) => {
                tracing::warn!(attempt, error = %error, "a commit write failed; retrying");
                last_error = Some(error);
            }
            Err(error) => return Err(error.into()),
        }
        let ceiling = RETRY_BACKOFF_BASE
            .saturating_mul(1 << attempt.min(16))
            .min(RETRY_BACKOFF_CAP);
        std::thread::sleep(jittered(ceiling));
    }
    Err(Error::CommitRetriesExhausted {
        attempts: COMMIT_RETRIES,
        source: last_error.expect("a retry only follows a recorded failure"),
    })
}

/// A sleep drawn from `[0, ceiling)`, seeded from the clock's subsecond
/// nanoseconds: enough to keep two writers from retrying in lockstep.
fn jittered(ceiling: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .subsec_nanos() as u64;
    Duration::from_nanos(nanos % (ceiling.as_nanos() as u64).max(1))
}

pub fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

/// A snapshot id no snapshot of `metadata` has: random and positive, drawn
/// again on the rare collision.
fn fresh_snapshot_id(metadata: &TableMetadata) -> i64 {
    loop {
        let (high, _) = Uuid::new_v4().as_u64_pair();
        let candidate = ((high >> 1) as i64).max(1);
        if metadata.snapshot_by_id(candidate).is_none() {
            return candidate;
        }
    }
}

/// When the commit happened, as the snapshot records it: now, but never
/// behind the table's last update or its last snapshot. A writer whose clock
/// lags another's would otherwise fail the builder's check, and a timestamp
/// that only ever errs later keeps data a little longer rather than
/// expiring it early.
fn commit_timestamp_ms(metadata: &TableMetadata, now_ms: i64) -> i64 {
    let last_snapshot = metadata
        .history()
        .last()
        .map(|entry| entry.timestamp_ms)
        .unwrap_or(i64::MIN);
    now_ms.max(metadata.last_updated_ms()).max(last_snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{DirectoryCatalog, metadata_location_version};
    use crate::layout::{VERSION_HINT_FILE, parse_hint, parse_metadata_file_version};
    use crate::load::{load_table, refresh_table};
    use arrow_array::{ArrayRef, Int64Array, Scalar};
    use object_storage::{FileRef, LocalStore, ObjectStore};
    use parquet_engine::{FileStats, PartitionValues, scalar_equal};
    use planner::types::Type;

    const HOUR_MS: i64 = 60 * 60 * 1000;

    fn columns() -> Vec<Column> {
        vec![Column {
            name: "v".to_string(),
            col_type: Type::Int64,
        }]
    }

    /// An entry for `name` holding `rows` rows of `v` in `low..=high`.
    fn entry(name: &str, rows: i64, low: i64, high: i64) -> DataFileEntry {
        let bound = |value: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![value])) };
        DataFileEntry {
            file: FileRef {
                path: ObjectPath::new(format!("{name}.parquet")),
                size: 10,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(rows),
                min_values: HashMap::from([("v".to_string(), bound(low))]),
                max_values: HashMap::from([("v".to_string(), bound(high))]),
                null_counts: HashMap::from([("v".to_string(), 0)]),
            })),
            origin: None,
        }
    }

    fn int_scalar(value: i64) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        location: TableLocation,
        catalog: DirectoryCatalog,
    }

    fn create_table(partition_by: &[String]) -> (Fixture, TableState) {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::new(store, ObjectPath::new("t1")).unwrap();
        let catalog = DirectoryCatalog::new(location.clone());
        let state = initialize_table(
            &catalog,
            &location,
            "main.t1",
            Uuid::new_v4(),
            TableCreation {
                columns: &columns(),
                partition_by,
                sort_by: &[],
                adopted: Vec::new(),
            },
            0,
        )
        .unwrap();
        (
            Fixture {
                _dir: dir,
                location,
                catalog,
            },
            state,
        )
    }

    fn append(
        fixture: &Fixture,
        state: &TableState,
        entries: Vec<DataFileEntry>,
        now_ms: i64,
    ) -> TableState {
        commit_file_changes(state, &fixture.catalog, &[], entries, true, now_ms)
            .unwrap()
            .expect("no other writer")
    }

    fn version(state: &TableState) -> u64 {
        metadata_location_version(&state.current.metadata_location).unwrap()
    }

    fn load(fixture: &Fixture) -> TableState {
        load_table(&fixture.catalog, &fixture.location, "main.t1").unwrap()
    }

    fn metadata_versions(location: &TableLocation) -> Vec<u64> {
        let mut versions: Vec<u64> = location
            .store()
            .list(&location.metadata_dir())
            .unwrap()
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        versions
    }

    fn avro_objects(location: &TableLocation) -> usize {
        location
            .store()
            .list(&location.metadata_dir())
            .unwrap()
            .objects
            .iter()
            .filter(|object| object.file.path.as_str().ends_with(".avro"))
            .count()
    }

    fn hint(location: &TableLocation) -> u64 {
        parse_hint(
            &location
                .store()
                .get(&location.metadata_key(VERSION_HINT_FILE))
                .unwrap()
                .unwrap(),
        )
        .unwrap()
    }

    /// The live files' names, sorted: a state lists new manifests before
    /// carried ones, so the order is not what a test asserts on.
    fn paths(state: &TableState) -> Vec<&str> {
        let mut paths: Vec<&str> = state
            .entries
            .iter()
            .map(|entry| entry.file.path.as_str())
            .collect();
        paths.sort_unstable();
        paths
    }

    fn origin_of(state: &TableState, path: &str) -> ManifestOrigin {
        state
            .entries
            .iter()
            .find(|entry| entry.file.path.as_str() == path)
            .unwrap_or_else(|| panic!("{path} is live"))
            .origin
            .clone()
            .unwrap()
    }

    #[test]
    fn a_new_table_is_version_one_with_a_hint_and_no_snapshot() {
        let (fixture, state) = create_table(&[]);

        let loaded = load(&fixture);

        assert_eq!(version(&state), 1);
        assert_eq!(metadata_versions(&fixture.location), [1]);
        assert_eq!(hint(&fixture.location), 1);
        assert_eq!(version(&loaded), 1);
        assert_eq!(loaded.current.metadata.format_version(), FormatVersion::V3);
        assert_eq!(
            loaded.current.metadata.uuid(),
            state.current.metadata.uuid()
        );
        assert!(loaded.current.metadata.current_snapshot().is_none());
        assert!(loaded.entries.is_empty());
        assert_eq!(loaded.columns, columns());
    }

    #[test]
    fn appends_land_at_successive_versions_with_contiguous_row_ids() {
        let (fixture, created) = create_table(&[]);

        let after_a = append(&fixture, &created, vec![entry("a", 100, 1, 100)], 1_000);
        let after_b = append(&fixture, &after_a, vec![entry("b", 50, 101, 150)], 2_000);
        let loaded = load(&fixture);

        assert_eq!((version(&after_a), version(&after_b)), (2, 3));
        assert_eq!(paths(&loaded), ["a.parquet", "b.parquet"]);
        let (a, b) = (
            origin_of(&loaded, "a.parquet"),
            origin_of(&loaded, "b.parquet"),
        );
        assert_eq!((a.sequence_number, b.sequence_number), (1, 2));
        assert_eq!((a.first_row_id, b.first_row_id), (Some(0), Some(100)));
        assert_eq!(a, origin_of(&after_b, "a.parquet"));
        assert_eq!(b, origin_of(&after_b, "b.parquet"));
        assert_eq!(loaded.current.metadata.next_row_id(), 150);
        assert_eq!(loaded.current.metadata.snapshots().len(), 2);
    }

    #[test]
    fn the_second_writer_from_one_base_loses_and_leaves_no_objects() {
        let (fixture, created) = create_table(&[]);
        let after_a = append(&fixture, &created, vec![entry("a", 1, 1, 1)], 1_000);
        let objects_before = avro_objects(&fixture.location);

        let lost = commit_file_changes(
            &created,
            &fixture.catalog,
            &[],
            vec![entry("b", 1, 2, 2)],
            true,
            1_000,
        )
        .unwrap();

        assert!(lost.is_none());
        assert_eq!(metadata_versions(&fixture.location), [1, 2]);
        assert_eq!(avro_objects(&fixture.location), objects_before);
        assert_eq!(paths(&after_a), ["a.parquet"]);
    }

    #[test]
    fn a_removal_retires_the_file_and_carries_the_rest_with_their_numbers() {
        let (fixture, created) = create_table(&[]);
        let state = append(
            &fixture,
            &created,
            vec![entry("a", 10, 1, 10), entry("b", 10, 11, 20)],
            1_000,
        );
        let state = append(&fixture, &state, vec![entry("c", 10, 21, 30)], 2_000);
        let removed = state
            .entries
            .iter()
            .find(|entry| entry.file.path.as_str() == "a.parquet")
            .unwrap()
            .clone();
        let b_origin = origin_of(&state, "b.parquet");

        let replaced = commit_file_changes(
            &state,
            &fixture.catalog,
            &[removed],
            vec![entry("d", 10, 1, 10)],
            false,
            3_000,
        )
        .unwrap()
        .unwrap();
        let loaded = load(&fixture);

        assert_eq!(paths(&replaced), ["b.parquet", "c.parquet", "d.parquet"]);
        assert_eq!(paths(&loaded), paths(&replaced));
        let b = origin_of(&loaded, "b.parquet");
        assert_eq!(b.sequence_number, b_origin.sequence_number);
        assert_eq!(b.first_row_id, b_origin.first_row_id);
        assert_ne!(b.manifest_path, b_origin.manifest_path);
        let snapshot = loaded.current.metadata.current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        assert_eq!(
            snapshot.summary().additional_properties["deleted-data-files"],
            "1"
        );
        assert_eq!(loaded.manifest_list.len(), 3);
    }

    #[test]
    fn refresh_sees_a_version_committed_from_another_copy() {
        let (fixture, created) = create_table(&[]);
        let reader = load(&fixture);
        let writer = append(&fixture, &created, vec![entry("a", 1, 1, 1)], 1_000);

        let refreshed = refresh_table(&fixture.catalog, &reader, "main.t1")
            .unwrap()
            .unwrap();
        let unchanged = refresh_table(&fixture.catalog, &refreshed, "main.t1").unwrap();

        assert_eq!(version(&refreshed), version(&writer));
        assert_eq!(paths(&refreshed), ["a.parquet"]);
        assert!(unchanged.is_none());
    }

    #[test]
    fn snapshots_past_the_window_are_dropped_by_the_next_commit() {
        // A fresh table is stamped with the wall clock, and a commit never
        // dates itself before the table's last update, so time runs from now.
        let created_at = now_unix_ms();
        let (fixture, created) = create_table(&[]);
        let state = append(&fixture, &created, vec![entry("a", 1, 1, 1)], created_at);
        let state = append(
            &fixture,
            &state,
            vec![entry("b", 1, 2, 2)],
            created_at + HOUR_MS,
        );

        let state = append(
            &fixture,
            &state,
            vec![entry("c", 1, 3, 3)],
            created_at + 6 * HOUR_MS,
        );

        let sequence_numbers: Vec<i64> = state
            .current
            .metadata
            .snapshots()
            .map(|snapshot| snapshot.sequence_number())
            .collect();
        assert_eq!(state.current.metadata.snapshots().len(), 2);
        assert!(!sequence_numbers.contains(&1));
        assert_eq!(paths(&state), ["a.parquet", "b.parquet", "c.parquet"]);
    }

    #[test]
    fn expiry_alone_commits_a_metadata_only_version() {
        // A metadata-only commit is dated by the wall clock, so the snapshots
        // it expires are dated by it too, and only the sweep's clock moves.
        let (fixture, created) = create_table(&[]);
        let state = append(&fixture, &created, vec![entry("a", 1, 1, 1)], now_unix_ms());
        let state = append(&fixture, &state, vec![entry("b", 1, 2, 2)], now_unix_ms());

        let nothing_yet =
            expire_snapshots(&state, &fixture.catalog, now_unix_ms() + 2 * HOUR_MS).unwrap();
        let expired = expire_snapshots(&state, &fixture.catalog, now_unix_ms() + 6 * HOUR_MS)
            .unwrap()
            .unwrap();

        assert!(nothing_yet.is_none());
        assert_eq!(version(&expired), version(&state) + 1);
        assert_eq!(expired.current.metadata.snapshots().len(), 1);
        assert_eq!(
            expired.current.metadata.current_snapshot_id(),
            state.current.metadata.current_snapshot_id()
        );
        assert_eq!(paths(&expired), paths(&state));
    }

    #[test]
    fn the_hint_follows_every_version() {
        let (fixture, mut state) = create_table(&[]);

        for version in 2..=11u64 {
            state = append(
                &fixture,
                &state,
                vec![entry(&format!("f{version}"), 1, 1, 1)],
                i64::try_from(version).unwrap(),
            );
            assert_eq!(hint(&fixture.location), version, "after version {version}");
        }
    }

    #[test]
    fn the_snapshot_timestamp_never_runs_behind_the_last_update() {
        let (fixture, created) = create_table(&[]);
        let state = append(&fixture, &created, vec![entry("a", 1, 1, 1)], 5_000);

        let behind = append(&fixture, &state, vec![entry("b", 1, 2, 2)], 1_000);

        let timestamps: Vec<i64> = behind
            .current
            .metadata
            .history()
            .iter()
            .map(|entry| entry.timestamp_ms)
            .collect();
        assert!(timestamps[1] >= timestamps[0]);
    }

    #[test]
    fn a_partitioned_table_records_each_files_tuple() {
        let (fixture, created) = create_table(&["v".to_string()]);
        let mut partitioned = entry("a", 3, 7, 7);
        partitioned.partition = Some(PartitionValues::from([("v".to_string(), int_scalar(7))]));

        let state = append(&fixture, &created, vec![partitioned], 1_000);
        let loaded = load(&fixture);

        assert_eq!(state.partition_by, ["v"]);
        let tuple = loaded.entries[0].partition.clone().unwrap();
        assert_eq!(scalar_equal(&tuple["v"], &int_scalar(7)), Some(true));
    }

    /// A catalog that refuses the first publish, the way a service does when
    /// another writer's commit landed in between.
    struct RacedCatalog {
        inner: DirectoryCatalog,
        refusals_left: std::sync::atomic::AtomicUsize,
    }

    impl TableCatalog for RacedCatalog {
        fn load(&self) -> Result<CurrentMetadata> {
            self.inner.load()
        }

        fn refresh(&self, held: &CurrentMetadata) -> Result<Option<CurrentMetadata>> {
            self.inner.refresh(held)
        }

        fn publish(
            &self,
            base: Option<&CurrentMetadata>,
            metadata: Arc<TableMetadata>,
            changes: &CommitChanges,
        ) -> Result<Option<CurrentMetadata>> {
            let refusals = &self.refusals_left;
            if refusals.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                refusals.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                return Ok(None);
            }
            self.inner.publish(base, metadata, changes)
        }

        fn expires_snapshots(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_refused_publish_withdraws_its_objects_and_a_retry_lands() {
        let (fixture, created) = create_table(&[]);
        let raced = RacedCatalog {
            inner: DirectoryCatalog::new(fixture.location.clone()),
            refusals_left: std::sync::atomic::AtomicUsize::new(1),
        };
        let objects_before = avro_objects(&fixture.location);

        let refused = commit_file_changes(
            &created,
            &raced,
            &[],
            vec![entry("a", 1, 1, 1)],
            true,
            1_000,
        )
        .unwrap();
        let objects_after_refusal = avro_objects(&fixture.location);
        let landed = commit_file_changes(
            &created,
            &raced,
            &[],
            vec![entry("a", 1, 1, 1)],
            true,
            1_000,
        )
        .unwrap()
        .unwrap();

        assert!(refused.is_none());
        assert_eq!(objects_after_refusal, objects_before);
        assert_eq!(paths(&landed), ["a.parquet"]);
        assert_eq!(version(&landed), 2);
        assert!(landed.current.metadata.current_snapshot().is_some());
    }
}
