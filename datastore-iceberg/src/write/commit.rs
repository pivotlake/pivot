//! Landing an INSERT's files as one append snapshot of a catalog table.
//!
//! The commit is built on the table as the catalog has it at commit time: a
//! manifest of the added files, a manifest list naming it ahead of the
//! current snapshot's manifests, and the snapshot. Both Avro objects are
//! stored through the table's own store; then one `updateTable` adds the
//! snapshot and moves `main` onto it, under the requirement that `main`
//! still names the snapshot the commit was built on. A catalog that refuses
//! the requirement answers a conflict: another writer landed first, the
//! objects are withdrawn, and the commit is built again on the winner.

use std::collections::HashMap;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use iceberg::spec::{
    FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestFile, ManifestList, Operation,
    Snapshot, SnapshotReference, SnapshotRetention, SnapshotSummaryCollector, Summary,
    TableMetadata,
};
use iceberg::table::Table;
use iceberg::{Catalog, ErrorKind, TableCommit, TableIdent, TableRequirement, TableUpdate};
use iceberg_catalog_rest::RestCatalog;
use object_storage::{ObjectPath, ObjectStore, StoreError, load_objects};
use uuid::Uuid;

use crate::rest_catalog::block_on;
use crate::write::data_files::{AddedFile, EntryContext, to_data_file};
use crate::write::manifest_writer::{write_data_manifest, write_manifest_list};
use crate::write::{WriteTarget, table_ident};
use crate::{Error, Result};

/// How many times a writer that lost the race rebuilds its commit on the
/// winner before giving up, and how many times a store write that failed
/// with an IO error is retried.
pub(crate) const COMMIT_RETRIES: usize = 10;
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// The table at the version a commit landed, with the size of the manifest
/// list that version names (what reading the table through the pool needs).
pub(crate) struct Committed {
    pub table: Table,
    pub manifest_list_size: u64,
}

/// Commit `files`, written for `target`, as one append snapshot of the table,
/// building the commit again on the winner each time another writer lands
/// first.
pub(crate) fn commit_append(
    catalog: &RestCatalog,
    dispatcher: &DataFlowDispatcher,
    target: &WriteTarget,
    files: &[AddedFile],
) -> Result<Committed> {
    let ident = table_ident(&target.name);
    for _ in 0..COMMIT_RETRIES {
        let table = block_on(catalog.load_table(&ident)).map_err(Box::new)?;
        let prepared = prepare_append(dispatcher, target, table.metadata(), files)?;
        write_objects(target.store.store().as_ref(), &prepared.objects)?;
        let published = publish(catalog, &ident, table.metadata(), &prepared.snapshot);
        match published {
            Ok(Some(table)) => {
                return Ok(Committed {
                    table,
                    manifest_list_size: prepared.manifest_list_size,
                });
            }
            Ok(None) => withdraw(target.store.store().as_ref(), &prepared.objects),
            Err(error) => {
                withdraw(target.store.store().as_ref(), &prepared.objects);
                return Err(error);
            }
        }
    }
    Err(Error::CommitRacesExhausted {
        table: target.name.to_string(),
        attempts: COMMIT_RETRIES,
    })
}

/// An append built in memory and not yet written: the snapshot to add and
/// the Avro objects it names, by key under the table's store.
struct PreparedAppend {
    snapshot: Snapshot,
    objects: Vec<(ObjectPath, Vec<u8>)>,
    manifest_list_size: u64,
}

/// Build the snapshot appending `files` to the table at `metadata`: one new
/// manifest, listed ahead of the current snapshot's manifests, which are
/// carried unchanged (delete manifests included: their deletes apply by
/// sequence number to the files committed before them, never to these).
fn prepare_append(
    dispatcher: &DataFlowDispatcher,
    target: &WriteTarget,
    metadata: &TableMetadata,
    files: &[AddedFile],
) -> Result<PreparedAppend> {
    let table_name = target.name.to_string();
    let format_version = metadata.format_version();
    let schema = metadata.current_schema();
    let partition_spec = metadata.default_partition_spec();
    // The files carry the partition tuples of the spec the insert was bound
    // under; a spec that changed since would record them under the wrong
    // fields.
    if partition_spec.spec_id() != target.partition_spec_id {
        return Err(Error::PartitionSpecChanged { table: table_name });
    }
    let ctx = EntryContext {
        schema,
        partition_spec,
        partition_value_types: &target.partition_keys.value_types,
    };
    let data_files = files
        .iter()
        .map(|file| to_data_file(file, &ctx))
        .collect::<Result<Vec<_>>>()?;
    let mut summary = SnapshotSummaryCollector::default();
    for data_file in &data_files {
        summary.add_file(data_file, schema.clone(), partition_spec.clone());
    }

    let current_manifests = read_current_manifests(dispatcher, target, metadata)?;
    let snapshot_id = fresh_snapshot_id(metadata);
    let sequence_number = metadata.next_sequence_number();
    let commit = Uuid::new_v4();
    let manifest_key = target.metadata_dir.join(&format!("{commit}-m0.avro"));
    let (manifest, manifest_bytes) = write_data_manifest(
        &target.store.uri_of(&manifest_key),
        format_version,
        snapshot_id,
        schema,
        partition_spec,
        data_files,
    )?;

    // The new manifest first, then the carried ones: a reader that stops
    // early sees what this commit changed.
    let mut manifests = vec![manifest];
    manifests.extend(current_manifests);
    let list_key = target
        .metadata_dir
        .join(&format!("snap-{snapshot_id}-1-{commit}.avro"));
    let list_uri = target.store.uri_of(&list_key);
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
        manifests.clone(),
    )?;

    let mut properties = summary.build();
    let added_bytes = files.iter().map(|file| file.size).sum::<u64>();
    properties.extend(total_properties(
        &manifests,
        metadata
            .current_snapshot()
            .map(|snapshot| snapshot.summary()),
        added_bytes,
    ));
    let snapshot = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        .with_parent_snapshot_id(metadata.current_snapshot_id())
        .with_sequence_number(sequence_number)
        .with_timestamp_ms(commit_timestamp_ms(metadata, now_unix_ms()))
        .with_manifest_list(list_uri)
        .with_summary(Summary {
            operation: Operation::Append,
            additional_properties: properties,
        })
        .with_schema_id(metadata.current_schema_id());
    let snapshot = match (first_row_id, list.next_row_id) {
        (Some(first), Some(next)) => snapshot.with_row_range(first, next - first).build(),
        _ => snapshot.build(),
    };
    let manifest_list_size = list.bytes.len() as u64;
    Ok(PreparedAppend {
        snapshot,
        objects: vec![(manifest_key, manifest_bytes), (list_key, list.bytes)],
        manifest_list_size,
    })
}

/// The manifests the table's current snapshot names, in list order, read
/// through the pool; none for a table without a snapshot.
fn read_current_manifests(
    dispatcher: &DataFlowDispatcher,
    target: &WriteTarget,
    metadata: &TableMetadata,
) -> Result<Vec<ManifestFile>> {
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(Vec::new());
    };
    let path = snapshot.manifest_list();
    let size = target
        .store
        .object_size(path)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: target.name.to_string(),
            path: path.to_string(),
        })?;
    let bytes = load_objects(dispatcher, &[target.store.data_file(path, size)?])?
        .pop()
        .expect("load_objects returns one object per file")
        .bytes;
    let list =
        ManifestList::parse_with_version(&bytes, metadata.format_version()).map_err(|source| {
            Error::MalformedMetadataObject {
                path: path.to_string(),
                source: Box::new(source),
            }
        })?;
    Ok(list.consume_entries().into_iter().collect())
}

/// The table-wide totals the snapshot summary records, as the new manifest
/// list has them: what every manifest counts, and the files' bytes carried
/// forward from the previous snapshot's total. A total whose counts some
/// manifest leaves out is left out itself rather than recorded short.
fn total_properties(
    manifests: &[ManifestFile],
    previous: Option<&Summary>,
    added_bytes: u64,
) -> HashMap<String, String> {
    let mut properties = HashMap::new();
    let sum =
        |content: ManifestContentType, count: fn(&ManifestFile) -> Option<u64>| -> Option<u64> {
            manifests
                .iter()
                .filter(|manifest| manifest.content == content)
                .map(count)
                .sum::<Option<u64>>()
        };
    let files = |manifest: &ManifestFile| {
        Some(u64::from(manifest.added_files_count?) + u64::from(manifest.existing_files_count?))
    };
    let rows =
        |manifest: &ManifestFile| Some(manifest.added_rows_count? + manifest.existing_rows_count?);
    if let Some(total) = sum(ManifestContentType::Data, files) {
        properties.insert("total-data-files".to_string(), total.to_string());
    }
    if let Some(total) = sum(ManifestContentType::Data, rows) {
        properties.insert("total-records".to_string(), total.to_string());
    }
    if let Some(total) = sum(ManifestContentType::Deletes, files) {
        properties.insert("total-delete-files".to_string(), total.to_string());
    }
    let previous_bytes = match previous {
        Some(summary) => summary
            .additional_properties
            .get("total-files-size")
            .and_then(|total| total.parse::<u64>().ok()),
        None => Some(0),
    };
    if let Some(previous_bytes) = previous_bytes {
        properties.insert(
            "total-files-size".to_string(),
            (previous_bytes + added_bytes).to_string(),
        );
    }
    // An append changes no deletes: the previous snapshot's counts stand.
    if let Some(summary) = previous {
        for key in ["total-position-deletes", "total-equality-deletes"] {
            if let Some(value) = summary.additional_properties.get(key) {
                properties.insert(key.to_string(), value.clone());
            }
        }
    }
    properties
}

/// Add `snapshot` and move `main` onto it, provided the table is still the
/// one the commit was built on (same uuid, `main` at the same snapshot). The
/// catalog's conflict answer is a lost race, `None`.
fn publish(
    catalog: &RestCatalog,
    ident: &TableIdent,
    base: &TableMetadata,
    snapshot: &Snapshot,
) -> Result<Option<Table>> {
    let requirements = vec![
        TableRequirement::UuidMatch { uuid: base.uuid() },
        TableRequirement::RefSnapshotIdMatch {
            r#ref: MAIN_BRANCH.to_string(),
            snapshot_id: base.current_snapshot_id(),
        },
    ];
    let updates = vec![
        TableUpdate::AddSnapshot {
            snapshot: snapshot.clone(),
        },
        TableUpdate::SetSnapshotRef {
            ref_name: MAIN_BRANCH.to_string(),
            reference: SnapshotReference::new(
                snapshot.snapshot_id(),
                SnapshotRetention::branch(None, None, None),
            ),
        },
    ];
    let commit = TableCommit::new(ident.clone(), requirements, updates);
    match block_on(catalog.update_table(commit)) {
        Ok(table) => Ok(Some(table)),
        Err(error) if error.kind() == ErrorKind::CatalogCommitConflicts => Ok(None),
        Err(error) => Err(Error::Catalog(Box::new(error))),
    }
}

/// Store every object of a commit, retrying a transient failure.
fn write_objects(store: &dyn ObjectStore, objects: &[(ObjectPath, Vec<u8>)]) -> Result<()> {
    for (key, bytes) in objects {
        retry_io(|| store.put(key, bytes))?;
    }
    Ok(())
}

/// Delete the objects of a commit that did not land. Best effort: an object
/// left behind is an orphan the catalog's own maintenance reclaims.
fn withdraw(store: &dyn ObjectStore, objects: &[(ObjectPath, Vec<u8>)]) {
    for (key, _) in objects {
        if let Err(error) = store.delete(key) {
            tracing::warn!(key = %key, error = %error, "deleting a withdrawn commit's object failed");
        }
    }
}

/// Run `write` until it succeeds, retrying an IO or HTTP failure up to
/// [`COMMIT_RETRIES`] times with a jittered backoff.
fn retry_io<T>(mut write: impl FnMut() -> object_storage::Result<T>) -> Result<T> {
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

fn now_unix_ms() -> i64 {
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
/// lags another's would otherwise fail the catalog's check, and a timestamp
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
