//! Reading and writing a managed table's Avro manifests and manifest lists,
//! and deciding which manifests a commit rewrites, carries or merges.
//!
//! The spec crate's writers want an `OutputFile`; they are given one in an
//! in-memory file IO, and the bytes are read back out and stored through the
//! datastore's own object store, so a manifest lands in the store exactly the
//! way a data file does and through the same credentials.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    FormatVersion, Manifest, ManifestContentType, ManifestFile,
    ManifestList as IcebergManifestList, ManifestListWriter, ManifestStatus, ManifestWriterBuilder,
    PartitionSpecRef, SchemaRef,
};

use super::data_files::{DataFileEntry, EntryContext, from_manifest_entry, to_data_file};
use super::{Error, Result};
use datastore_iceberg::block_on;

/// A commit that would carry more manifests than this merges the smallest of
/// them into one, so the number of manifests a reader fetches stays bounded
/// however many appends the table has seen.
pub(crate) const MIN_MANIFESTS_TO_MERGE: usize = 100;

/// A manifest the table's current snapshot names, with its live entries
/// decoded: what a read serves files from and a commit rewrites from.
pub(crate) struct LoadedManifest {
    pub file: ManifestFile,
    /// The ADDED and EXISTING entries, in manifest order, each stamped with
    /// where this manifest records it.
    pub entries: Vec<DataFileEntry>,
}

/// The current snapshot's manifests by path. Manifests are immutable, so an
/// entry stays valid for as long as its path is named by some snapshot; a
/// refresh fetches only the paths it does not hold.
pub(crate) type ManifestCache = HashMap<String, Arc<LoadedManifest>>;

/// The data manifests a manifest list names, in list order. A delete manifest
/// with live files is refused: row-level deletes are not applied, so a table
/// carrying them would read wrongly.
pub(crate) fn read_manifest_list(
    table: &str,
    path: &str,
    bytes: &[u8],
) -> Result<Vec<ManifestFile>> {
    let list =
        IcebergManifestList::parse_with_version(bytes, FormatVersion::V3).map_err(|source| {
            Error::MalformedMetadataObject {
                table: table.to_string(),
                path: path.to_string(),
                source: Box::new(source),
            }
        })?;
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

/// The manifest `file` names, decoded from `bytes`: its live entries with
/// everything the manifest leaves implicit filled in. First row ids are
/// assigned the way the spec inherits them: an entry without one gets the
/// manifest's, advanced by the row counts of the entries assigned before it.
pub(crate) fn read_manifest(
    table: &str,
    file: &ManifestFile,
    bytes: &[u8],
    ctx: &EntryContext<'_>,
) -> Result<LoadedManifest> {
    let manifest =
        Manifest::parse_avro(bytes).map_err(|source| Error::MalformedMetadataObject {
            table: table.to_string(),
            path: file.manifest_path.clone(),
            source: Box::new(source),
        })?;
    let mut next_row_id = file.first_row_id.map(|first| first as i64);
    let mut entries = Vec::new();
    for entry in manifest.entries() {
        let data = entry.data_file();
        let first_row_id = match data.first_row_id() {
            Some(assigned) => Some(assigned),
            None if entry.is_alive() => {
                let assigned = next_row_id;
                if let Some(next) = next_row_id.as_mut() {
                    *next += data.record_count() as i64;
                }
                assigned
            }
            None => None,
        };
        if !entry.is_alive() {
            continue;
        }
        if data.file_format() != iceberg::spec::DataFileFormat::Parquet {
            return Err(Error::NonParquetFile {
                table: table.to_string(),
                file: data.file_path().to_string(),
                format: data.file_format().to_string(),
            });
        }
        entries.push(from_manifest_entry(entry, file, first_row_id, ctx)?);
    }
    Ok(LoadedManifest {
        file: file.clone(),
        entries,
    })
}

/// One entry a commit writes into a manifest, with the status it gets there.
pub(crate) struct EntryWrite {
    pub entry: DataFileEntry,
    pub status: ManifestStatus,
}

/// What a commit does with the current snapshot's manifests.
pub(crate) struct ManifestPlan {
    /// Every live entry of the manifests a removal touched, as one manifest
    /// to write: the removed files DELETED, the rest EXISTING. `None` when
    /// nothing was removed.
    pub rewritten: Option<Vec<EntryWrite>>,
    /// The entries the commit adds, as one manifest to write.
    pub added: Vec<DataFileEntry>,
    /// Live entries of the smallest untouched manifests, merged into one
    /// EXISTING manifest to keep the manifest count bounded.
    pub merged: Option<Vec<DataFileEntry>>,
    /// Untouched manifests the new snapshot names unchanged, in list order.
    pub carried: Vec<ManifestFile>,
}

/// Decide the manifests for a commit removing `removed` and adding `added`
/// over the `current` manifests (in list order). Every removed entry must
/// name the manifest it was read from, and that manifest must be current.
pub(crate) fn plan_manifests(
    current: &[Arc<LoadedManifest>],
    removed: &[DataFileEntry],
    added: Vec<DataFileEntry>,
    min_manifests_to_merge: usize,
) -> Result<ManifestPlan> {
    let mut removed_paths_by_manifest: HashMap<&str, Vec<&str>> = HashMap::new();
    for entry in removed {
        let origin = entry
            .origin
            .as_ref()
            .ok_or_else(|| Error::UncommittedRemoval {
                file: entry.file.path.to_string(),
            })?;
        removed_paths_by_manifest
            .entry(origin.manifest_path.as_str())
            .or_default()
            .push(entry.file.path.as_str());
    }
    for manifest_path in removed_paths_by_manifest.keys() {
        if !current
            .iter()
            .any(|manifest| manifest.file.manifest_path == *manifest_path)
        {
            return Err(Error::UnknownManifest {
                manifest: manifest_path.to_string(),
            });
        }
    }

    let mut rewritten = Vec::new();
    let mut carried = Vec::new();
    for manifest in current {
        match removed_paths_by_manifest.get(manifest.file.manifest_path.as_str()) {
            Some(removed_paths) => {
                for entry in &manifest.entries {
                    let status = if removed_paths.contains(&entry.file.path.as_str()) {
                        ManifestStatus::Deleted
                    } else {
                        ManifestStatus::Existing
                    };
                    rewritten.push(EntryWrite {
                        entry: entry.clone(),
                        status,
                    });
                }
            }
            // A manifest whose files are all retired names nothing any more.
            None if manifest.entries.is_empty() => {}
            None => carried.push(Arc::clone(manifest)),
        }
    }

    let mut merged = None;
    if carried.len() > min_manifests_to_merge {
        let merge_count = carried.len() - min_manifests_to_merge + 1;
        let mut by_size: Vec<usize> = (0..carried.len()).collect();
        by_size.sort_by_key(|index| carried[*index].file.manifest_length);
        let mut merged_indexes: Vec<usize> = by_size.into_iter().take(merge_count).collect();
        merged_indexes.sort_unstable();
        let mut entries = Vec::new();
        for index in merged_indexes.iter().rev() {
            let manifest = carried.remove(*index);
            entries.splice(0..0, manifest.entries.iter().cloned());
        }
        merged = Some(entries);
    }

    Ok(ManifestPlan {
        rewritten: (!removed.is_empty()).then_some(rewritten),
        added,
        merged,
        carried: carried
            .into_iter()
            .map(|manifest| manifest.file.clone())
            .collect(),
    })
}

/// A manifest a commit wrote: its list entry (sequence numbers and first row
/// id unassigned until the manifest list assigns them), its bytes, and the
/// live entries it holds in order.
pub(crate) struct WrittenManifest {
    pub file: ManifestFile,
    pub bytes: Vec<u8>,
    pub live_entries: Vec<DataFileEntry>,
}

/// Write the data manifest at `uri` holding `entries` for the snapshot
/// `snapshot_id`. ADDED entries leave their sequence number for the manifest
/// list to assign; EXISTING and DELETED ones keep the numbers they were
/// committed with.
pub(crate) fn write_manifest(
    uri: &str,
    snapshot_id: i64,
    schema: &SchemaRef,
    partition_spec: &PartitionSpecRef,
    entries: Vec<EntryWrite>,
    ctx: &EntryContext<'_>,
) -> Result<WrittenManifest> {
    let file_io = FileIO::new_with_memory();
    let output = file_io.new_output(uri).map_err(Error::spec)?;
    let mut writer = ManifestWriterBuilder::new(
        output,
        Some(snapshot_id),
        schema.clone(),
        partition_spec.as_ref().clone(),
    )
    .build_v3_data();
    let mut live_entries = Vec::new();
    for EntryWrite { entry, status } in entries {
        let data_file = to_data_file(&entry, ctx)?;
        match status {
            ManifestStatus::Added => {
                writer.add_file(data_file, -1).map_err(Error::spec)?;
                live_entries.push(entry);
            }
            ManifestStatus::Existing => {
                let origin = entry
                    .origin
                    .as_ref()
                    .ok_or_else(|| Error::UncommittedRemoval {
                        file: entry.file.path.to_string(),
                    })?;
                writer
                    .add_existing_file(
                        data_file,
                        origin.snapshot_id,
                        origin.sequence_number,
                        Some(origin.file_sequence_number),
                    )
                    .map_err(Error::spec)?;
                live_entries.push(entry);
            }
            ManifestStatus::Deleted => {
                let origin = entry
                    .origin
                    .as_ref()
                    .ok_or_else(|| Error::UncommittedRemoval {
                        file: entry.file.path.to_string(),
                    })?;
                writer
                    .add_delete_file(
                        data_file,
                        origin.sequence_number,
                        Some(origin.file_sequence_number),
                    )
                    .map_err(Error::spec)?;
            }
        }
    }
    let file = block_on(writer.write_manifest_file()).map_err(Error::spec)?;
    let bytes = read_back(&file_io, uri)?;
    Ok(WrittenManifest {
        file,
        bytes,
        live_entries,
    })
}

/// A manifest list a commit wrote, with the first row id the list assigned
/// to each manifest (in the order given) and the row id the table continues
/// from.
pub(crate) struct WrittenManifestList {
    pub bytes: Vec<u8>,
    pub first_row_ids: Vec<Option<u64>>,
    pub next_row_id: u64,
}

/// Write the manifest list at `uri` for snapshot `snapshot_id` naming
/// `manifests` in order. A manifest this commit wrote gets this commit's
/// sequence number and the next free first row id, advanced by its rows; a
/// carried manifest passes through unchanged.
pub(crate) fn write_manifest_list(
    uri: &str,
    snapshot_id: i64,
    parent_snapshot_id: Option<i64>,
    sequence_number: i64,
    first_row_id: u64,
    manifests: Vec<ManifestFile>,
) -> Result<WrittenManifestList> {
    let file_io = FileIO::new_with_memory();
    let output = file_io.new_output(uri).map_err(Error::spec)?;
    let file_writer = block_on(output.writer()).map_err(Error::spec)?;
    let mut writer = ManifestListWriter::v3(
        file_writer,
        snapshot_id,
        parent_snapshot_id,
        sequence_number,
        Some(first_row_id),
    );
    let mut first_row_ids = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        let assigned = manifest.first_row_id.is_none().then(|| {
            writer
                .next_row_id()
                .expect("a v3 manifest list writer tracks row ids")
        });
        writer
            .add_manifests(std::iter::once(manifest))
            .map_err(Error::spec)?;
        first_row_ids.push(assigned);
    }
    let next_row_id = writer
        .next_row_id()
        .expect("a v3 manifest list writer tracks row ids");
    block_on(writer.close()).map_err(Error::spec)?;
    let bytes = read_back(&file_io, uri)?;
    Ok(WrittenManifestList {
        bytes,
        first_row_ids,
        next_row_id,
    })
}

/// The bytes a spec writer left at `uri` in the in-memory file IO.
fn read_back(file_io: &FileIO, uri: &str) -> Result<Vec<u8>> {
    let input = file_io.new_input(uri).map_err(Error::spec)?;
    Ok(block_on(input.read()).map_err(Error::spec)?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::data_files::ManifestOrigin;
    use object_storage::{FileRef, ObjectPath};

    fn entry(path: &str, manifest: &str) -> DataFileEntry {
        DataFileEntry {
            file: FileRef {
                path: ObjectPath::new(path),
                size: 1,
            },
            partition: None,
            stats: None,
            origin: Some(ManifestOrigin {
                snapshot_id: 1,
                sequence_number: 1,
                file_sequence_number: 1,
                first_row_id: Some(0),
                manifest_path: manifest.to_string(),
            }),
        }
    }

    fn manifest(path: &str, length: i64, files: &[&str]) -> Arc<LoadedManifest> {
        let mut file = iceberg::spec::ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: length,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 1,
            added_files_count: Some(files.len() as u32),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(0),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: Some(0),
        };
        file.manifest_length = length;
        Arc::new(LoadedManifest {
            file,
            entries: files.iter().map(|name| entry(name, path)).collect(),
        })
    }

    fn paths(entries: &[DataFileEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.file.path.as_str()).collect()
    }

    #[test]
    fn a_removal_rewrites_only_the_manifests_holding_a_removed_file() {
        let current = [
            manifest("m0", 10, &["a", "b"]),
            manifest("m1", 10, &["c"]),
            manifest("m2", 10, &[]),
        ];

        let plan =
            plan_manifests(&current, &[entry("a", "m0")], vec![entry("d", "")], 100).unwrap();

        let rewritten = plan.rewritten.unwrap();
        assert_eq!(
            rewritten
                .iter()
                .map(|write| (write.entry.file.path.as_str(), write.status))
                .collect::<Vec<_>>(),
            [
                ("a", ManifestStatus::Deleted),
                ("b", ManifestStatus::Existing)
            ]
        );
        assert_eq!(paths(&plan.added), ["d"]);
        assert_eq!(
            plan.carried
                .iter()
                .map(|file| file.manifest_path.as_str())
                .collect::<Vec<_>>(),
            ["m1"]
        );
        assert!(plan.merged.is_none());
    }

    #[test]
    fn the_smallest_manifests_past_the_threshold_are_merged() {
        let current = [
            manifest("m0", 30, &["a"]),
            manifest("m1", 10, &["b"]),
            manifest("m2", 20, &["c"]),
            manifest("m3", 40, &["d"]),
        ];

        let plan = plan_manifests(&current, &[], vec![], 3).unwrap();

        assert_eq!(paths(&plan.merged.unwrap()), ["b", "c"]);
        assert_eq!(
            plan.carried
                .iter()
                .map(|file| file.manifest_path.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m3"]
        );
    }

    #[test]
    fn removing_a_file_from_a_manifest_the_snapshot_no_longer_names_is_an_error() {
        let current = [manifest("m0", 10, &["a"])];

        let Err(error) = plan_manifests(&current, &[entry("z", "gone")], vec![], 100) else {
            panic!("a removal from an unknown manifest must be refused");
        };

        assert!(matches!(error, Error::UnknownManifest { .. }), "{error}");
    }

    #[test]
    fn removing_an_uncommitted_file_is_an_error() {
        let mut uncommitted = entry("a", "m0");
        uncommitted.origin = None;

        let Err(error) = plan_manifests(&[], &[uncommitted], vec![], 100) else {
            panic!("a removal of an uncommitted file must be refused");
        };

        assert!(matches!(error, Error::UncommittedRemoval { .. }), "{error}");
    }
}
