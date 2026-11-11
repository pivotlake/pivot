//! What an expiry leaves to delete: the files the snapshots it dropped
//! retired, the manifests and manifest lists only those snapshots name, the
//! statistics files dropped with them, and the previous metadata files it
//! dropped from the metadata log.
//!
//! # Linear history
//!
//! The table's history is taken to be linear, its current snapshot and
//! that snapshot's ancestors, of which an expiry drops the oldest: the
//! dropped snapshots are the nearest ancestors of the oldest snapshot kept.
//! Everything is found from the manifest lists of the dropped snapshots and
//! of the oldest snapshot kept, and from the manifests of theirs that
//! record a removal. No other kept snapshot is read and no live file's path
//! is held, so the cost follows what expired and not the size of the table.
//! Nothing else checks that the history is linear, so it is checked here: a
//! table with a branch or a tag besides `main`, or with a snapshot that is
//! not an ancestor of the current one, is refused, and so are dropped
//! snapshots that are not those ancestors.
//!
//! # What is deleted, by kind of object
//!
//! - A **manifest** that a dropped snapshot lists and the oldest kept
//!   snapshot does not. A snapshot lists manifests of its parent and
//!   manifests it writes under new names, so in a linear history a manifest
//!   that a snapshot stopped listing is listed by none after it: one the
//!   oldest kept snapshot lists is in use, and any other is not. Of the
//!   snapshots kept after the oldest, the current one is checked, against
//!   the manifest list the version already holds, with nothing read: a
//!   manifest to delete that it lists was listed again, and the whole
//!   collection is refused ([`Error::ExpiredManifestStillListed`]).
//! - A **data file** that an entry with status `DELETED` names, when the
//!   snapshot the entry is stamped with is a dropped one or the oldest kept
//!   one. A file leaves a table only by a commit that records its removal
//!   so, in a manifest the commit's snapshot lists, and a file removed at
//!   or before the oldest kept snapshot is live in no kept snapshot. The
//!   record is looked for in the manifests to delete and in those the
//!   oldest kept snapshot lists, and only in one that the removing snapshot
//!   wrote and whose manifest-list entry counts a removal or leaves the
//!   count out: a later rewrite of a manifest carries no earlier snapshot's
//!   removals over. A live entry of a manifest to delete is no evidence:
//!   its file was carried into another manifest, or removed by a commit
//!   that recorded it. A path is taken to name one file for the whole life
//!   of the table, and nothing checks that it does: a file added again
//!   under a path that one of those snapshots removed is deleted on the
//!   record of that removal.
//! - A **delete file** by the same rule as a data file, unless it is a
//!   deletion vector, which is never deleted. A deletion vector is one blob
//!   of a Puffin file that holds the vectors of every data file its commit
//!   touched, so the record of one vector's removal says nothing of the
//!   others, and no manifest read here says whether one of them is still
//!   live: the Puffin file is left in place, and no later expiry deletes
//!   it.
//! - The **manifest list** of each dropped snapshot.
//! - The **statistics files** and the **previous metadata files** the
//!   expiry dropped from the metadata document.
//!
//! An object the metadata document of the committed version itself names is
//! in use whatever the expiry dropped, and is never deleted.
//!
//! Deleting an object that is already gone is no error, so nothing is
//! remembered from one expiry to the next: what the oldest kept snapshot
//! removed is found again when that snapshot is itself dropped.

use std::collections::{HashMap, HashSet};

use iceberg::spec::{
    ManifestFile, ManifestList as IcebergManifestList, ManifestStatus, Snapshot, TableMetadata,
};
use object_storage::ObjectPath;

use crate::files::deletes::is_deletion_vector;
use crate::maintenance::linear_history::list_linear_history;
use crate::table::{MANIFESTS_PER_READ, Table};
use crate::{Error, Result};

/// The snapshots one committed expiry dropped, with the statistics files
/// and the previous metadata files it dropped beside them, by what the
/// metadata before it named for them and the metadata after it no longer
/// does. Only the commit that dropped them hands one out, so only the
/// process that committed an expiry deletes for it.
pub struct ExpiredSnapshots {
    /// The dropped snapshots, in no order.
    snapshots: Vec<ExpiredSnapshot>,
    /// The statistics and partition statistics files the expiry dropped
    /// from the metadata, by URI.
    statistics_files: Vec<String>,
    /// The previous metadata files the expiry dropped from the metadata
    /// log, by URI.
    metadata_files: Vec<String>,
}

/// What is kept of a snapshot an expiry dropped: where it stood in the
/// table's linear history, and its manifest list.
struct ExpiredSnapshot {
    snapshot_id: i64,
    parent_snapshot_id: Option<i64>,
    /// The snapshot's manifest list, by URI.
    manifest_list: String,
}

impl ExpiredSnapshots {
    /// What the metadata `before` an expiry names that the metadata `after`
    /// it does not: the snapshots dropped, the statistics files dropped
    /// with them, and the previous metadata files dropped from the log.
    pub(crate) fn between(before: &TableMetadata, after: &TableMetadata) -> Self {
        let snapshots = before
            .snapshots()
            .filter(|snapshot| after.snapshot_by_id(snapshot.snapshot_id()).is_none())
            .map(|snapshot| ExpiredSnapshot {
                snapshot_id: snapshot.snapshot_id(),
                parent_snapshot_id: snapshot.parent_snapshot_id(),
                manifest_list: snapshot.manifest_list().to_string(),
            })
            .collect();
        let retained_statistics: HashSet<&str> = list_statistics_files(after).collect();
        let statistics_files = list_statistics_files(before)
            .filter(|uri| !retained_statistics.contains(uri))
            .map(str::to_string)
            .collect();
        let retained_metadata_files: HashSet<&str> = list_logged_metadata_files(after).collect();
        let metadata_files = list_logged_metadata_files(before)
            .filter(|uri| !retained_metadata_files.contains(uri))
            .map(str::to_string)
            .collect();
        Self {
            snapshots,
            statistics_files,
            metadata_files,
        }
    }

    /// How many snapshots the expiry dropped.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// How many previous metadata files the expiry dropped from the log.
    pub fn metadata_file_count(&self) -> usize {
        self.metadata_files.len()
    }

    /// Whether the dropped snapshots are the nearest ancestors of
    /// `oldest_retained` in a linear history: its parent is one of them,
    /// and following parents from that one reaches every other, with no
    /// snapshot between them that is not one of them.
    fn are_nearest_ancestors_of(&self, oldest_retained: &Snapshot) -> bool {
        let mut parents_of_unreached: HashMap<i64, Option<i64>> = self
            .snapshots
            .iter()
            .map(|snapshot| (snapshot.snapshot_id, snapshot.parent_snapshot_id))
            .collect();
        let mut next_ancestor_id = oldest_retained.parent_snapshot_id();
        while let Some(parent_snapshot_id) =
            next_ancestor_id.and_then(|snapshot_id| parents_of_unreached.remove(&snapshot_id))
        {
            next_ancestor_id = parent_snapshot_id;
        }
        parents_of_unreached.is_empty()
    }
}

/// The previous metadata files the metadata log of `metadata` names, by URI.
fn list_logged_metadata_files(metadata: &TableMetadata) -> impl Iterator<Item = &str> {
    metadata
        .metadata_log()
        .iter()
        .map(|entry| entry.metadata_file.as_str())
}

/// The statistics and partition statistics files `metadata` names, by URI.
fn list_statistics_files(metadata: &TableMetadata) -> impl Iterator<Item = &str> {
    metadata
        .statistics_iter()
        .map(|file| file.statistics_path.as_str())
        .chain(
            metadata
                .partition_statistics_iter()
                .map(|file| file.statistics_path.as_str()),
        )
}

/// The file names of the objects the metadata document `metadata`, loaded
/// from `metadata_location`, itself names: its own file and those its
/// metadata log names, the manifest list of each of its snapshots, and its
/// statistics and partition statistics files. An object is told by its file
/// name, wherever it sits and however its URI is spelled: the names are
/// unique to the object.
fn list_metadata_object_names<'m>(
    metadata_location: &'m str,
    metadata: &'m TableMetadata,
) -> HashSet<&'m str> {
    std::iter::once(metadata_location)
        .chain(list_logged_metadata_files(metadata))
        .chain(list_statistics_files(metadata))
        .chain(
            metadata
                .snapshots()
                .map(|snapshot| snapshot.manifest_list()),
        )
        .map(object_name)
        .collect()
}

impl Table {
    /// The objects `expired`, the expiry that committed this version, leaves
    /// to delete, by the rules of this module: the data and delete files its
    /// snapshots retired first, then the manifests and the manifest lists
    /// only they name, then the statistics files and the metadata files it
    /// dropped. Each is named as the table names a file: relative to the
    /// table's directory when it lives under it, else by its
    /// store-root-absolute key.
    ///
    /// Read for it are the manifest list of the oldest snapshot this
    /// version keeps, the manifest list of each dropped snapshot, and, a
    /// few at a time, the manifests among theirs that record a removal by
    /// one of those snapshots. Any read that fails fails the whole
    /// collection, so nothing is deleted on a partial view.
    ///
    /// A table whose history is not linear is refused
    /// ([`Error::NonLinearHistory`]), with nothing read, and so is a
    /// version whose oldest snapshot is not the child of the newest dropped
    /// one, as a version other than the one the expiry committed may be.
    /// The collection is refused whole as well when a manifest it would
    /// name is one this version's current snapshot lists
    /// ([`Error::ExpiredManifestStillListed`]), which the manifest list the
    /// version holds tells with nothing read.
    pub fn collect_expired_objects(&self, expired: &ExpiredSnapshots) -> Result<Vec<ObjectPath>> {
        let table_name = self.name().to_string();
        let history = list_linear_history(&table_name, self.metadata())?;
        let location = self.location();
        let named_by_metadata =
            list_metadata_object_names(self.metadata_location(), self.metadata());

        let mut objects = Vec::new();
        let mut unused_manifests: HashMap<String, ManifestFile> = HashMap::new();
        let mut unused_manifest_lists = Vec::new();
        if !expired.snapshots.is_empty() {
            let oldest_retained = match history.first() {
                Some(oldest_retained) if expired.are_nearest_ancestors_of(oldest_retained) => {
                    oldest_retained
                }
                _ => {
                    return Err(Error::NonLinearHistory {
                        table: table_name,
                        reason: "the snapshots an expiry dropped are not the nearest ancestors of the oldest snapshot it kept".to_string(),
                    });
                }
            };
            // Read from the store even when it is the current snapshot's:
            // the list a version holds leaves out a delete manifest with no
            // file left to apply, which the snapshot still names.
            let oldest_retained_manifests =
                self.read_manifest_list(oldest_retained.manifest_list())?;
            let oldest_retained_manifest_names: HashSet<&str> = oldest_retained_manifests
                .iter()
                .map(|manifest| object_name(&manifest.manifest_path))
                .collect();
            for snapshot in &expired.snapshots {
                if named_by_metadata.contains(object_name(&snapshot.manifest_list)) {
                    continue;
                }
                for manifest in self.read_manifest_list(&snapshot.manifest_list)? {
                    if !oldest_retained_manifest_names
                        .contains(object_name(&manifest.manifest_path))
                    {
                        unused_manifests
                            .entry(manifest.manifest_path.clone())
                            .or_insert(manifest);
                    }
                }
                unused_manifest_lists.push(&snapshot.manifest_list);
            }
            // The manifests the current snapshot lists are in use whatever
            // the oldest kept snapshot lists, and this version holds them.
            let unused_names: HashSet<&str> = unused_manifests
                .keys()
                .map(|uri| object_name(uri.as_str()))
                .collect();
            let current_manifests = self.manifest_list();
            let still_listed = current_manifests
                .data_manifests()
                .chain(current_manifests.delete_manifests())
                .find(|manifest| unused_names.contains(object_name(&manifest.manifest_path)));
            if let Some(still_listed) = still_listed {
                return Err(Error::ExpiredManifestStillListed {
                    table: table_name,
                    manifest: still_listed.manifest_path.clone(),
                });
            }

            let mut removing_snapshot_ids: HashSet<i64> = expired
                .snapshots
                .iter()
                .map(|snapshot| snapshot.snapshot_id)
                .collect();
            removing_snapshot_ids.insert(oldest_retained.snapshot_id());
            let mut seen_paths: HashSet<&str> = HashSet::new();
            let manifests_with_removals: Vec<&ManifestFile> = unused_manifests
                .values()
                .chain(&oldest_retained_manifests)
                .filter(|manifest| {
                    manifest.has_deleted_files()
                        && removing_snapshot_ids.contains(&manifest.added_snapshot_id)
                })
                .filter(|manifest| seen_paths.insert(manifest.manifest_path.as_str()))
                .collect();
            objects =
                self.collect_removed_files(&manifests_with_removals, &removing_snapshot_ids)?;
        }

        let unused_statistics_files = expired
            .statistics_files
            .iter()
            .filter(|uri| !named_by_metadata.contains(object_name(uri)));
        let unused_metadata_files = expired
            .metadata_files
            .iter()
            .filter(|uri| !named_by_metadata.contains(object_name(uri)));
        for uri in unused_manifests
            .keys()
            .chain(unused_manifest_lists)
            .chain(unused_statistics_files)
            .chain(unused_metadata_files)
        {
            objects.push(location.file_path_for_uri(uri)?);
        }
        Ok(objects)
    }

    /// Every manifest the manifest list at `uri` names, the list read whole
    /// from the table's store.
    fn read_manifest_list(&self, uri: &str) -> Result<Vec<ManifestFile>> {
        let bytes = self.location().read_metadata_object(uri)?;
        let list =
            IcebergManifestList::parse_with_version(&bytes, self.metadata().format_version())
                .map_err(|source| Error::MalformedMetadataObject {
                    table: self.name().to_string(),
                    path: uri.to_string(),
                    source: Box::new(source),
                })?;
        Ok(list.consume_entries().into_iter().collect())
    }

    /// The files that the entries with status `DELETED` of `manifests`
    /// name, each once, when the snapshot the entry is stamped with is one
    /// of `removing_snapshot_ids`; a deletion vector is left out, since the
    /// Puffin file it is a blob of may hold vectors that are still live.
    /// `manifests`, each a distinct manifest, are read over the pool, a few
    /// at a time, and nothing else of them is kept.
    fn collect_removed_files(
        &self,
        manifests: &[&ManifestFile],
        removing_snapshot_ids: &HashSet<i64>,
    ) -> Result<Vec<ObjectPath>> {
        let location = self.location();
        let mut removed_files = Vec::new();
        let mut seen_files: HashSet<ObjectPath> = HashSet::new();
        for manifests in manifests.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().copied())?;
            for (listed, manifest) in manifests.iter().zip(&parsed) {
                for entry in manifest.entries() {
                    if entry.status() != ManifestStatus::Deleted {
                        continue;
                    }
                    // An entry that states no snapshot inherits that of its
                    // manifest.
                    let removing_snapshot_id =
                        entry.snapshot_id().unwrap_or(listed.added_snapshot_id);
                    if !removing_snapshot_ids.contains(&removing_snapshot_id)
                        || is_deletion_vector(entry.data_file())
                    {
                        continue;
                    }
                    let path = location.file_path_for_uri(entry.file_path())?;
                    if seen_files.insert(path.clone()) {
                        removed_files.push(path);
                    }
                }
            }
        }
        Ok(removed_files)
    }
}

/// The file name a metadata object's URI ends in.
fn object_name(uri: &str) -> &str {
    uri.rsplit('/').next().unwrap_or(uri)
}
