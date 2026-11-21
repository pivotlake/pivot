//! Committing to a table: the spec crate's transaction builds the new
//! snapshot's manifests and manifest list and the changes to the metadata
//! document, and the table's [`MetadataStore`]
//! makes the result current.
//!
//! A commit from one version writes its Avro objects under new names, then
//! asks the store to make the new metadata the successor of the version
//! it was built on. Exactly one of several writers from the same base
//! succeeds; the rest learn they lost, the transaction deleting the objects
//! each of them wrote, reload, and try again on top of the winner.
//!
//! A commit of files drops no snapshot and no entry of the metadata log.
//! Snapshots and previous metadata files are expired by a commit of its own
//! ([`Table::expire_snapshots`]), which hands back what it dropped so the
//! files those snapshots retired, the manifests and manifest lists only
//! they name, and the metadata files can be deleted: finding them reads
//! manifests, which no INSERT should wait for.

use std::collections::HashMap;
use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use iceberg::ErrorKind;
use iceberg::Runtime;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, TableMetadata,
    TableMetadataBuilder,
};
use iceberg::table::Table as SpecTable;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use object_storage::ObjectPath;
use planner::catalog::{Column, SchemaQualifiedTableName};
use uuid::Uuid;

use crate::files::data_files::DataFileEntry;
use crate::files::manifests::ManifestList;
use crate::files::partition::Partition;
use crate::io::runtime::block_on;
use crate::maintenance::expired_objects::ExpiredSnapshots;
use crate::maintenance::retention::{DEFAULT_SNAPSHOT_RETENTION, SNAPSHOT_RETENTION_PROPERTY};
use crate::metadata::layout::TableLocation;
use crate::metadata::metadata_store::{
    CommitCatalog, MetadataFile, MetadataStore, build_table_ident,
};
use crate::table::{FoundFile, Table};
use crate::types::schema::{
    LOGICAL_TYPES_PROPERTY, NAME_MAPPING_PROPERTY, build_partition_spec, build_schema,
    build_sort_order, logical_types_property, name_mapping_property,
};
use crate::{Error, Result};

/// How many times a writer that lost the race rebuilds its commit on the
/// winner before giving up, and how many times a store that writes a
/// table's metadata itself retries a write that failed with an IO error.
pub const COMMIT_RETRIES: usize = 10;

/// What `CREATE TABLE` asks a new table to be.
pub struct TableCreation<'a> {
    /// The table's identity across every metadata version.
    pub table_uuid: Uuid,
    pub columns: &'a [Column],
    /// The terms the table is partitioned by, in order, each a column or a
    /// transform of one as `CREATE TABLE` spells it (`day(ts)`).
    pub partition_by: &'a [String],
    pub sort_by: &'a [String],
    /// Files the table adopts as its initial data, each with its statistics
    /// and, for a partitioned table, its constant partition tuple.
    pub adopted: Vec<DataFileEntry>,
}

impl TableCreation<'_> {
    /// The metadata document of a new table at `location` with the shape
    /// asked for, before any version of it is published: format
    /// version 3, every field optional, the table's retention window set,
    /// and the Pivot types Iceberg has no spelling for recorded.
    pub(crate) fn build_metadata(&self, location: &TableLocation) -> Result<TableMetadata> {
        let schema = build_schema(self.columns)?;
        let partition_spec = build_partition_spec(&schema, self.partition_by)?;
        let sort_order = build_sort_order(&schema, self.sort_by)?;
        let built = TableMetadataBuilder::new(
            schema,
            partition_spec,
            sort_order,
            location.table_uri(),
            FormatVersion::V3,
            HashMap::new(),
        )
        .map_err(Error::spec)?
        .assign_uuid(self.table_uuid)
        .build()
        .map_err(Error::spec)?
        .metadata;
        // The properties key on field ids, so they are taken from the schema as
        // built, after the builder's own id assignment.
        let mut properties = HashMap::from([(
            SNAPSHOT_RETENTION_PROPERTY.to_string(),
            DEFAULT_SNAPSHOT_RETENTION.as_millis().to_string(),
        )]);
        if let Some(logical_types) = logical_types_property(built.current_schema(), self.columns) {
            properties.insert(LOGICAL_TYPES_PROPERTY.to_string(), logical_types);
        }
        if !self.adopted.is_empty() {
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
}

impl Table {
    /// Create the table at `location`: make its first version current
    /// through `metadata_store`, holding the adopted files as the first
    /// snapshot when there are any. The first version is committed as any
    /// other is, as the successor of a document, here one that was never
    /// published, so a store that already holds a version of the table
    /// refuses it: a table that exists is not this one's to take over.
    pub fn create(
        dispatcher: &DataFlowDispatcher,
        metadata_store: Arc<dyn MetadataStore>,
        location: TableLocation,
        table_name: &SchemaQualifiedTableName,
        creation: TableCreation<'_>,
    ) -> Result<Table> {
        let metadata = Arc::new(creation.build_metadata(&location)?);
        let unpublished = Table::from_manifest_list(
            dispatcher,
            metadata_store,
            location,
            table_name,
            Arc::new(MetadataFile {
                location: String::new(),
                metadata: Arc::clone(&metadata),
            }),
            ManifestList::empty(&metadata),
            None,
        )?;
        let created = if creation.adopted.is_empty() {
            unpublished.commit_transaction(Vec::new(), |transaction| {
                // A transaction with no action commits nothing, so the one
                // that publishes a table without files states the
                // document's properties again, which changes none.
                metadata
                    .properties()
                    .iter()
                    .fold(
                        transaction.update_table_properties(),
                        |properties, (key, value)| properties.set(key.clone(), value.clone()),
                    )
                    .apply(transaction)
            })?
        } else {
            unpublished.commit(&[], &[], creation.adopted)?
        };
        created.ok_or_else(|| Error::TableAlreadyInitialized {
            table: table_name.to_string(),
            location: unpublished.location().table_uri(),
        })
    }

    /// Commit, on top of this version, the removal of the live files at
    /// `removed_paths`, files of the partition with the key
    /// `removed_partition`, and the addition of `added`, as one new
    /// version: an append when nothing is removed, which names no
    /// partition, else a replace, which leaves the table's rows as they
    /// were. `None` means another writer committed a version first: the
    /// caller refreshes and retries.
    ///
    /// The files to remove are looked up once, by their paths, among the
    /// partition's: one this version has no live file at is refused
    /// ([`Error::RemovedFileNotLive`]). Every file added is written under
    /// the table's default partition spec. A replace is refused when the
    /// files it removes were written under another spec: what replaces a
    /// file takes that file's partition values, which only the fields of
    /// the default spec can hold.
    pub fn commit(
        &self,
        removed_partition: &[u8],
        removed_paths: &[ObjectPath],
        added: Vec<DataFileEntry>,
    ) -> Result<Option<Table>> {
        let removed = self.find_removed_files(removed_partition, removed_paths)?;
        // An append lists the delete manifests as they are. A replace is
        // refused: compacting a table that carries deletion vectors is not
        // supported yet (https://github.com/pivotlake/pivot/issues/95).
        if !removed_paths.is_empty()
            && let Some(manifest) = self.manifest_list().delete_manifests().next()
        {
            return Err(Error::DeleteFiles {
                table: self.name().to_string(),
                manifest: manifest.manifest_path.clone(),
            });
        }
        // A version 1 table's manifests have no sequence numbers, and Pivot
        // commits to none.
        let format_version = self.metadata().format_version();
        if format_version < FormatVersion::V2 {
            return Err(Error::UnsupportedFormatVersion {
                table: self.name().to_string(),
                version: format_version as u8,
            });
        }
        let default_spec_id = self.metadata().default_partition_spec_id();
        if let Some((partition, files)) = &removed
            && partition.spec_id() != default_spec_id
        {
            return Err(Error::ReplacedFileOfAnotherPartitionSpec {
                table: self.name().to_string(),
                file: files[0].file.path.to_string(),
                spec_id: partition.spec_id(),
                default_spec_id,
            });
        }
        let removed = match removed {
            Some((partition, files)) => self.name_removed_files(&partition, &files)?,
            None => Vec::new(),
        };
        let added = added
            .iter()
            .map(|entry| entry.create_data_file(self.metadata(), self.location()))
            .collect::<Result<Vec<DataFile>>>()?;
        self.commit_transaction(Vec::new(), move |transaction| {
            // The files were just written under fresh names, so none is in
            // the table already; the check for one would read every manifest.
            // Small manifests are merged as they pile up, so the table's
            // manifests grow with its files and not with its commits.
            if removed.is_empty() {
                return transaction
                    .fast_append()
                    .with_check_duplicate(false)
                    .with_manifest_merge(true)
                    .add_data_files(added)
                    .apply(transaction);
            }
            let mut rewrite = transaction
                .rewrite_files()
                .with_check_duplicate(false)
                .with_manifest_merge(true);
            for file in removed {
                rewrite = rewrite.delete_file(file);
            }
            for file in added {
                rewrite = rewrite.add_file(file);
            }
            rewrite.apply(transaction)
        })
    }

    /// The live files at `paths` in the partition with the key
    /// `partition`, which a commit is to remove, beside that partition;
    /// `None` when there are none to remove. A path this version has no
    /// live file of the partition at is refused.
    fn find_removed_files(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> Result<Option<(Arc<Partition>, Vec<FoundFile>)>> {
        if paths.is_empty() {
            return Ok(None);
        }
        let partition = self.find_partition(partition)?;
        let found = self.find_files(&partition, paths)?;
        let files = paths.iter().zip(found).map(|(path, found)| {
            found.ok_or_else(|| Error::RemovedFileNotLive {
                table: self.name().to_string(),
                file: path.to_string(),
            })
        });
        Ok(Some((
            partition,
            files.collect::<Result<Vec<FoundFile>>>()?,
        )))
    }

    /// Each of `files`, live files of `partition`, as the transaction of a
    /// commit that removes it is handed it: named by its URI, with its
    /// partition's tuple, its size and its row count. The transaction
    /// finds the file's manifest entry by that name, and records what the
    /// entry holds.
    fn name_removed_files(
        &self,
        partition: &Partition,
        files: &[FoundFile],
    ) -> Result<Vec<DataFile>> {
        let name_file = |found: &FoundFile| {
            let path = &found.file.path;
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(self.location().uri_for(path)?)
                .file_format(DataFileFormat::Parquet)
                .partition(partition.tuple().clone())
                .partition_spec_id(partition.spec_id())
                .record_count(found.record_count)
                .file_size_in_bytes(found.file.size)
                .build()
                .map_err(|error| Error::InvalidDataFile {
                    file: path.to_string(),
                    message: error.to_string(),
                })
        };
        files.iter().map(name_file).collect()
    }

    /// Whether a snapshot or a previous metadata file of this version has
    /// outlived the table's retention window as of `now_ms`: whether
    /// [`expire_snapshots`](Self::expire_snapshots) has anything to commit.
    /// Read from the metadata held, with nothing read from the store. A
    /// table whose history is not linear is refused
    /// ([`Error::NonLinearHistory`]).
    pub fn has_expired_history(&self, now_ms: i64) -> Result<bool> {
        Ok(!self.expired_snapshot_ids(now_ms)?.is_empty()
            || !self.expired_metadata_files(now_ms)?.is_empty())
    }

    /// Drop every snapshot that has outlived the table's retention window as
    /// of `now_ms`, and from the metadata log every previous metadata file
    /// that has, as one metadata-only commit with no change to the current
    /// snapshot. `None` when nothing has expired
    /// ([`has_expired_history`](Self::has_expired_history)), or when another
    /// writer committed the next version first: the caller refreshes and
    /// looks again.
    ///
    /// A table whose history is not linear is refused
    /// ([`Error::NonLinearHistory`]) before anything is committed, so no
    /// snapshot is dropped whose retired files could then not be found
    /// ([`collect_expired_objects`](Self::collect_expired_objects)).
    ///
    /// The expiry handed back is of a commit that succeeded. An error of
    /// the commit leaves open whether it did ([`Error::may_have_committed`])
    /// and hands back nothing to delete by.
    pub fn expire_snapshots(&self, now_ms: i64) -> Result<Option<SnapshotExpiry>> {
        let expired_snapshot_ids = self.expired_snapshot_ids(now_ms)?;
        let expired_metadata_files = self.expired_metadata_files(now_ms)?;
        if expired_snapshot_ids.is_empty() && expired_metadata_files.is_empty() {
            return Ok(None);
        }
        let committed = self.commit_transaction(expired_metadata_files, |transaction| {
            // With no snapshot to remove this still commits a version, the
            // one that drops expired metadata files from the log. Which
            // snapshots expire is decided by `Table::expired_snapshot_ids`,
            // so the spec crate's own selection by age is switched off with
            // a cutoff nothing is older than.
            transaction
                .expire_snapshots()
                .expire_snapshot_ids(expired_snapshot_ids)
                .expire_older_than_ms(i64::MIN)
                .apply(transaction)
        })?;
        let Some(table) = committed else {
            return Ok(None);
        };
        let expired = ExpiredSnapshots::between(self.metadata(), table.metadata());
        Ok(Some(SnapshotExpiry { table, expired }))
    }

    /// Commit the transaction `build_transaction` builds on this version
    /// through the table's metadata store, the new version's metadata log
    /// dropping `expired_metadata_files`. `Ok(None)` when another writer
    /// committed first. What the transaction reads of this version is read
    /// whole from the table's store, as often as it asks: the manifest
    /// list, twice by a commit that removes files, every data manifest by
    /// such a commit, and the manifests a merge joins. A commit that
    /// succeeded is loaded at its version, its manifest list read from the
    /// table's store ([`load_committed`](Self::load_committed)). When that
    /// load fails the commit stands all the same: the error says so
    /// ([`Error::CommittedTableUnloaded`]), and nothing the commit wrote or
    /// added is deleted for it.
    ///
    /// The transaction deletes the objects it wrote that nothing names. Of
    /// a commit known not to have been applied, those are every manifest
    /// and the manifest list: it failed before the metadata store was
    /// asked, or the store answered a commit conflict, which is how a
    /// writer learns it lost. Of a commit that succeeded, they are the
    /// manifests it wrote and then merged into another, which its manifest
    /// list does not name. After any other error of the store the
    /// transaction keeps them all, since the store may have applied the
    /// commit. The error says so when the store left the outcome unknown
    /// ([`Error::may_have_committed`]); when the store refused the commit
    /// outright, as a catalog does for a table that was dropped, they stay
    /// in the table's directory, named by no metadata.
    fn commit_transaction(
        &self,
        expired_metadata_files: Vec<String>,
        build_transaction: impl FnOnce(Transaction) -> iceberg::Result<Transaction>,
    ) -> Result<Option<Table>> {
        let file_io = self.location().build_file_io();
        let base = SpecTable::builder()
            .metadata(Arc::clone(self.metadata()))
            .identifier(build_table_ident(self.name()))
            .file_io(file_io)
            // A commit reads the manifests it rewrites once.
            .disable_cache();
        // A table being created has no published document to succeed.
        let base = match self.metadata_location() {
            "" => base,
            location => base.metadata_location(location),
        };
        let committed = block_on(async {
            let base = base
                .runtime(Runtime::try_current().map_err(Error::spec)?)
                .build()
                .map_err(Error::spec)?;
            let transaction = build_transaction(Transaction::new(&base)).map_err(Error::spec)?;
            let catalog = CommitCatalog {
                metadata_store: Arc::clone(self.metadata_store()),
                base,
                expired_metadata_files,
            };
            transaction
                .commit(&catalog)
                .await
                .map_err(|error| Error::Catalog(Box::new(error)))
        });
        let committed = match committed {
            Ok(committed) => committed,
            Err(Error::Catalog(error)) if error.kind() == ErrorKind::CatalogCommitConflicts => {
                return Ok(None);
            }
            Err(Error::Catalog(error)) if is_outcome_unknown(&error) => {
                return Err(Error::CommitOutcomeUnknown {
                    source: Box::new(Error::Catalog(error)),
                });
            }
            Err(error) => return Err(error),
        };
        self.load_committed(&committed)
            .map(Some)
            .map_err(|error| Error::CommittedTableUnloaded {
                table: self.name().to_string(),
                source: Box::new(error),
            })
    }

    /// The table at the version `committed`, which a commit on top of this
    /// version just made current, derived from this version. Its manifest
    /// list is this version's when the commit added no snapshot; the one a
    /// snapshot it added names is read whole from the table's store, which
    /// tells its length. No manifest is read: the parts of the manifests
    /// the commit wrote are read when something first asks for them.
    fn load_committed(&self, committed: &SpecTable) -> Result<Table> {
        let metadata_file = MetadataFile::try_from(committed)?;
        let metadata = &metadata_file.metadata;
        let held_manifest_list = self
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list());
        let (manifest_list, manifest_list_size) = match metadata.current_snapshot() {
            None => (ManifestList::empty(metadata), None),
            Some(snapshot) if Some(snapshot.manifest_list()) == held_manifest_list => {
                (self.manifest_list().clone(), self.manifest_list_size())
            }
            Some(snapshot) => {
                let bytes = self
                    .location()
                    .read_metadata_object(snapshot.manifest_list())?;
                (
                    ManifestList::parse(&self.name().to_string(), metadata, &bytes)?,
                    Some(bytes.len() as u64),
                )
            }
        };
        Table::from_manifest_list(
            self.dispatcher(),
            Arc::clone(self.metadata_store()),
            self.location().clone(),
            self.name(),
            Arc::new(metadata_file),
            manifest_list,
            manifest_list_size,
        )
    }
}

/// Whether `error`, a failed commit's, comes from a metadata commit whose
/// outcome is unknown: the catalog passes that commit's own error along as
/// the source.
fn is_outcome_unknown(error: &iceberg::Error) -> bool {
    std::error::Error::source(error)
        .and_then(|source| source.downcast_ref::<Error>())
        .is_some_and(Error::may_have_committed)
}

/// An expiry this process committed: the version it produced and the snapshots
/// and previous metadata files it dropped, which
/// [`Table::collect_expired_objects`] of that version turns into the objects
/// to delete.
pub struct SnapshotExpiry {
    pub table: Table,
    pub expired: ExpiredSnapshots,
}
