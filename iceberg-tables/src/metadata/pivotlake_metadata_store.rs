//! A table whose current version is kept in its own directory: numbered
//! metadata files under `<directory>/metadata/`, each version committed with a
//! conditional write that admits exactly one writer, so writers in several
//! processes share a table with no catalog service. Readable by any Iceberg
//! reader from the table's path.
//!
//! The metadata log of a version names every earlier metadata file except
//! the ones the commit that wrote it was told had expired
//! ([`Table::expired_metadata_files`](crate::Table::expired_metadata_files)):
//! `write.metadata.previous-versions-max` does not bound it, so a file leaves
//! the log, and is deleted, only once the file after it is older than the
//! retention window.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use iceberg::Runtime;
use iceberg::spec::TableMetadata;
use iceberg::table::Table as SpecTable;
use iceberg::{ErrorKind, TableCommit};
use object_storage::StoreError;

use crate::commit::{now_unix_ms, retry_io};
use crate::maintenance::retention::snapshot_retention;
use crate::metadata::layout::{
    METADATA_FILE_PREFIX, TableLocation, VERSION_HINT_FILE, hint_content, metadata_file_name,
    parse_hint, parse_metadata_file_version,
};
use crate::metadata::metadata_store::MetadataStore;
use crate::metadata::state::CurrentMetadata;
use crate::{Error, Result};

/// The numbered metadata files of the table at one location.
#[derive(Clone)]
pub struct PivotlakeMetadataStore {
    location: TableLocation,
    /// The latest version this store read or wrote, shared by its clones:
    /// where the listing for the table's latest version starts, and the
    /// metadata that answers when the listing ends there. Only ever replaced
    /// by a later version.
    latest_seen: Arc<Mutex<Option<SeenVersion>>>,
}

/// A metadata version as the store read or wrote it. A metadata file never
/// changes, so this stays true of its version.
#[derive(Clone)]
struct SeenVersion {
    version: u64,
    metadata: CurrentMetadata,
}

impl PivotlakeMetadataStore {
    pub fn new(location: TableLocation) -> Self {
        Self {
            location,
            latest_seen: Arc::default(),
        }
    }

    /// The version the table's hint names, or `None` when there is no hint,
    /// which only costs the reader a longer listing. A hint that does not
    /// parse is an error.
    pub fn read_version_hint(&self) -> Result<Option<u64>> {
        let location = &self.location;
        match location
            .store()
            .get(&location.metadata_key(VERSION_HINT_FILE))?
        {
            Some(bytes) => Ok(Some(parse_hint(&bytes)?)),
            None => {
                tracing::warn!(
                    table = %location.table_uri(),
                    "no version hint; listing the table's metadata from its first version"
                );
                Ok(None)
            }
        }
    }

    /// Write the version hint naming `version`. Best effort by the caller:
    /// the hint is for readers that open the table by its path and is never
    /// trusted by this crate, so a failed write costs nothing but a warning.
    pub fn write_version_hint(&self, version: u64) -> Result<()> {
        let location = &self.location;
        location
            .store()
            .put(
                &location.metadata_key(VERSION_HINT_FILE),
                hint_content(version).as_bytes(),
            )
            .map_err(Error::from)
    }

    /// The table's current version, its highest metadata file's, or `None`
    /// when it has no metadata file at all. The files are listed from the
    /// latest version this store has seen, or else from the version the
    /// hint names; when the table has none at or after that one, every
    /// metadata file is listed.
    fn find_current_version(&self) -> Result<Option<u64>> {
        // Copied out, so the lock is not held while the store is asked.
        let seen_version = self
            .latest_seen
            .lock()
            .unwrap()
            .as_ref()
            .map(|seen| seen.version);
        let start = match seen_version {
            Some(version) => Some(version),
            None => self.read_version_hint()?,
        };
        match (self.list_highest_version(start)?, start) {
            (None, Some(start)) => {
                tracing::warn!(
                    table = %self.location.table_uri(),
                    start,
                    "no metadata version at or after the one known exists; listing from the first version"
                );
                self.list_highest_version(None)
            }
            (found, _) => Ok(found),
        }
    }

    /// The highest metadata version listed from the version `start` up, or
    /// among every metadata file without one; `None` when the listing shows
    /// none.
    fn list_highest_version(&self, start: Option<u64>) -> Result<Option<u64>> {
        let location = &self.location;
        // The padded names sort in version order, so the name of the version
        // before the start is a cursor that admits the start itself and
        // everything newer.
        let after = start
            .and_then(|version| version.checked_sub(1))
            .map(metadata_file_name);
        let listed = location.store().list_filtered(
            &location.metadata_dir(),
            METADATA_FILE_PREFIX,
            after.as_deref(),
        )?;
        let mut versions: Vec<u64> = listed
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        Ok(versions.last().copied())
    }

    /// Read and parse metadata `version`.
    pub fn read_metadata(&self, version: u64) -> Result<CurrentMetadata> {
        let location = &self.location;
        let name = metadata_file_name(version);
        let bytes = location
            .store()
            .get(&location.metadata_key(&name))?
            .ok_or_else(|| Error::MetadataVersionMissing {
                table: location.table_uri(),
                version,
            })?;
        let metadata: TableMetadata = serde_json::from_slice(&bytes)?;
        Ok(CurrentMetadata {
            metadata_location: location.metadata_uri(&name),
            metadata: Arc::new(metadata),
        })
    }

    /// Remember `metadata` as version `version`, unless a later version is
    /// already remembered.
    fn remember_version(&self, version: u64, metadata: &CurrentMetadata) {
        let mut latest_seen = self.latest_seen.lock().unwrap();
        if latest_seen
            .as_ref()
            .is_none_or(|seen| seen.version < version)
        {
            *latest_seen = Some(SeenVersion {
                version,
                metadata: metadata.clone(),
            });
        }
    }

    /// Read metadata `version` and remember it.
    fn read_and_remember_metadata(&self, version: u64) -> Result<CurrentMetadata> {
        let metadata = self.read_metadata(version)?;
        self.remember_version(version, &metadata);
        Ok(metadata)
    }

    /// Whether `base` was committed at least its retention window ago: the
    /// age from which the metadata file after it may have been deleted,
    /// which takes the file after that one to be as old as the window.
    fn is_stale_base(base: &CurrentMetadata) -> bool {
        let retention_ms = snapshot_retention(&base.metadata)
            .map(|window| i64::try_from(window.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        base.metadata.last_updated_ms().saturating_add(retention_ms) <= now_unix_ms()
    }

    /// Make `metadata` the table's current version, as the successor of
    /// `base` (`None` for a table's first version): create
    /// `v{N+1}.metadata.json` only if nothing has that name yet.
    /// Every writer that loaded N derives the same name, so exactly one
    /// succeeds, and `None` tells the others they lost. The hint is rewritten
    /// afterwards, best effort.
    ///
    /// A write that fails without saying whether the store applied it leaves
    /// the outcome open: the object under the name is then read back, and is
    /// this commit's exactly when it holds these bytes. An outcome that
    /// cannot be settled is [`Error::CommitOutcomeUnknown`], never a loss.
    pub fn commit_metadata(
        &self,
        base: Option<&CurrentMetadata>,
        metadata: Arc<TableMetadata>,
    ) -> Result<Option<CurrentMetadata>> {
        let location = &self.location;
        let store = location.store();
        let version = match base {
            Some(base) => metadata_location_version(&base.metadata_location)? + 1,
            None => 1,
        };
        let metadata_bytes = serde_json::to_vec(metadata.as_ref())?;
        let metadata_name = metadata_file_name(version);
        let metadata_key = location.metadata_key(&metadata_name);
        // Whether some attempt at the write failed without saying whether
        // the store applied it.
        let mut is_put_ambiguous = false;
        let put = retry_io(
            || match store.put_if_absent(&metadata_key, &metadata_bytes) {
                Ok(()) => Ok(true),
                Err(StoreError::VersionConflict { .. }) => Ok(false),
                Err(error) => {
                    is_put_ambiguous = true;
                    Err(error)
                }
            },
        );
        let is_committed = match put {
            Ok(true) => true,
            Ok(false) if !is_put_ambiguous => false,
            // The name is taken after a write of ours may have been applied.
            Ok(false) => self.holds_committed_bytes(&metadata_key, &metadata_bytes, None)?,
            Err(error) => {
                self.holds_committed_bytes(&metadata_key, &metadata_bytes, Some(error))?
            }
        };
        if !is_committed {
            return Ok(None);
        }
        // A writer that loaded its base longer ago than the retention window
        // may be committing a version the table had and has since deleted: the
        // name was free only because the file under it expired. The table's
        // current file is never deleted, so a later version is then still
        // listed, and the commit, which no reader would see, withdraws
        // itself.
        if base.is_some_and(Self::is_stale_base) {
            // The version is written: a listing that fails leaves open
            // whether it is the table's current one.
            let listed = retry_io(|| {
                store.list_filtered(
                    &location.metadata_dir(),
                    METADATA_FILE_PREFIX,
                    Some(&metadata_name),
                )
            })
            .map_err(|error| Error::CommitOutcomeUnknown {
                source: Box::new(error),
            })?;
            // The version hint's name also starts with `v` and sorts after
            // every version's, so only a numbered metadata file is a
            // successor.
            let has_successor = listed
                .objects
                .iter()
                .any(|object| parse_metadata_file_version(object.file.path.as_str()).is_some());
            if has_successor {
                let _ = store.delete(&metadata_key);
                return Ok(None);
            }
        }
        if let Err(error) = self.write_version_hint(version) {
            tracing::warn!(
                table = %location.table_uri(),
                version,
                error = %error,
                "writing the version hint failed; path readers see the previous hint"
            );
        }
        let committed = CurrentMetadata {
            metadata_location: location.metadata_uri(&metadata_name),
            metadata,
        };
        self.remember_version(version, &committed);
        Ok(Some(committed))
    }
}

impl PivotlakeMetadataStore {
    /// Whether the object at `metadata_key` is the version this commit wrote
    /// as `metadata_bytes`, after a write of it that may or may not have been
    /// applied (`put_error` is the failure the write ended with, if it ended
    /// with one). Another writer's version under the name means this commit
    /// lost. No object, or a read that fails, settles nothing.
    fn holds_committed_bytes(
        &self,
        metadata_key: &object_storage::ObjectPath,
        metadata_bytes: &[u8],
        put_error: Option<Error>,
    ) -> Result<bool> {
        let unknown = |error: Error| Error::CommitOutcomeUnknown {
            source: Box::new(error),
        };
        let store = self.location.store();
        match retry_io(|| store.get(metadata_key)) {
            Ok(Some(published)) => Ok(published == metadata_bytes),
            Ok(None) => Err(unknown(put_error.unwrap_or_else(|| {
                Error::MetadataVersionMissing {
                    table: self.location.table_uri(),
                    version: parse_metadata_file_version(metadata_key.as_str()).unwrap_or(0),
                }
            }))),
            Err(read_error) => Err(unknown(put_error.unwrap_or(read_error))),
        }
    }
}

#[async_trait]
impl MetadataStore for PivotlakeMetadataStore {
    /// The metadata of the table's current version: the one this store
    /// remembers when that is still the current version, which costs one
    /// short listing and no read, else read and remembered.
    fn load_latest(&self) -> Result<CurrentMetadata> {
        let version = self
            .find_current_version()?
            .ok_or_else(|| Error::TableMetadataMissing {
                table: self.location.table_uri(),
            })?;
        // Copied out, so the lock is not held while the store is asked.
        let latest_seen = self.latest_seen.lock().unwrap().clone();
        match latest_seen {
            Some(seen) if seen.version == version => Ok(seen.metadata),
            _ => self.read_and_remember_metadata(version),
        }
    }

    /// Apply the commit to the base's metadata and publish the result as the
    /// base's successor with the one conditional write that settles whether
    /// the base is still current, or as the table's first version when the
    /// base was never published (a table being created). The successor's
    /// metadata log loses `expired_metadata_files` and nothing else, however
    /// many entries it holds: it keeps every other entry of the base's log
    /// and adds the base's file. A commit conflict means another writer
    /// published that version first.
    async fn commit_successor(
        &self,
        base: &SpecTable,
        mut commit: TableCommit,
        expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable> {
        let base_metadata = base.metadata_ref();
        let published_base = base.metadata_location().map(|location| CurrentMetadata {
            metadata_location: location.to_string(),
            metadata: Arc::clone(&base_metadata),
        });
        for requirement in commit.take_requirements() {
            requirement.check(Some(&base_metadata))?;
        }
        let mut builder = base_metadata.as_ref().clone().into_builder(
            published_base
                .as_ref()
                .map(|published| published.metadata_location.clone()),
        );
        for update in commit.take_updates() {
            builder = update.apply(builder)?;
        }
        let metadata = Arc::new(
            builder
                .retain_metadata_log_except(expired_metadata_files.iter().cloned())
                .build()?
                .metadata,
        );
        let metadata_store = self.clone();
        // The store's calls block on the network, and the runtime's workers
        // must not.
        let published = tokio::task::spawn_blocking(move || {
            metadata_store.commit_metadata(published_base.as_ref(), metadata)
        })
        .await
        // A commit that panicked may have written the version.
        .unwrap_or_else(|panicked| {
            Err(Error::CommitOutcomeUnknown {
                source: Box::new(Error::spec(
                    iceberg::Error::new(ErrorKind::Unexpected, "publishing a version panicked")
                        .with_source(panicked),
                )),
            })
        })
        // The error rides along as the source, which is how the
        // table tells an unknown outcome from a failure.
        .map_err(|error| {
            iceberg::Error::new(
                ErrorKind::Unexpected,
                format!("publishing a version: {error}"),
            )
            .with_source(error)
        })?;
        let Some(current) = published else {
            return Err(iceberg::Error::new(
                ErrorKind::CatalogCommitConflicts,
                "another writer published the table's next version first",
            ));
        };
        SpecTable::builder()
            .metadata(current.metadata)
            .metadata_location(current.metadata_location)
            .identifier(base.identifier().clone())
            .file_io(base.file_io().clone())
            .runtime(Runtime::try_current()?)
            .build()
    }
}

/// The version number a directory table's metadata location carries.
pub fn metadata_location_version(metadata_location: &str) -> Result<u64> {
    let name = metadata_location
        .rsplit('/')
        .next()
        .unwrap_or(metadata_location);
    parse_metadata_file_version(name).ok_or_else(|| Error::UnnumberedMetadataLocation {
        location: metadata_location.to_string(),
    })
}
