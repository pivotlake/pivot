//! A table whose current version is kept in its own directory: numbered
//! metadata files under `<directory>/metadata/`, each version landed with a
//! conditional write that admits exactly one writer, so writers in several
//! processes share a table with no catalog service. Readable by any Iceberg
//! reader from the table's path.

use std::sync::Arc;

use iceberg::spec::TableMetadata;
use object_storage::StoreError;

use crate::catalog::{CommitChanges, TableCatalog};
use crate::commit::{now_unix_ms, retry_io};
use crate::layout::{
    TableLocation, VERSION_HINT_FILE, hint_content, metadata_file_name, parse_hint,
    parse_metadata_file_version,
};
use crate::retention::snapshot_retention;
use crate::state::CurrentMetadata;
use crate::{Error, Result};

/// The numbered metadata files of the table at one location.
#[derive(Clone)]
pub struct DirectoryCatalog {
    location: TableLocation,
}

impl DirectoryCatalog {
    pub fn new(location: TableLocation) -> Self {
        Self { location }
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

    /// The table's current version: the highest metadata file listed from
    /// `from` up, or `None` when the table has no metadata file at all.
    /// `from` names a version known to have existed (the hint's, or the one
    /// a reader holds); a listing that no longer shows it means it was
    /// deleted, which the caller decides what to make of.
    pub fn find_current_version(&self, from: u64) -> Result<Option<u64>> {
        let location = &self.location;
        // The padded names sort in version order, so the name of the version
        // before `from` is a cursor that admits `from` itself and everything
        // newer.
        let after = match from {
            0 => String::new(),
            from => metadata_file_name(from - 1),
        };
        let listed = location
            .store()
            .list_from(&location.metadata_dir(), "v", &after)?;
        let mut versions: Vec<u64> = listed
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        if from > 0 && !versions.contains(&from) {
            return Err(Error::MetadataVersionMissing {
                table: location.table_uri(),
                version: from,
            });
        }
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

    /// The current version, found from the hint: a cold open.
    fn find_version_from_hint(&self) -> Result<Option<u64>> {
        let hinted = self.read_version_hint()?.unwrap_or(0);
        match self.find_current_version(hinted) {
            // The hinted version was deleted from under a hint a sweep had not
            // caught up with yet: the whole listing is still exact.
            Err(Error::MetadataVersionMissing { .. }) if hinted > 0 => {
                tracing::warn!(
                    table = %self.location.table_uri(),
                    hinted,
                    "the hinted metadata version no longer exists; listing from the first version"
                );
                self.find_current_version(0)
            }
            found => found,
        }
    }

    /// Whether `base` was committed longer ago than its retention window.
    fn is_stale_base(base: &CurrentMetadata) -> bool {
        let retention_ms = snapshot_retention(&base.metadata)
            .map(|window| i64::try_from(window.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        base.metadata.last_updated_ms().saturating_add(retention_ms) < now_unix_ms()
    }
}

impl TableCatalog for DirectoryCatalog {
    fn load(&self) -> Result<CurrentMetadata> {
        let version =
            self.find_version_from_hint()?
                .ok_or_else(|| Error::TableMetadataMissing {
                    table: self.location.table_uri(),
                    location: self.location.table_uri(),
                })?;
        self.read_metadata(version)
    }

    /// One short listing from the held version; a held version that was
    /// swept (this copy is older than the retention window) is reloaded cold.
    fn refresh(&self, held: &CurrentMetadata) -> Result<Option<CurrentMetadata>> {
        let held_version = metadata_location_version(&held.metadata_location)?;
        let version = match self.find_current_version(held_version) {
            Ok(Some(version)) => version,
            Ok(None) => {
                return Err(Error::TableMetadataMissing {
                    table: self.location.table_uri(),
                    location: self.location.table_uri(),
                });
            }
            Err(Error::MetadataVersionMissing { .. }) => return self.load().map(Some),
            Err(error) => return Err(error),
        };
        if version == held_version {
            return Ok(None);
        }
        self.read_metadata(version).map(Some)
    }

    /// Create `v{N+1}.metadata.json` only if nothing has that name yet.
    /// Every writer that loaded N derives the same name, so exactly one
    /// lands. The hint is rewritten afterwards, best effort.
    fn publish(
        &self,
        base: Option<&CurrentMetadata>,
        metadata: Arc<TableMetadata>,
        _changes: &CommitChanges,
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
        let landed = retry_io(
            || match store.put_if_absent(&metadata_key, &metadata_bytes) {
                Ok(()) => Ok(true),
                Err(StoreError::VersionConflict { .. }) => Ok(false),
                Err(error) => Err(error),
            },
        )?;
        if !landed {
            return Ok(None);
        }
        // A writer that loaded its base longer ago than the retention window
        // may be landing a version a sweep already deleted the successors
        // of: the name was free only because its file was swept. Such a
        // commit would be invisible, so it withdraws itself.
        if base.is_some_and(Self::is_stale_base) {
            let listed = store.list_from(&location.metadata_dir(), "v", &metadata_name)?;
            if !listed.objects.is_empty() {
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
        Ok(Some(CurrentMetadata {
            metadata_location: location.metadata_uri(&metadata_name),
            metadata,
        }))
    }

    fn expires_snapshots(&self) -> bool {
        true
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
