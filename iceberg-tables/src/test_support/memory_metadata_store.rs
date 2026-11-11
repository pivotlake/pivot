//! A table's metadata versions kept in memory: the metadata store of a
//! test that is about a table, whatever keeps its versions.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use iceberg::spec::{TableMetadata, TableMetadataBuilder};
use iceberg::table::Table as SpecTable;
use iceberg::{ErrorKind, Runtime, TableCommit};
use planner::catalog::SchemaQualifiedTableName;
use uuid::Uuid;

use crate::metadata::layout::TableLocation;
use crate::metadata::metadata_store::{MetadataFile, MetadataStore};
use crate::{Error, Result};

/// The metadata versions of one table, oldest first, the last one current.
/// A version becomes current only as the successor of the one that is, so
/// of several writers from one version exactly one succeeds. The table's
/// manifests and data files are in its location's store as ever; only the
/// metadata documents are held here.
pub struct MemoryMetadataStore {
    /// Where the table's objects are, which each version's metadata file is
    /// named under.
    location: TableLocation,
    versions: Mutex<Vec<MetadataFile>>,
}

impl MemoryMetadataStore {
    /// The store of a table at `location` that has no version yet.
    pub fn new(location: TableLocation) -> Arc<Self> {
        Arc::new(Self {
            location,
            versions: Mutex::default(),
        })
    }

    /// How many versions were made current.
    pub fn count_versions(&self) -> usize {
        self.versions.lock().unwrap().len()
    }

    /// Which version, counted from 1, the metadata file with the name
    /// `metadata_location` ends in is, or `None` when it is none of the
    /// table's.
    pub fn find_version_number(&self, metadata_location: &str) -> Option<u64> {
        let name = file_name(metadata_location);
        let versions = self.versions.lock().unwrap();
        let position = versions
            .iter()
            .position(|version| file_name(&version.location) == name)?;
        Some(position as u64 + 1)
    }

    /// Make the current metadata with `change` applied the table's next
    /// version, the way another tool alters a table.
    pub fn publish_changed(
        &self,
        change: impl FnOnce(TableMetadataBuilder) -> TableMetadataBuilder,
    ) -> MetadataFile {
        let current = self
            .versions
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("a table that is altered has a version");
        let builder =
            TableMetadata::clone(&current.metadata).into_builder(Some(current.location.clone()));
        let changed = change(builder).build().unwrap().metadata;
        self.publish_successor(Some(&current.location), changed)
            .expect("no other writer")
    }

    /// Make `metadata` current as the successor of the version at
    /// `base_location` (`None` for a table's first version); `None` when
    /// that version is not the current one.
    fn publish_successor(
        &self,
        base_location: Option<&str>,
        metadata: TableMetadata,
    ) -> Option<MetadataFile> {
        let mut versions = self.versions.lock().unwrap();
        let current_location = versions.last().map(|version| version.location.as_str());
        if current_location != base_location {
            return None;
        }
        let name = format!("{:05}-{}.metadata.json", versions.len(), Uuid::new_v4());
        let published = MetadataFile {
            location: self.location.metadata_uri(&name),
            metadata: Arc::new(metadata),
        };
        versions.push(published.clone());
        Some(published)
    }
}

/// The file name a metadata location ends in.
fn file_name(metadata_location: &str) -> &str {
    metadata_location
        .rsplit('/')
        .next()
        .unwrap_or(metadata_location)
}

#[async_trait]
impl MetadataStore for MemoryMetadataStore {
    fn load_latest(&self, _name: &SchemaQualifiedTableName) -> Result<MetadataFile> {
        self.versions
            .lock()
            .unwrap()
            .last()
            .cloned()
            .ok_or_else(|| Error::TableMetadataMissing {
                table: self.location.table_uri(),
            })
    }

    /// Apply the commit to the base's metadata and make the result current
    /// when the base still is, or the table's first version when the base
    /// was never published. The store names the metadata files, so the
    /// successor's metadata log drops `expired_metadata_files`.
    async fn commit_successor(
        &self,
        base: &SpecTable,
        mut commit: TableCommit,
        expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable> {
        let base_metadata = base.metadata_ref();
        let base_location = base.metadata_location();
        for requirement in commit.take_requirements() {
            requirement.check(Some(&base_metadata))?;
        }
        let mut builder = base_metadata
            .as_ref()
            .clone()
            .into_builder(base_location.map(str::to_string));
        for update in commit.take_updates() {
            builder = update.apply(builder)?;
        }
        let metadata = builder
            .retain_metadata_log_except(expired_metadata_files.iter().cloned())
            .build()?
            .metadata;
        let Some(published) = self.publish_successor(base_location, metadata) else {
            return Err(iceberg::Error::new(
                ErrorKind::CatalogCommitConflicts,
                "another writer published the table's next version first",
            ));
        };
        SpecTable::builder()
            .metadata(published.metadata)
            .metadata_location(published.location)
            .identifier(base.identifier().clone())
            .file_io(base.file_io().clone())
            .runtime(Runtime::try_current()?)
            .build()
    }
}
