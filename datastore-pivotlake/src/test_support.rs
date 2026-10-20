//! Table-level test helpers layered on the reusable object-store harness:
//! committing to a table through its handle, and seeing or touching an
//! Iceberg table's metadata the way another process would.

pub use object_storage::test_support::*;

use std::sync::Arc;

use iceberg_tables::commit::{COMMIT_RETRIES, commit_file_changes, now_unix_ms};
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::directory::metadata_location_version;
use iceberg_tables::layout::{VERSION_HINT_FILE, parse_hint, parse_metadata_file_version};
use iceberg_tables::load::load_table;
use iceberg_tables::{DirectoryCatalog, TableLocation};
use object_storage::{FileRef, ObjectPath, ObjectStore, open_store};
use parquet_engine::FileStats;

use crate::{CatalogTable, Error, FileEntry, PartitionValues, Result};

impl CatalogTable {
    /// Write `bytes` as a new data file and commit it into the table.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<PartitionValues>,
    ) -> Result<()> {
        if self.file_refs().iter().any(|file| file.path == path) {
            return Ok(());
        }
        self.store()
            .put(&self.object_location().resolve(&path), bytes)?;
        let entry = FileEntry {
            file: FileRef {
                path,
                size: bytes.len() as u64,
            },
            partition,
            stats: None,
        };
        self.commit_entries(&[], &[entry], true)?;
        Ok(())
    }

    /// Atomically replace `removed` files with `added` files in one version.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileEntry],
    ) -> Result<()> {
        self.commit_entries(removed, added, false)
    }
}

/// The Iceberg table at `table_directory` under the database at `root_uri`,
/// as a process that opens it from its path sees it.
fn open_directory_table(root_uri: &str, table_directory: &str) -> Result<DirectoryTable> {
    let store: Arc<dyn ObjectStore> = open_store(root_uri)?.into();
    let location = TableLocation::new(store, ObjectPath::new(table_directory))?;
    let catalog = DirectoryCatalog::new(location.clone());
    Ok(DirectoryTable { location, catalog })
}

struct DirectoryTable {
    location: TableLocation,
    catalog: DirectoryCatalog,
}

/// Commit `bytes` as a new data file `file_name` of the Iceberg table at
/// `table_directory` under the database at `root_uri`, the way a writer in
/// another process would: load the table's current version from the store
/// and commit one version above it. `rows` is recorded as the file's row
/// count; no column bounds are recorded.
pub fn commit_external_append(
    root_uri: &str,
    table_directory: &str,
    file_name: &str,
    bytes: &[u8],
    rows: i64,
) -> Result<u64> {
    commit_from_outside(root_uri, table_directory, &[], file_name, bytes, rows, true)
}

/// Commit, as one version of the Iceberg table at `table_directory`, the
/// retirement of the live files named `removed` and the addition of `bytes`
/// as `file_name` holding `rows` rows: what a rewrite of the table's files
/// (compaction, a delete) commits. Returns the landed version.
pub fn commit_external_replace(
    root_uri: &str,
    table_directory: &str,
    removed: &[&str],
    file_name: &str,
    bytes: &[u8],
    rows: i64,
) -> Result<u64> {
    commit_from_outside(
        root_uri,
        table_directory,
        removed,
        file_name,
        bytes,
        rows,
        false,
    )
}

/// Commit, from outside the datastore, the retirement of the live files
/// named `removed` and the addition of `bytes` as `file_name` holding
/// `rows` rows, reloading and retrying when another writer lands first, as
/// a writer in another process does. Returns the landed version.
fn commit_from_outside(
    root_uri: &str,
    table_directory: &str,
    removed: &[&str],
    file_name: &str,
    bytes: &[u8],
    rows: i64,
    data_change: bool,
) -> Result<u64> {
    let table = open_directory_table(root_uri, table_directory)?;
    let path = ObjectPath::new(file_name);
    table
        .location
        .store()
        .put(&table.location.data_file_key(&path), bytes)?;
    for _ in 0..COMMIT_RETRIES {
        let state = load_table(&table.catalog, &table.location, table_directory)?;
        let removed_entries: Vec<DataFileEntry> = removed
            .iter()
            .map(|name| {
                state
                    .entries
                    .iter()
                    .find(|entry| entry.file.path.as_str() == *name)
                    .cloned()
                    .ok_or_else(|| Error::CommitConflict {
                        location: table_directory.to_string(),
                        file: name.to_string(),
                    })
            })
            .collect::<Result<_>>()?;
        let added = DataFileEntry {
            file: FileRef {
                path: path.clone(),
                size: bytes.len() as u64,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(rows),
                min_values: Default::default(),
                max_values: Default::default(),
                null_counts: Default::default(),
            })),
            origin: None,
        };
        if let Some(committed) = commit_file_changes(
            &state,
            &table.catalog,
            &removed_entries,
            vec![added],
            data_change,
            now_unix_ms(),
        )? {
            return Ok(metadata_location_version(
                &committed.current.metadata_location,
            )?);
        }
    }
    Err(iceberg_tables::Error::CommitRacesExhausted {
        attempts: COMMIT_RETRIES,
    }
    .into())
}

/// How many snapshots the current metadata of the Iceberg table at
/// `table_directory` retains.
pub fn retained_snapshots(root_uri: &str, table_directory: &str) -> Result<usize> {
    let table = open_directory_table(root_uri, table_directory)?;
    let state = load_table(&table.catalog, &table.location, table_directory)?;
    Ok(state.current.metadata.snapshots().len())
}

/// The metadata versions present under the Iceberg table at
/// `table_directory`, in ascending order.
pub fn metadata_versions(root_uri: &str, table_directory: &str) -> Result<Vec<u64>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let mut versions: Vec<u64> = table
        .location
        .store()
        .list(&table.location.metadata_dir())?
        .objects
        .iter()
        .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
        .collect();
    versions.sort_unstable();
    Ok(versions)
}

/// The version the Iceberg table's hint names.
pub fn hinted_version(root_uri: &str, table_directory: &str) -> Result<Option<u64>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let key = table.location.metadata_key(VERSION_HINT_FILE);
    Ok(table
        .location
        .store()
        .get(&key)?
        .map(|bytes| parse_hint(&bytes))
        .transpose()?)
}

/// The names of the live data files of the Iceberg table at
/// `table_directory`, as its current version names them, sorted.
pub fn live_data_file_names(root_uri: &str, table_directory: &str) -> Result<Vec<String>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let state = load_table(&table.catalog, &table.location, table_directory)?;
    let mut names: Vec<String> = state
        .entries
        .iter()
        .map(|entry| entry.file.path.to_string())
        .collect();
    names.sort();
    Ok(names)
}
