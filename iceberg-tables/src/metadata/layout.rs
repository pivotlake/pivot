//! Where a table's objects live and how they are named: its metadata
//! objects, and the full URIs manifests record data files by.

use std::sync::Arc;

use object_storage::{DataFile, FileRef, ObjectPath, ObjectStore, StoreScheme, local_path};
use planner::catalog::SchemaQualifiedTableName;

use crate::metadata::metadata_store::MetadataFile;
use crate::{Error, Result};

/// The directory under a table's own holding its metadata objects, unless
/// the table names another.
pub const METADATA_DIR: &str = "metadata";
/// The directory under a table's own that a catalog table's data files are
/// written to, unless the table names another: the layout every other
/// writer uses.
pub const DATA_DIR: &str = "data";
/// The table property naming where data files are written, when not under
/// the table's own `data/`.
pub const DATA_PATH_PROPERTY: &str = "write.data.path";
/// The table property naming where metadata objects are written, when not
/// under the table's own `metadata/`.
pub const METADATA_PATH_PROPERTY: &str = "write.metadata.path";

/// Split a file location into its root, as the location spells it
/// (`s3a://lake/`, `file:///`, `/`), and the key of the object under it: a
/// bucket root for `s3://`, `s3a://` and `gs://` locations, the filesystem
/// root for a local path. Two roots that spell one place differently are
/// told apart by [`is_same_root`], not by comparing them.
pub fn split_location(location: &str) -> Result<(String, ObjectPath)> {
    let unsupported = |message: String| Error::UnsupportedLocation {
        path: location.to_string(),
        message,
    };
    let scheme = StoreScheme::from_uri(location).map_err(|error| unsupported(error.to_string()))?;
    let key = if scheme == StoreScheme::Local {
        let path = local_path(location);
        if !path.starts_with('/') {
            return Err(unsupported(
                "a local location must be an absolute path".to_string(),
            ));
        }
        path.trim_start_matches('/')
    } else {
        let (_, rest) = location
            .split_once("://")
            .expect("a bucket location carries a scheme");
        match rest.split_once('/') {
            Some((bucket, key)) if !bucket.is_empty() && !key.is_empty() => key,
            _ => {
                return Err(unsupported(
                    "expected a bucket and an object key".to_string(),
                ));
            }
        }
    };
    let root = location[..location.len() - key.len()].to_string();
    Ok((root, ObjectPath::new(key)))
}

/// Whether two roots, as [`split_location`] names them, are one place: the
/// same bucket on the same backend, or both the local filesystem, however
/// each is spelled (`s3://` and `s3a://` name one bucket, `file:///` and `/`
/// one filesystem). Spark spells a bucket `s3a://` and Trino `s3://`, and a
/// table both have written carries both.
pub fn is_same_root(first: &str, second: &str) -> bool {
    fn bucket(root: &str) -> &str {
        root.split_once("://")
            .map_or("", |(_, rest)| rest)
            .trim_end_matches('/')
    }
    match (StoreScheme::from_uri(first), StoreScheme::from_uri(second)) {
        (Ok(StoreScheme::Local), Ok(StoreScheme::Local)) => true,
        (Ok(first_scheme), Ok(second_scheme)) => {
            first_scheme == second_scheme && bucket(first) == bucket(second)
        }
        _ => false,
    }
}

/// Where one table lives: the store its files are read and written
/// through, its directory under it, and the root every URI the table
/// records is written against. Iceberg names files by full URI, so a key
/// under the store and its URI must convert both ways exactly: a data file
/// written as `pivot-….parquet` under the table reads back as that same
/// relative path, and a file the table adopted from elsewhere in the store
/// as the same store-root-absolute key it was adopted by. Every read goes
/// through it, so a file under any other root is refused before it is
/// read: a table's files all live under one root, and a table deliberately
/// split across buckets is not served rather than read through the wrong
/// store.
#[derive(Clone)]
pub struct TableLocation {
    /// The name the table is known by in its datastore, for its errors.
    table: SchemaQualifiedTableName,
    store: Arc<dyn ObjectStore>,
    /// The table's directory, relative to the store's prefix (or absolute
    /// from the store root).
    pub(crate) directory: ObjectPath,
    /// Where the table's metadata objects go: its own `metadata/` unless the
    /// table names another directory.
    metadata_dir: ObjectPath,
    /// What a store-root-absolute key is appended to for a URI: the root as
    /// the table spells it (`s3a://lake/`, `file:///`), so a file the table
    /// records is named the way its other files are. A URI the table names is
    /// checked against it by [`is_same_root`], whatever its spelling.
    root_uri: String,
    /// The table's directory as a store-root-absolute key, which relative data
    /// file paths are resolved against and URIs are stripped of.
    absolute_directory: String,
}

impl TableLocation {
    /// The table `table` at `directory` of a datastore's own `store`, which
    /// every file of the table lives in, keeping its metadata under its own
    /// `metadata/`. Its URIs are written against the store's root (the
    /// bucket, or the filesystem root), whatever prefix below it the store
    /// itself is opened on.
    pub fn in_datastore_store(
        table: &SchemaQualifiedTableName,
        store: Arc<dyn ObjectStore>,
        directory: ObjectPath,
    ) -> Result<Self> {
        let metadata_dir = directory.join(METADATA_DIR);
        let root_uri = store_root_uri(store.as_ref())?;
        Self::at(table, store, root_uri, directory, metadata_dir)
    }

    /// The catalog table `table` at the version `metadata_file`, read
    /// through `store`, opened on the root of `metadata_file`'s location: its
    /// directory is the one its metadata names, its metadata objects go
    /// under `write.metadata.path` when the table sets it, else under the
    /// directory's `metadata/`, and its URIs are spelled as its own
    /// metadata location is.
    pub fn of_catalog_table(
        table: &SchemaQualifiedTableName,
        store: Arc<dyn ObjectStore>,
        metadata_file: &MetadataFile,
    ) -> Result<Self> {
        let (root_uri, _) = split_location(&metadata_file.location)?;
        let metadata = &metadata_file.metadata;
        let directory = key_under_root(table, &root_uri, metadata.location())?;
        let metadata_dir = match metadata.properties().get(METADATA_PATH_PROPERTY) {
            Some(path) => key_under_root(table, &root_uri, path.trim_end_matches('/'))?,
            None => directory.join(METADATA_DIR),
        };
        Self::at(table, store, root_uri, directory, metadata_dir)
    }

    /// The table at `directory` keeping its metadata objects under
    /// `metadata_dir` (a key under the store, relative to its prefix or
    /// absolute from its root), its URIs written against `root_uri`: the root
    /// as the table's own metadata spells it.
    fn at(
        table: &SchemaQualifiedTableName,
        store: Arc<dyn ObjectStore>,
        root_uri: String,
        directory: ObjectPath,
        metadata_dir: ObjectPath,
    ) -> Result<Self> {
        let absolute_directory = find_absolute_directory(store.as_ref(), &directory)?;
        Ok(Self {
            table: table.clone(),
            store,
            directory,
            metadata_dir,
            root_uri,
            absolute_directory,
        })
    }

    /// The store itself, for writing under the table's root with the same
    /// credentials the reads use.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The key under the store of the object at `uri`, refusing a URI
    /// under any other root.
    pub fn find_key_for_uri(&self, uri: &str) -> Result<ObjectPath> {
        let key = key_under_root(&self.table, &self.root_uri, uri)?;
        Ok(address_from_store_root(key.as_str()))
    }

    /// The file at `uri`, of `size` bytes, located for reading. Its
    /// identity is the full URI, as the table's metadata names it.
    pub fn locate_file_at_uri(&self, uri: &str, size: u64) -> Result<DataFile> {
        let source = self.store.source(&self.find_key_for_uri(uri)?)?;
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(uri),
                size,
            },
            source,
        })
    }

    /// The directory the metadata objects live in.
    pub fn metadata_dir(&self) -> ObjectPath {
        self.metadata_dir.clone()
    }

    /// The key of the metadata object `name`.
    pub fn metadata_key(&self, name: &str) -> ObjectPath {
        self.metadata_dir.join(name)
    }

    /// The key of an object the table names by `path` (relative to the
    /// table, or store-root-absolute for an adopted file).
    pub fn resolve_table_path(&self, path: &ObjectPath) -> ObjectPath {
        self.directory.resolve(path)
    }

    /// The bytes of the object at `uri`, which must be in the table's
    /// store.
    pub fn read_object_at_uri(&self, uri: &str) -> Result<Vec<u8>> {
        let path = self.find_path_for_uri(uri)?;
        self.store
            .get(&self.resolve_table_path(&path))?
            .ok_or_else(|| Error::MissingObject {
                table: self.table.to_string(),
                path: uri.to_string(),
            })
    }

    /// The table's own URI, recorded as the metadata's `location`.
    pub fn table_uri(&self) -> String {
        format_table_uri(&self.root_uri, &self.absolute_directory)
    }

    /// The URI of the metadata object `name`.
    pub fn build_metadata_uri(&self, name: &str) -> String {
        let absolute = self
            .store
            .absolute_key(&self.metadata_dir)
            .expect("a metadata directory under the table's store has a key");
        format!(
            "{}{}/{name}",
            self.root_uri,
            absolute.as_str().trim_matches('/')
        )
    }

    /// The URI a manifest records the data file at `path` by.
    pub fn build_uri_for_path(&self, path: &ObjectPath) -> Result<String> {
        Ok(format!("{}{}", self.root_uri, self.key_from_root(path)?))
    }

    /// The data file the table names by `path`, of `size` bytes, located
    /// for reading: the file [`locate_file_at_uri`](Self::locate_file_at_uri) locates at
    /// [`build_uri_for_path`](Self::build_uri_for_path) `path`, under the same identity and the
    /// same key, found without splitting that URI again.
    pub fn locate_file_at_path(&self, path: &ObjectPath, size: u64) -> Result<DataFile> {
        let key_from_root = self.key_from_root(path)?;
        let key = address_from_store_root(&key_from_root);
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(format!("{}{key_from_root}", self.root_uri)),
                size,
            },
            source: self.store.source(&key)?,
        })
    }

    /// The key, under the root the table's URIs are written against, of the
    /// data file the table names by `path`: what follows the root in the
    /// file's URI.
    fn key_from_root(&self, path: &ObjectPath) -> Result<String> {
        let absolute = self.store.absolute_key(&self.resolve_table_path(path))?;
        Ok(absolute.as_str().trim_start_matches('/').to_string())
    }

    /// The path the table names the data file at `uri` by: relative when the
    /// file lives under the table's directory, else the store-root-absolute
    /// key. A URI under another root is refused: the table's files are all in
    /// this store, and a reader cannot reach any other.
    pub fn find_path_for_uri(&self, uri: &str) -> Result<ObjectPath> {
        let key = key_under_root(&self.table, &self.root_uri, uri)?;
        Ok(self.name_absolute_key(&format!("/{}", key.as_str())))
    }

    /// The path the table names the file at the store-root-absolute key
    /// `absolute` by: relative when the file lives under the table's
    /// directory, else that key.
    fn name_absolute_key(&self, absolute: &str) -> ObjectPath {
        let directory_prefix = format!("{}/", self.absolute_directory);
        match absolute.strip_prefix(&directory_prefix) {
            Some(relative) if !relative.is_empty() => ObjectPath::new(relative),
            _ => ObjectPath::new(absolute),
        }
    }

    /// The path the table names the file at `path` under `directory` by,
    /// `directory` being a key under the table's store, as a writer that
    /// uploaded into it reports its files: relative to the table's directory
    /// when the file lives under it, else the store-root-absolute key.
    pub fn find_path_under(&self, directory: &ObjectPath, path: &ObjectPath) -> Result<ObjectPath> {
        let absolute = self.store.absolute_key(&directory.resolve(path))?;
        Ok(self.name_absolute_key(absolute.as_str()))
    }

    /// The directory new data files are written into, as a key under the
    /// table's store: the one `uri` names when the table names one and it
    /// lies in this store, wherever in the store that is, or `data/` under
    /// the table's own directory. The one place that knows a table's data
    /// files do not sit directly in its directory.
    pub fn data_directory_for(&self, uri: Option<&str>) -> Result<ObjectPath> {
        let named_by_table = match uri {
            Some(uri) => self.find_path_for_uri(uri.trim_end_matches('/'))?,
            None => ObjectPath::new(DATA_DIR),
        };
        Ok(self.resolve_table_path(&named_by_table))
    }
}

/// The key a table hands its store for the object whose key under the
/// table's root is `key_under_root`: addressed from the store's root, so it
/// resolves to the same object whatever prefix the store is opened on.
fn address_from_store_root(key_under_root: &str) -> ObjectPath {
    ObjectPath::new(format!("/{key_under_root}"))
}

/// The key under `root` of the object at `location`, refusing a location
/// under any other root: `table`'s files must all be under `root`.
pub fn key_under_root(
    table: &SchemaQualifiedTableName,
    root: &str,
    location: &str,
) -> Result<ObjectPath> {
    let (location_root, key) = split_location(location)?;
    if !is_same_root(&location_root, root) {
        return Err(Error::FileOutsideTableStore {
            table: table.to_string(),
            file: location.to_string(),
            root: root.to_string(),
        });
    }
    Ok(key)
}

/// `directory`, a key under `store`, as a store-root-absolute key with no
/// trailing slash.
fn find_absolute_directory(store: &dyn ObjectStore, directory: &ObjectPath) -> Result<String> {
    Ok(store
        .absolute_key(directory)?
        .as_str()
        .trim_end_matches('/')
        .to_string())
}

/// The URI of the table whose directory is the store-root-absolute key
/// `absolute_directory`, written against `root_uri`.
fn format_table_uri(root_uri: &str, absolute_directory: &str) -> String {
    format!("{root_uri}{}", absolute_directory.trim_start_matches('/'))
}

/// The root of `store`, as [`split_location`] names one: its bucket
/// (`s3://lake/`, `gs://lake/`), or `file:///` for a local store. What a
/// table that has no spelling of its own yet (one this datastore creates)
/// writes its URIs against.
pub(crate) fn store_root_uri(store: &dyn ObjectStore) -> Result<String> {
    let location = store.location_uri();
    Ok(match StoreScheme::from_uri(&location)? {
        StoreScheme::Local => "file:///".to_string(),
        scheme => {
            let (_, rest) = location
                .split_once("://")
                .expect("a bucket location carries a scheme");
            let bucket = rest.split('/').next().unwrap_or(rest);
            format!("{}{bucket}/", scheme.uri_prefix())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_storage::{DataFileLocation, LocalStore};

    /// A table whose URIs are written against the `lake` bucket, read
    /// through a store opened on its root: the local filesystem's root
    /// stands for the bucket's, since only the root check is under test.
    fn lake_table() -> TableLocation {
        TableLocation::at(
            &SchemaQualifiedTableName::new("db", "t"),
            Arc::new(LocalStore::new("/").unwrap()),
            "s3a://lake/".to_string(),
            ObjectPath::new("t"),
            ObjectPath::new("t/metadata"),
        )
        .unwrap()
    }

    #[test]
    fn a_file_anywhere_in_the_tables_bucket_is_located_by_its_key() {
        let location = lake_table();

        let file = location
            .locate_file_at_uri("s3a://lake/elsewhere/data/f.parquet", 7)
            .unwrap();

        assert_eq!(
            file.file.path.as_str(),
            "s3a://lake/elsewhere/data/f.parquet"
        );
        assert_eq!(file.file.size, 7);
        assert!(matches!(
            file.source,
            DataFileLocation::Local(path)
                if path == std::path::Path::new("/elsewhere/data/f.parquet")
        ));
    }

    #[test]
    fn a_file_in_the_tables_bucket_spelled_another_way_is_located() {
        let location = lake_table();

        let file = location
            .locate_file_at_uri("s3://lake/data/f.parquet", 7)
            .unwrap();

        assert_eq!(file.file.path.as_str(), "s3://lake/data/f.parquet");
    }

    #[test]
    fn a_file_in_another_bucket_is_refused() {
        let location = lake_table();

        let error = location
            .locate_file_at_uri("s3://other-lake/data/f.parquet", 7)
            .unwrap_err();

        assert!(
            matches!(
                &error,
                Error::FileOutsideTableStore { file, root, .. }
                    if file == "s3://other-lake/data/f.parquet" && root == "s3a://lake/"
            ),
            "{error}"
        );
    }

    #[test]
    fn a_bucket_location_splits_into_the_bucket_root_and_the_key() {
        let (root, key) = split_location("s3a://lake/warehouse/db/t/data/f.parquet").unwrap();

        assert_eq!(root, "s3a://lake/");
        assert_eq!(key.as_str(), "warehouse/db/t/data/f.parquet");
    }

    #[test]
    fn a_local_location_is_keyed_from_the_filesystem_root() {
        let (root, key) = split_location("file:///tmp/warehouse/f.parquet").unwrap();

        assert_eq!(root, "file:///");
        assert_eq!(key.as_str(), "tmp/warehouse/f.parquet");
    }

    #[test]
    fn a_local_location_without_an_authority_is_keyed_the_same() {
        let (root, key) = split_location("file:/tmp/warehouse/f.parquet").unwrap();

        assert_eq!(root, "file:/");
        assert_eq!(key.as_str(), "tmp/warehouse/f.parquet");
    }

    #[test]
    fn an_unknown_scheme_is_refused() {
        let error = split_location("hdfs://namenode/warehouse/f.parquet").unwrap_err();

        assert!(
            matches!(error, Error::UnsupportedLocation { .. }),
            "{error}"
        );
    }
}
