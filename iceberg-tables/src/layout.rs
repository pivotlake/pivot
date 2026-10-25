//! Where a table's objects live and how they are named: the numbered
//! metadata files that are a directory table's commit log, the version hint,
//! the Avro manifests, and the full URIs manifests record data files by.

use std::sync::Arc;

use object_storage::{DataFile, FileRef, ObjectPath, ObjectStore, StoreScheme, local_path};
use uuid::Uuid;

use crate::{Error, Result};

/// The directory under a table's own holding its metadata objects, unless
/// the table names another.
pub const METADATA_DIR: &str = "metadata";
/// The directory under a table's own that a catalog table's data files are
/// written to, unless the table names another: the layout every other
/// writer uses.
pub const DATA_DIR: &str = "data";
/// The file under the metadata directory naming the current metadata version
/// of a directory table, for readers that open the table by its path.
/// Rewritten after every commit, best effort; a reader lists from the version
/// it names, never trusting it.
pub const VERSION_HINT_FILE: &str = "version-hint.text";
/// The table property naming where data files are written, when not under
/// the table's own `data/`.
pub const DATA_PATH_PROPERTY: &str = "write.data.path";
/// The table property naming where metadata objects are written, when not
/// under the table's own `metadata/`.
pub const METADATA_PATH_PROPERTY: &str = "write.metadata.path";
/// Metadata versions are zero-padded to this many digits, so that the
/// store's byte-ordered listing is the version order: a listing started after
/// one version returns only newer ones, and the last key is the current one.
const VERSION_DIGITS: usize = 20;
const METADATA_FILE_PREFIX: &str = "v";
const METADATA_FILE_SUFFIX: &str = ".metadata.json";

/// `v{version}.metadata.json`, the version zero-padded to twenty digits.
pub fn metadata_file_name(version: u64) -> String {
    format!("{METADATA_FILE_PREFIX}{version:0VERSION_DIGITS$}{METADATA_FILE_SUFFIX}")
}

/// The version a metadata file name carries, or `None` for a name that is not
/// one (a sidecar, a file another tool dropped in): exactly `v`, the padded
/// digits, and the suffix.
pub fn parse_metadata_file_version(name: &str) -> Option<u64> {
    let digits = name
        .strip_prefix(METADATA_FILE_PREFIX)?
        .strip_suffix(METADATA_FILE_SUFFIX)?;
    if digits.len() != VERSION_DIGITS || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The hint's content for `version`: the same padded digits as the file name,
/// which a path reader substitutes straight into `v{hint}.metadata.json`.
pub fn hint_content(version: u64) -> String {
    format!("{version:0VERSION_DIGITS$}")
}

/// The version a hint file holds. Anything but the padded digits (with
/// surrounding whitespace tolerated) is an error, never a silently ignored
/// hint: a corrupt hint is worth knowing about.
pub fn parse_hint(bytes: &[u8]) -> Result<u64> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| Error::MalformedVersionHint {
            content: String::from_utf8_lossy(bytes).into_owned(),
        })?
        .trim();
    if text.len() != VERSION_DIGITS || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::MalformedVersionHint {
            content: text.to_string(),
        });
    }
    text.parse().map_err(|_| Error::MalformedVersionHint {
        content: text.to_string(),
    })
}

/// The manifest list of snapshot `snapshot_id`, written by commit `commit`.
pub fn manifest_list_file_name(snapshot_id: i64, commit: Uuid) -> String {
    format!("snap-{snapshot_id}-1-{commit}.avro")
}

/// The `index`th manifest written by commit `commit`.
pub fn manifest_file_name(commit: Uuid, index: usize) -> String {
    format!("{commit}-m{index}.avro")
}

/// Split a file location into the root a store is opened on and the key of
/// the object under it: a bucket root for `s3://`, `s3a://` and `gs://`
/// locations, the filesystem root for a local path.
pub fn split_location(location: &str) -> Result<(String, ObjectPath)> {
    let unsupported = |message: String| Error::UnsupportedLocation {
        path: location.to_string(),
        message,
    };
    let scheme = StoreScheme::of(location).map_err(|error| unsupported(error.to_string()))?;
    if scheme == StoreScheme::Local {
        let path = local_path(location);
        if !path.starts_with('/') {
            return Err(unsupported(
                "a local location must be an absolute path".to_string(),
            ));
        }
        return Ok((
            "/".to_string(),
            ObjectPath::new(path.trim_start_matches('/')),
        ));
    }
    let (_, rest) = location
        .split_once("://")
        .expect("a bucket location carries a scheme");
    match rest.split_once('/') {
        Some((bucket, key)) if !bucket.is_empty() && !key.is_empty() => Ok((
            format!("{}{bucket}", scheme.uri_prefix()),
            ObjectPath::new(key),
        )),
        _ => Err(unsupported(
            "expected a bucket and an object key".to_string(),
        )),
    }
}

/// Where one table lives: its store and its directory under it, with the
/// root every URI the table records is written against. Iceberg names files
/// by full URI, so a key under the store and its URI must convert both ways
/// exactly: a data file written as `pivot-….parquet` under the table reads
/// back as that same relative path, and a file the table adopted from
/// elsewhere in the store as the same store-root-absolute key it was adopted
/// by.
#[derive(Clone)]
pub struct TableLocation {
    store: Arc<dyn ObjectStore>,
    /// The table's directory, relative to the store's prefix (or absolute
    /// from the store root).
    directory: ObjectPath,
    /// Where the table's metadata objects go: its own `metadata/` unless the
    /// table names another directory.
    metadata_dir: ObjectPath,
    /// What a store-root-absolute key is appended to for a URI: the bucket
    /// (`s3://bucket`, `gs://bucket`) or `file://` for a local store.
    root_uri: String,
    /// The root [`split_location`] names for a URI in this store: the bucket,
    /// or `/` for a local store.
    split_root: String,
    /// The table's directory as a store-root-absolute key, which relative data
    /// file paths are resolved against and URIs are stripped of.
    absolute_directory: String,
}

impl TableLocation {
    /// The table at `directory` keeping its metadata under its own
    /// `metadata/`.
    pub fn new(store: Arc<dyn ObjectStore>, directory: ObjectPath) -> Result<Self> {
        let metadata_dir = directory.join(METADATA_DIR);
        Self::with_metadata_dir(store, directory, metadata_dir)
    }

    /// The table at `directory` keeping its metadata objects under
    /// `metadata_dir` (a key under the store, relative to its prefix or
    /// absolute from its root).
    pub fn with_metadata_dir(
        store: Arc<dyn ObjectStore>,
        directory: ObjectPath,
        metadata_dir: ObjectPath,
    ) -> Result<Self> {
        let root_uri = root_uri(store.as_ref())?;
        let split_root = if root_uri == "file://" {
            "/".to_string()
        } else {
            root_uri.clone()
        };
        let absolute_directory = store
            .absolute_key(&directory)?
            .as_str()
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            store,
            directory,
            metadata_dir,
            root_uri,
            split_root,
            absolute_directory,
        })
    }

    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The table's directory under the store.
    pub fn directory(&self) -> &ObjectPath {
        &self.directory
    }

    /// The directory the metadata objects live in.
    pub fn metadata_dir(&self) -> ObjectPath {
        self.metadata_dir.clone()
    }

    /// The key of the metadata object `name`.
    pub fn metadata_key(&self, name: &str) -> ObjectPath {
        self.metadata_dir.join(name)
    }

    /// The key of a data file the table names by `path` (relative to the
    /// table, or store-root-absolute for an adopted file).
    pub fn data_file_key(&self, path: &ObjectPath) -> ObjectPath {
        self.directory.resolve(path)
    }

    /// The object at `uri`, `size` bytes long, located for a read over the
    /// worker pool: what a manifest, whose length its manifest list records,
    /// is read as, through the pool's caches.
    pub fn data_file(&self, uri: &str, size: u64) -> Result<DataFile> {
        let key = self.data_file_key(&self.file_path_for_uri(uri)?);
        let source = self.store.source(&key)?;
        Ok(DataFile {
            file: FileRef { path: key, size },
            source,
        })
    }

    /// The table's own URI, recorded as the metadata's `location`.
    pub fn table_uri(&self) -> String {
        format!("{}{}", self.root_uri, self.absolute_directory)
    }

    /// The URI of the metadata object `name`.
    pub fn metadata_uri(&self, name: &str) -> String {
        let absolute = self
            .store
            .absolute_key(&self.metadata_dir)
            .expect("a metadata directory under the table's store has a key");
        format!(
            "{}{}/{name}",
            self.root_uri,
            absolute.as_str().trim_end_matches('/')
        )
    }

    /// The URI a manifest records the data file at `path` by.
    pub fn uri_for(&self, path: &ObjectPath) -> Result<String> {
        let absolute = self.store.absolute_key(&self.data_file_key(path))?;
        Ok(format!("{}{}", self.root_uri, absolute.as_str()))
    }

    /// The path the table names the data file at `uri` by: relative when the
    /// file lives under the table's directory, else the store-root-absolute
    /// key. A URI under another root is refused: the table's files are all in
    /// this store, and a reader cannot reach any other.
    pub fn file_path_for_uri(&self, uri: &str) -> Result<ObjectPath> {
        let (root, key) = split_location(uri)?;
        if root != self.split_root {
            return Err(Error::FileOutsideTableStore {
                table: self.directory.to_string(),
                file: uri.to_string(),
                root: self.root_uri.clone(),
            });
        }
        let absolute = format!("/{}", key.as_str());
        let directory_prefix = format!("{}/", self.absolute_directory);
        Ok(match absolute.strip_prefix(&directory_prefix) {
            Some(relative) if !relative.is_empty() => ObjectPath::new(relative),
            _ => ObjectPath::new(absolute),
        })
    }

    /// The path, as the table names data files, under which new data files
    /// are written: `uri` when the table names one and it lies in this
    /// store (relative to the table when under its directory, else
    /// store-root-absolute), or `data/` under the table's own directory.
    pub fn data_prefix_for(&self, uri: Option<&str>) -> Result<ObjectPath> {
        match uri {
            Some(uri) => self.file_path_for_uri(uri.trim_end_matches('/')),
            None => Ok(ObjectPath::new(DATA_DIR)),
        }
    }
}

/// What a store-root-absolute key is appended to for a URI in `store`.
fn root_uri(store: &dyn ObjectStore) -> Result<String> {
    let location = store.location_uri();
    Ok(match StoreScheme::of(&location)? {
        StoreScheme::Local => "file://".to_string(),
        scheme => {
            let (_, rest) = location
                .split_once("://")
                .expect("a bucket location carries a scheme");
            let bucket = rest.split('/').next().unwrap_or(rest);
            format!("{}{bucket}", scheme.uri_prefix())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_storage::LocalStore;

    #[test]
    fn metadata_file_names_are_padded_and_parse_back_strictly() {
        assert_eq!(
            metadata_file_name(12),
            "v00000000000000000012.metadata.json"
        );
        assert_eq!(
            parse_metadata_file_version("v00000000000000000012.metadata.json"),
            Some(12)
        );
        assert_eq!(parse_metadata_file_version("v12.metadata.json"), None);
        assert_eq!(parse_metadata_file_version("version-hint.text"), None);
        assert_eq!(
            parse_metadata_file_version("v00000000000000000012.metadata.json.tmp"),
            None
        );
    }

    #[test]
    fn padded_names_sort_in_version_order() {
        let mut names = vec![metadata_file_name(10), metadata_file_name(9)];

        names.sort();

        assert_eq!(names, [metadata_file_name(9), metadata_file_name(10)]);
    }

    #[test]
    fn a_hint_holds_the_padded_digits_and_rejects_anything_else() {
        assert_eq!(hint_content(7), "00000000000000000007");
        assert_eq!(parse_hint(b"00000000000000000007\n").unwrap(), 7);
        assert!(matches!(
            parse_hint(b"7"),
            Err(Error::MalformedVersionHint { .. })
        ));
    }

    #[test]
    fn a_bucket_location_splits_into_the_bucket_root_and_the_key() {
        let (root, key) = split_location("s3a://lake/warehouse/db/t/data/f.parquet").unwrap();

        assert_eq!(root, "s3://lake");
        assert_eq!(key.as_str(), "warehouse/db/t/data/f.parquet");
    }

    #[test]
    fn a_local_location_is_keyed_from_the_filesystem_root() {
        let (root, key) = split_location("file:///tmp/warehouse/f.parquet").unwrap();

        assert_eq!(root, "/");
        assert_eq!(key.as_str(), "tmp/warehouse/f.parquet");
    }

    #[test]
    fn a_local_location_without_an_authority_is_keyed_the_same() {
        let (root, key) = split_location("file:/tmp/warehouse/f.parquet").unwrap();

        assert_eq!(root, "/");
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

    #[test]
    fn a_data_file_uri_round_trips_to_the_same_path() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::new(store, ObjectPath::new("t1")).unwrap();
        let relative = ObjectPath::new("pivot-1.parquet");
        let adopted = ObjectPath::new(format!("{}/shared/a.parquet", dir.path().display()));

        let relative_uri = location.uri_for(&relative).unwrap();
        let adopted_uri = location.uri_for(&adopted).unwrap();

        assert_eq!(
            relative_uri,
            format!("file://{}/t1/pivot-1.parquet", dir.path().display())
        );
        assert_eq!(location.file_path_for_uri(&relative_uri).unwrap(), relative);
        assert_eq!(location.file_path_for_uri(&adopted_uri).unwrap(), adopted);
        assert_eq!(
            location.table_uri(),
            format!("file://{}/t1", dir.path().display())
        );
        assert_eq!(
            location.metadata_uri("m.avro"),
            format!("file://{}/t1/metadata/m.avro", dir.path().display())
        );
    }

    #[test]
    fn a_uri_under_another_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::new(store, ObjectPath::new("t1")).unwrap();

        let error = location
            .file_path_for_uri("s3://elsewhere/t1/pivot-1.parquet")
            .unwrap_err();

        assert!(
            matches!(error, Error::FileOutsideTableStore { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_table_naming_its_own_paths_keeps_metadata_and_data_there() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let metadata_dir = ObjectPath::new("elsewhere/meta");
        let location =
            TableLocation::with_metadata_dir(store, ObjectPath::new("t1"), metadata_dir).unwrap();

        let data_prefix = location
            .data_prefix_for(Some(&format!("file://{}/t1/files/", dir.path().display())))
            .unwrap();

        assert_eq!(
            location.metadata_uri("m.avro"),
            format!("file://{}/elsewhere/meta/m.avro", dir.path().display())
        );
        assert_eq!(
            location.metadata_key("m.avro"),
            ObjectPath::new("elsewhere/meta/m.avro")
        );
        assert_eq!(data_prefix, ObjectPath::new("files"));
        assert_eq!(
            location.data_prefix_for(None).unwrap(),
            ObjectPath::new("data")
        );
    }
}
