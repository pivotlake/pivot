//! Where a table's objects live and how they are named: the numbered
//! metadata files that are a directory table's commit log, the version hint,
//! the Avro manifests, and the full URIs manifests record data files by.

use std::sync::Arc;

use object_storage::{DataFile, FileRef, ObjectPath, ObjectStore, StoreScheme, local_path};

use crate::metadata::state::CurrentMetadata;
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
pub(crate) const METADATA_FILE_PREFIX: &str = "v";
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
    table: String,
    store: Arc<dyn ObjectStore>,
    /// The table's directory, relative to the store's prefix (or absolute
    /// from the store root).
    directory: ObjectPath,
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
    /// Whether `store` is rooted below `root_uri` (a datastore's own store,
    /// opened on a prefix of its bucket), so a URI's key under the root
    /// addresses it as a store-root-absolute key rather than one relative
    /// to the prefix.
    keys_from_store_root: bool,
}

impl TableLocation {
    /// The table `table` at `directory` of a datastore's own `store`, which
    /// every file of the table lives in, keeping its metadata under its own
    /// `metadata/`. Its URIs are written against the store's root (the
    /// bucket, or the filesystem root), and a URI's key is formed from that
    /// root, because the store itself may be opened on a prefix below it.
    pub fn in_datastore_store(
        table: &str,
        store: Arc<dyn ObjectStore>,
        directory: ObjectPath,
    ) -> Result<Self> {
        let metadata_dir = directory.join(METADATA_DIR);
        let root_uri = store_root_uri(store.as_ref())?;
        Self::at(table, store, root_uri, directory, metadata_dir, true)
    }

    /// The catalog table `table` at the version `current`, read through
    /// `store`, opened on the root of `current`'s metadata location (see
    /// [`open_store_at_root`](crate::io::store::open_store_at_root)): its
    /// directory is the one its metadata names, its metadata objects go
    /// under `write.metadata.path` when the table sets it, else under the
    /// directory's `metadata/`, and its URIs are spelled as its own
    /// metadata location is.
    pub fn of_catalog_table(
        table: &str,
        store: Arc<dyn ObjectStore>,
        current: &CurrentMetadata,
    ) -> Result<Self> {
        let (root_uri, _) = split_location(&current.metadata_location)?;
        let metadata = &current.metadata;
        let directory = key_under_root(table, &root_uri, metadata.location())?;
        let metadata_dir = match metadata.properties().get(METADATA_PATH_PROPERTY) {
            Some(path) => key_under_root(table, &root_uri, path.trim_end_matches('/'))?,
            None => directory.join(METADATA_DIR),
        };
        Self::at(table, store, root_uri, directory, metadata_dir, false)
    }

    /// The table at `directory` keeping its metadata objects under
    /// `metadata_dir` (a key under the store, relative to its prefix or
    /// absolute from its root), its URIs written against `root_uri`: the root
    /// as the table's own metadata spells it.
    fn at(
        table: &str,
        store: Arc<dyn ObjectStore>,
        root_uri: String,
        directory: ObjectPath,
        metadata_dir: ObjectPath,
        keys_from_store_root: bool,
    ) -> Result<Self> {
        let absolute_directory = store
            .absolute_key(&directory)?
            .as_str()
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            table: table.to_string(),
            store,
            directory,
            metadata_dir,
            root_uri,
            absolute_directory,
            keys_from_store_root,
        })
    }

    /// The store itself, for writing under the table's root with the same
    /// credentials the reads use.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The key under the store of the object at `location`, refusing a
    /// location under any other root.
    pub fn key_of(&self, location: &str) -> Result<ObjectPath> {
        let key = key_under_root(&self.table, &self.root_uri, location)?;
        if self.keys_from_store_root {
            return Ok(ObjectPath::new(format!("/{}", key.as_str())));
        }
        Ok(key)
    }

    /// The file at `location`, of `size` bytes, located for reading. Its
    /// identity is the full location, as the table's metadata names it.
    pub fn data_file(&self, location: &str, size: u64) -> Result<DataFile> {
        let source = self.store.source(&self.key_of(location)?)?;
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(location),
                size,
            },
            source,
        })
    }

    /// The size of the object at `location`, or `None` when no such object
    /// exists: one listing narrowed to the object's own name.
    pub fn object_size(&self, location: &str) -> Result<Option<u64>> {
        Ok(self.store.object_size(&self.key_of(location)?)?)
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

    /// The bytes of the metadata object at `uri`, which must be in the table's
    /// store.
    pub fn read_metadata_object(&self, uri: &str) -> Result<Vec<u8>> {
        let path = self.file_path_for_uri(uri)?;
        self.store
            .get(&self.data_file_key(&path))?
            .ok_or_else(|| Error::MissingMetadataObject {
                table: self.table.clone(),
                path: uri.to_string(),
            })
    }

    /// The table's own URI, recorded as the metadata's `location`.
    pub fn table_uri(&self) -> String {
        format!(
            "{}{}",
            self.root_uri,
            self.absolute_directory.trim_start_matches('/')
        )
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
            absolute.as_str().trim_matches('/')
        )
    }

    /// The URI a manifest records the data file at `path` by.
    pub fn uri_for(&self, path: &ObjectPath) -> Result<String> {
        Ok(format!("{}{}", self.root_uri, self.key_from_root(path)?))
    }

    /// The data file the table names by `path`, of `size` bytes, located
    /// for reading: the file [`data_file`](Self::data_file) locates at
    /// [`uri_for`](Self::uri_for) `path`, under the same identity and the
    /// same key, found without splitting that URI again.
    pub fn data_file_at(&self, path: &ObjectPath, size: u64) -> Result<DataFile> {
        let key_from_root = self.key_from_root(path)?;
        let key = if self.keys_from_store_root {
            ObjectPath::new(format!("/{key_from_root}"))
        } else {
            ObjectPath::new(key_from_root.as_str())
        };
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
        let absolute = self.store.absolute_key(&self.data_file_key(path))?;
        Ok(absolute.as_str().trim_start_matches('/').to_string())
    }

    /// The path the table names the data file at `uri` by: relative when the
    /// file lives under the table's directory, else the store-root-absolute
    /// key. A URI under another root is refused: the table's files are all in
    /// this store, and a reader cannot reach any other.
    pub fn file_path_for_uri(&self, uri: &str) -> Result<ObjectPath> {
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
    pub fn file_path_under(&self, directory: &ObjectPath, path: &ObjectPath) -> Result<ObjectPath> {
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
            Some(uri) => self.file_path_for_uri(uri.trim_end_matches('/'))?,
            None => ObjectPath::new(DATA_DIR),
        };
        Ok(self.data_file_key(&named_by_table))
    }
}

/// The key under `root` of the object at `location`, refusing a location
/// under any other root: `table`'s files must all be under `root`.
pub fn key_under_root(table: &str, root: &str, location: &str) -> Result<ObjectPath> {
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

    /// A table whose root is the `lake` bucket, read through a store opened
    /// on that root: backed by the local filesystem, since only the root
    /// check is under test.
    fn lake_table() -> TableLocation {
        TableLocation::at(
            "db.t",
            Arc::new(LocalStore::new(std::env::temp_dir()).unwrap()),
            "s3a://lake/".to_string(),
            ObjectPath::new("t"),
            ObjectPath::new("t/metadata"),
            false,
        )
        .unwrap()
    }

    #[test]
    fn a_datastores_own_store_addresses_files_from_its_root() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path().join("db")).unwrap());

        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t")).unwrap();
        let file = location
            .data_file(
                &format!("file://{}/db/t/pivot-1.parquet", dir.path().display()),
                3,
            )
            .unwrap();

        assert_eq!(location.root_uri, "file:///");
        assert!(matches!(
            file.source,
            DataFileLocation::Local(path) if path == dir.path().join("db/t/pivot-1.parquet")
        ));
    }

    /// What a read of `file` is cached under and where it reads from.
    fn identity_and_source(file: DataFile) -> (String, u64, std::path::PathBuf) {
        let DataFileLocation::Local(source) = file.source else {
            panic!("a local store reads from a path");
        };
        (file.file.path.as_str().to_string(), file.file.size, source)
    }

    #[test]
    fn a_file_named_by_its_path_is_the_file_at_its_uri() {
        let dir = tempfile::tempdir().unwrap();
        let prefixed: Arc<dyn ObjectStore> =
            Arc::new(LocalStore::new(dir.path().join("db")).unwrap());
        let at_root: Arc<dyn ObjectStore> = Arc::new(LocalStore::new("/").unwrap());
        let directory_from_root = dir.path().join("db/t").display().to_string();
        let locations = [
            TableLocation::in_datastore_store("main.t", prefixed, ObjectPath::new("t")).unwrap(),
            TableLocation::at(
                "db.t",
                at_root,
                "file:/".to_string(),
                ObjectPath::new(directory_from_root.trim_start_matches('/')),
                ObjectPath::new("unused"),
                false,
            )
            .unwrap(),
        ];
        let paths = [
            ObjectPath::new("data/pivot-1.parquet"),
            ObjectPath::new(format!("{}/shared/a.parquet", dir.path().display())),
        ];

        for location in &locations {
            for path in &paths {
                let by_path = location.data_file_at(path, 7).unwrap();
                let by_uri = location
                    .data_file(&location.uri_for(path).unwrap(), 7)
                    .unwrap();

                assert_eq!(identity_and_source(by_path), identity_and_source(by_uri));
            }
        }
    }

    #[test]
    fn a_file_anywhere_in_the_tables_bucket_is_located_by_its_key() {
        let location = lake_table();

        let file = location
            .data_file("s3a://lake/elsewhere/data/f.parquet", 7)
            .unwrap();

        assert_eq!(
            file.file.path.as_str(),
            "s3a://lake/elsewhere/data/f.parquet"
        );
        assert_eq!(file.file.size, 7);
        assert!(matches!(
            file.source,
            DataFileLocation::Local(path) if path.ends_with("elsewhere/data/f.parquet")
        ));
    }

    #[test]
    fn a_file_in_the_tables_bucket_spelled_another_way_is_located() {
        let location = lake_table();

        let file = location.data_file("s3://lake/data/f.parquet", 7).unwrap();

        assert_eq!(file.file.path.as_str(), "s3://lake/data/f.parquet");
    }

    #[test]
    fn a_file_in_another_bucket_is_refused() {
        let location = lake_table();

        let error = location
            .data_file("s3://other-lake/data/f.parquet", 7)
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

    #[test]
    fn a_data_file_uri_round_trips_to_the_same_path() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();
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
    fn a_table_spelling_its_root_another_way_names_its_files_that_way() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location = TableLocation::at(
            "main.t",
            store,
            "file:/".to_string(),
            ObjectPath::new("t1"),
            ObjectPath::new("t1/metadata"),
            true,
        )
        .unwrap();
        let relative = ObjectPath::new("pivot-1.parquet");

        let uri = location.uri_for(&relative).unwrap();
        let path_of_other_spelling = location
            .file_path_for_uri(&format!(
                "file://{}/t1/pivot-1.parquet",
                dir.path().display()
            ))
            .unwrap();

        assert_eq!(
            uri,
            format!("file:{}/t1/pivot-1.parquet", dir.path().display())
        );
        assert_eq!(path_of_other_spelling, relative);
    }

    #[test]
    fn a_uri_under_another_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();

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
        let root_uri = store_root_uri(store.as_ref()).unwrap();
        let location = TableLocation::at(
            "main.t",
            store,
            root_uri,
            ObjectPath::new("t1"),
            metadata_dir,
            true,
        )
        .unwrap();

        let data_directory = location
            .data_directory_for(Some(&format!("file://{}/t1/files/", dir.path().display())))
            .unwrap();

        assert_eq!(
            location.metadata_uri("m.avro"),
            format!("file://{}/elsewhere/meta/m.avro", dir.path().display())
        );
        assert_eq!(
            location.metadata_key("m.avro"),
            ObjectPath::new("elsewhere/meta/m.avro")
        );
        assert_eq!(data_directory, ObjectPath::new("t1/files"));
        assert_eq!(
            location.data_directory_for(None).unwrap(),
            ObjectPath::new("t1/data")
        );
    }

    #[test]
    fn a_file_uploaded_under_the_table_is_named_relative_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();
        let file = ObjectPath::new("a.parquet");

        let in_data_directory = location
            .file_path_under(&location.data_directory_for(None).unwrap(), &file)
            .unwrap();
        let in_table_directory = location
            .file_path_under(location.directory(), &file)
            .unwrap();

        assert_eq!(in_data_directory, ObjectPath::new("data/a.parquet"));
        assert_eq!(in_table_directory, file);
        assert_eq!(
            location.uri_for(&in_data_directory).unwrap(),
            format!("file://{}/t1/data/a.parquet", dir.path().display())
        );
    }

    #[test]
    fn a_file_uploaded_to_a_data_path_outside_the_table_is_named_by_its_absolute_key() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();
        let data_path = format!("file://{}/elsewhere/t1-data", dir.path().display());

        let data_directory = location.data_directory_for(Some(&data_path)).unwrap();
        let named = location
            .file_path_under(&data_directory, &ObjectPath::new("a.parquet"))
            .unwrap();

        assert!(data_directory.is_absolute(), "{data_directory}");
        assert_eq!(
            named,
            ObjectPath::new(format!(
                "{}/elsewhere/t1-data/a.parquet",
                dir.path().display()
            ))
        );
        assert_eq!(
            location.uri_for(&named).unwrap(),
            format!("{data_path}/a.parquet")
        );
    }
}
