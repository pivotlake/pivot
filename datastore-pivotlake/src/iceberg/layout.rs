//! Where a managed table's objects live and how they are named: the numbered
//! metadata files that are the table's commit log, the version hint, the Avro
//! manifests, and the full URIs manifests record data files by.

use std::sync::Arc;

use object_storage::{ObjectPath, ObjectStore, StoreScheme};
use uuid::Uuid;

use super::{Error, Result};
use datastore_iceberg::split_location;

/// The directory under a table's own, holding every metadata object.
pub(crate) const METADATA_DIR: &str = "metadata";
/// The file under [`METADATA_DIR`] naming the current metadata version, for
/// readers that open the table by its path. Rewritten after every commit, best
/// effort; a reader lists from the version it names, never trusting it.
pub(crate) const VERSION_HINT_FILE: &str = "version-hint.text";
/// Metadata versions are zero-padded to this many digits, so that the
/// store's byte-ordered listing is the version order: a listing started after
/// one version returns only newer ones, and the last key is the current one.
const VERSION_DIGITS: usize = 20;
const METADATA_FILE_PREFIX: &str = "v";
const METADATA_FILE_SUFFIX: &str = ".metadata.json";

/// `v{version}.metadata.json`, the version zero-padded to [`VERSION_DIGITS`].
pub(crate) fn metadata_file_name(version: u64) -> String {
    format!("{METADATA_FILE_PREFIX}{version:0VERSION_DIGITS$}{METADATA_FILE_SUFFIX}")
}

/// The version a metadata file name carries, or `None` for a name that is not
/// one (a sidecar, a file another tool dropped in): exactly `v`, the padded
/// digits, and the suffix.
pub(crate) fn parse_metadata_file_version(name: &str) -> Option<u64> {
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
pub(crate) fn hint_content(version: u64) -> String {
    format!("{version:0VERSION_DIGITS$}")
}

/// The version a hint file holds. Anything but the padded digits (with
/// surrounding whitespace tolerated) is an error, never a silently ignored
/// hint: a corrupt hint is worth knowing about.
pub(crate) fn parse_hint(bytes: &[u8]) -> Result<u64> {
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
pub(crate) fn manifest_list_file_name(snapshot_id: i64, commit: Uuid) -> String {
    format!("snap-{snapshot_id}-1-{commit}.avro")
}

/// The `index`th manifest written by commit `commit`.
pub(crate) fn manifest_file_name(commit: Uuid, index: usize) -> String {
    format!("{commit}-m{index}.avro")
}

/// Where one managed table lives: the datastore's store and the table's
/// directory under it, with the root every URI the table records is written
/// against. Iceberg names files by full URI, so a key under the store and its
/// URI must convert both ways exactly: a data file written as
/// `pivot-….parquet` under the table reads back as that same relative path,
/// and a file the table adopted from elsewhere in the store as the same
/// store-root-absolute key it was adopted by.
#[derive(Clone)]
pub(crate) struct TableLocation {
    store: Arc<dyn ObjectStore>,
    /// The table's directory, relative to the database root (or absolute from
    /// the store root, for a table that lives outside the database).
    directory: ObjectPath,
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
    pub(crate) fn new(store: Arc<dyn ObjectStore>, directory: ObjectPath) -> Result<Self> {
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
            root_uri,
            split_root,
            absolute_directory,
        })
    }

    pub(crate) fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The directory the metadata objects live in.
    pub(crate) fn metadata_dir(&self) -> ObjectPath {
        self.directory.join(METADATA_DIR)
    }

    /// The key of the metadata object `name`.
    pub(crate) fn metadata_key(&self, name: &str) -> ObjectPath {
        self.metadata_dir().join(name)
    }

    /// The key of a data file the table names by `path` (relative to the
    /// table, or store-root-absolute for an adopted file).
    pub(crate) fn data_file_key(&self, path: &ObjectPath) -> ObjectPath {
        self.directory.resolve(path)
    }

    /// The table's own URI, recorded as the metadata's `location`.
    pub(crate) fn table_uri(&self) -> String {
        format!("{}{}", self.root_uri, self.absolute_directory)
    }

    /// The URI of the metadata object `name`.
    pub(crate) fn metadata_uri(&self, name: &str) -> String {
        format!("{}/{METADATA_DIR}/{name}", self.table_uri())
    }

    /// The URI a manifest records the data file at `path` by.
    pub(crate) fn uri_for(&self, path: &ObjectPath) -> Result<String> {
        let absolute = self.store.absolute_key(&self.data_file_key(path))?;
        Ok(format!("{}{}", self.root_uri, absolute.as_str()))
    }

    /// The path the table names the data file at `uri` by: relative when the
    /// file lives under the table's directory, else the store-root-absolute
    /// key. A URI under another root is refused: the table's files are all in
    /// this store, and a reader cannot reach any other.
    pub(crate) fn file_path_for_uri(&self, uri: &str) -> Result<ObjectPath> {
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
}
