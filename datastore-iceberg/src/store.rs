//! The object store a table's files live in.
//!
//! Iceberg names every file by a full location (`s3://bucket/warehouse/...`).
//! A table's files are read and written through one store, opened on the
//! root of its metadata file (the bucket, or the local filesystem) with the
//! credentials the catalog vended for the table when it vended any, else
//! through the factory that applies the process's credential policy. A file
//! under another root is refused: the managed catalogs keep a table in one
//! bucket, and a table deliberately split across buckets is not served
//! rather than read through the wrong store.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::{
    CLIENT_REGION, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
};
use object_storage::{
    DataFile, ExternalStoreFactory, FileRef, ObjectPath, ObjectStore, S3Credentials, S3Keys,
    S3Store, StoreScheme, local_path,
};

use crate::{Error, Result};

/// The store one table's files are read and written through, with the root
/// it was opened on. Every read goes through it so that a file is located by
/// its key under that root, and a file under any other root is refused
/// before it is read: the two must travel together, and a reader cannot
/// skip the check.
#[derive(Clone)]
pub(crate) struct TableStore {
    table: String,
    /// The root every file of the table must be under, as the table's own
    /// metadata location spells it (`s3a://lake/`, `file:///`). A file read
    /// is checked against it by [`is_same_root`], whatever its spelling; a
    /// file this datastore writes is located at it with the key appended, so
    /// the files a commit records are named the way every other file of the
    /// table is.
    root: String,
    store: Arc<dyn ObjectStore>,
}

impl TableStore {
    /// Open the store `table`'s files live in: the root of
    /// `metadata_location`, with `vended` credentials when the catalog vended
    /// any and the root is S3, else through `factory`.
    pub(crate) fn open(
        table: &str,
        metadata_location: &str,
        factory: &dyn ExternalStoreFactory,
        vended: Option<S3Credentials>,
    ) -> Result<Self> {
        let (root, _) = split_location(metadata_location)?;
        let store = open_store_at_root(&root, factory, vended)?;
        Ok(Self {
            table: table.to_string(),
            root,
            store,
        })
    }

    /// The store itself, for a write through the table's credentials.
    pub(crate) fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The key of `location` under the table's root, refusing a location
    /// under any other.
    pub(crate) fn key_of(&self, location: &str) -> Result<ObjectPath> {
        self.key_under_root(location)
    }

    /// The full location of the object at `key` under the table's root, as
    /// the table's metadata spells locations.
    pub(crate) fn uri_of(&self, key: &ObjectPath) -> String {
        format!("{}{}", self.root, key.as_str())
    }

    /// The file at `location`, of `size` bytes, located for reading. Its
    /// identity is the full location, as the catalog names it.
    pub(crate) fn data_file(&self, location: &str, size: u64) -> Result<DataFile> {
        let source = self.store.source(&self.key_under_root(location)?)?;
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(location),
                size,
            },
            source,
        })
    }

    /// The size of the object at `location`, or `None` when no such object
    /// exists: one listing narrowed to the object's own name. What a file the
    /// catalog names without a length (a manifest list) is sized by.
    pub(crate) fn object_size(&self, location: &str) -> Result<Option<u64>> {
        Ok(self.store.object_size(&self.key_under_root(location)?)?)
    }

    /// The key of `location` under the table's root, refusing a location
    /// under any other.
    fn key_under_root(&self, location: &str) -> Result<ObjectPath> {
        let (root, key) = split_location(location)?;
        if !is_same_root(&root, &self.root) {
            return Err(Error::FileOutsideTableStore {
                table: self.table.clone(),
                file: location.to_string(),
                root: self.root.clone(),
            });
        }
        Ok(key)
    }
}

/// The store for the files under `root`: with `vended` credentials when the
/// catalog vended any and the root is S3, else through `factory`, which
/// applies the process's credential policy.
pub(crate) fn open_store_at_root(
    root: &str,
    factory: &dyn ExternalStoreFactory,
    vended: Option<S3Credentials>,
) -> Result<Arc<dyn ObjectStore>> {
    match vended {
        Some(credentials) if StoreScheme::from_uri(root)? == StoreScheme::S3 => {
            Ok(Arc::new(S3Store::with_credentials(root, credentials)?))
        }
        _ => Ok(factory.open(root)?),
    }
}

/// The S3 credentials a catalog vended for a table, read from the table's
/// storage properties: the `s3.*` keys the load-table response carried, as
/// the client merged them. `None` when the catalog vended nothing. A key
/// without its pair is an error rather than a silent fall back to the
/// process's credentials, which would read the table as someone else.
pub(crate) fn read_vended_s3_credentials(
    table: &str,
    properties: &HashMap<String, String>,
) -> Result<Option<S3Credentials>> {
    let access_key = properties.get(S3_ACCESS_KEY_ID);
    let secret_key = properties.get(S3_SECRET_ACCESS_KEY);
    match (access_key, secret_key) {
        // The REST spec names the region `client.region`; `s3.region` is the
        // older spelling. The first wins, as it does in the client library.
        (Some(access_key), Some(secret_key)) => Ok(Some(S3Credentials {
            region: properties
                .get(CLIENT_REGION)
                .or_else(|| properties.get(S3_REGION))
                .cloned(),
            keys: S3Keys {
                access_key: access_key.clone(),
                secret_key: secret_key.clone(),
                session_token: properties.get(S3_SESSION_TOKEN).cloned(),
            },
            endpoint: properties.get(S3_ENDPOINT).cloned(),
        })),
        (None, None) => Ok(None),
        _ => Err(Error::IncompleteVendedCredentials {
            table: table.to_string(),
        }),
    }
}

/// Split a file location into its root, as the location spells it
/// (`s3a://lake/`, `file:///`, `/`), and the key of the object under it: a
/// bucket root for `s3://`, `s3a://` and `gs://` locations, the filesystem
/// root for a local path. Two roots that spell one place differently are
/// told apart by [`is_same_root`], not by comparing them.
pub(crate) fn split_location(location: &str) -> Result<(String, ObjectPath)> {
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
pub(crate) fn is_same_root(first: &str, second: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use object_storage::{DataFileLocation, LocalStore};

    use super::*;

    /// A store rooted at the `lake` bucket, backed by the local filesystem
    /// since only the root check is under test.
    fn lake_store() -> TableStore {
        TableStore {
            table: "db.t".to_string(),
            root: "s3a://lake/".to_string(),
            store: Arc::new(LocalStore::new(std::env::temp_dir()).unwrap()),
        }
    }

    #[test]
    fn a_written_objects_location_is_spelled_like_the_tables_own() {
        let store = lake_store();

        let location = store.uri_of(&ObjectPath::new("warehouse/db/t/data/f.parquet"));

        assert_eq!(location, "s3a://lake/warehouse/db/t/data/f.parquet");
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
    fn a_file_anywhere_in_the_tables_bucket_is_located_by_its_key() {
        let store = lake_store();

        let file = store
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
        let store = lake_store();

        let file = store.data_file("s3://lake/data/f.parquet", 7).unwrap();

        assert_eq!(file.file.path.as_str(), "s3://lake/data/f.parquet");
    }

    #[test]
    fn a_file_in_another_bucket_is_refused() {
        let store = lake_store();

        let error = store
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
    fn vended_keys_become_credentials_with_their_token_region_and_endpoint() {
        let properties = HashMap::from([
            ("s3.access-key-id".to_string(), "access".to_string()),
            ("s3.secret-access-key".to_string(), "secret".to_string()),
            ("s3.session-token".to_string(), "token".to_string()),
            ("s3.region".to_string(), "eu-west-1".to_string()),
            (
                "s3.endpoint".to_string(),
                "http://objects.example".to_string(),
            ),
            (
                "write.parquet.compression-codec".to_string(),
                "zstd".to_string(),
            ),
        ]);

        let credentials = read_vended_s3_credentials("db.t", &properties)
            .unwrap()
            .expect("both keys were vended");

        assert_eq!(credentials.keys.access_key, "access");
        assert_eq!(credentials.keys.secret_key, "secret");
        assert_eq!(credentials.keys.session_token.as_deref(), Some("token"));
        assert_eq!(credentials.region.as_deref(), Some("eu-west-1"));
        assert_eq!(
            credentials.endpoint.as_deref(),
            Some("http://objects.example")
        );
    }

    #[test]
    fn the_region_is_read_from_client_region_before_s3_region() {
        let properties = HashMap::from([
            ("s3.access-key-id".to_string(), "access".to_string()),
            ("s3.secret-access-key".to_string(), "secret".to_string()),
            ("client.region".to_string(), "us-east-1".to_string()),
            ("s3.region".to_string(), "eu-west-1".to_string()),
        ]);

        let credentials = read_vended_s3_credentials("db.t", &properties)
            .unwrap()
            .expect("both keys were vended");

        assert_eq!(credentials.region.as_deref(), Some("us-east-1"));
    }

    #[test]
    fn properties_without_keys_vend_nothing() {
        let properties = HashMap::from([("s3.region".to_string(), "eu-west-1".to_string())]);

        let credentials = read_vended_s3_credentials("db.t", &properties).unwrap();

        assert!(credentials.is_none());
    }

    #[test]
    fn a_key_without_its_pair_is_refused() {
        let properties = HashMap::from([("s3.access-key-id".to_string(), "access".to_string())]);

        let Err(error) = read_vended_s3_credentials("db.t", &properties) else {
            panic!("a key without its pair must be refused");
        };

        assert!(
            matches!(error, Error::IncompleteVendedCredentials { .. }),
            "{error}"
        );
    }
}
