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
//!
//! Vended credentials are a session's: short-lived. A later load of the table
//! that vends keys for the same root, region and endpoint renews the keys of
//! the store already open rather than opening another, so a query holding the
//! store signs its next request with the new keys.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use iceberg::io::{
    CLIENT_REGION, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
};
use object_storage::{
    DataFile, ExternalStoreFactory, FileRef, ObjectPath, ObjectStore, S3Credentials, S3Keys,
    S3Store, StoreScheme, local_path,
};

use crate::{Error, Result};

/// The table property a catalog sets to say when the session keys it vended
/// stop working, as Unix milliseconds.
const S3_SESSION_TOKEN_EXPIRES_AT_MS: &str = "s3.session-token-expires-at-ms";

/// The S3 credentials a catalog vended for a table, with when they stop
/// working when the catalog said.
pub(crate) struct VendedCredentials {
    pub credentials: S3Credentials,
    pub expires_at: Option<SystemTime>,
}

/// The store one table's files are read and written through, with the root
/// it was opened on. Every read goes through it so that a file is located by
/// its key under that root, and a file under any other root is refused
/// before it is read: the two must travel together, and a reader cannot
/// skip the check.
#[derive(Clone)]
pub(crate) struct TableStore {
    table: String,
    /// The root every file of the table must be under, as
    /// [`split_location`] names it.
    root: String,
    /// What a key under the root is appended to for the location of a file
    /// this datastore writes: the root as the table's own metadata location
    /// spells it (`s3a://bucket/`, `file:///`), so the files a commit records
    /// are named the way every other file of the table is.
    uri_prefix: String,
    store: Arc<dyn ObjectStore>,
    /// This store as the S3 store the catalog's vended keys opened, when they
    /// did: what a later load's keys renew.
    vended: Option<VendedStore>,
}

/// The S3 store a table's vended keys opened, with what they were vended for:
/// the keys of a later load that match on both renew this store rather than
/// open another.
#[derive(Clone)]
struct VendedStore {
    store: Arc<S3Store>,
    region: Option<String>,
    endpoint: Option<String>,
    /// When the keys the store signs with stop working, when the catalog said.
    expires_at: Option<SystemTime>,
}

impl TableStore {
    /// Open the store `table`'s files live in: the root of
    /// `metadata_location`, with `vended` credentials when the catalog vended
    /// any and the root is S3, else through `factory`.
    pub(crate) fn open(
        table: &str,
        metadata_location: &str,
        factory: &dyn ExternalStoreFactory,
        vended: Option<VendedCredentials>,
    ) -> Result<Self> {
        let (root, key) = split_location(metadata_location)?;
        let uri_prefix = metadata_location
            .strip_suffix(key.as_str())
            .ok_or_else(|| Error::UnsupportedLocation {
                path: metadata_location.to_string(),
                message: "the location does not end in its object key".to_string(),
            })?
            .to_string();
        let (store, vended): (Arc<dyn ObjectStore>, _) = match vended {
            Some(vended) if StoreScheme::of(&root)? == StoreScheme::S3 => {
                let region = vended.credentials.region.clone();
                let endpoint = vended.credentials.endpoint.clone();
                let store = Arc::new(S3Store::with_credentials(&root, vended.credentials)?);
                (
                    Arc::clone(&store) as Arc<dyn ObjectStore>,
                    Some(VendedStore {
                        store,
                        region,
                        endpoint,
                        expires_at: vended.expires_at,
                    }),
                )
            }
            _ => (factory.open(&root)?, None),
        };
        Ok(Self {
            table: table.to_string(),
            root,
            uri_prefix,
            store,
            vended,
        })
    }

    /// The store `table`'s files live in, as [`open`](Self::open) gives it,
    /// except that when `previous` is the store an earlier load opened with
    /// vended keys for the same root, region and endpoint, `vended`'s keys
    /// renew it in place and it is the store returned: a query holding
    /// `previous` signs its next request with the new keys.
    pub(crate) fn open_or_renew(
        previous: Option<&TableStore>,
        table: &str,
        metadata_location: &str,
        factory: &dyn ExternalStoreFactory,
        vended: Option<VendedCredentials>,
    ) -> Result<Self> {
        let (root, _) = split_location(metadata_location)?;
        let renewable = previous
            .filter(|previous| previous.root == root)
            .and_then(|previous| previous.vended.as_ref().map(|store| (previous, store)));
        match (renewable, vended) {
            (Some((previous, held)), Some(vended))
                if held.region == vended.credentials.region
                    && held.endpoint == vended.credentials.endpoint =>
            {
                held.store.renew_keys(vended.credentials.keys)?;
                Ok(Self {
                    vended: Some(VendedStore {
                        expires_at: vended.expires_at,
                        ..held.clone()
                    }),
                    ..previous.clone()
                })
            }
            (_, vended) => Self::open(table, metadata_location, factory, vended),
        }
    }

    /// When the keys this store signs with stop working: what the catalog said
    /// when it vended them, `None` when it vended none or said nothing.
    pub(crate) fn keys_expire_at(&self) -> Option<SystemTime> {
        self.vended.as_ref().and_then(|vended| vended.expires_at)
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
        format!("{}{}", self.uri_prefix, key.as_str())
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
        let key = self.key_under_root(location)?;
        let directory = key.parent().unwrap_or_default();
        let listing = self.store.list_with_name_prefix(&directory, key.name())?;
        Ok(listing
            .objects
            .into_iter()
            .find(|object| object.file.path.as_str() == key.name())
            .map(|object| object.file.size))
    }

    /// The key of `location` under the table's root, refusing a location
    /// under any other.
    fn key_under_root(&self, location: &str) -> Result<ObjectPath> {
        let (root, key) = split_location(location)?;
        if root != self.root {
            return Err(Error::FileOutsideTableStore {
                table: self.table.clone(),
                file: location.to_string(),
                root: self.root.clone(),
            });
        }
        Ok(key)
    }
}

/// The S3 credentials a catalog vended for a table, read from the table's
/// storage properties: the `s3.*` keys the load-table response carried, as
/// the client merged them, and when they expire when the catalog said. `None`
/// when the catalog vended nothing. A key without its pair is an error rather
/// than a silent fall back to the process's credentials, which would read the
/// table as someone else; an expiry that does not parse is an error rather
/// than keys renewed too late.
pub(crate) fn read_vended_s3_credentials(
    table: &str,
    properties: &HashMap<String, String>,
) -> Result<Option<VendedCredentials>> {
    let access_key = properties.get(S3_ACCESS_KEY_ID);
    let secret_key = properties.get(S3_SECRET_ACCESS_KEY);
    let (access_key, secret_key) = match (access_key, secret_key) {
        (Some(access_key), Some(secret_key)) => (access_key, secret_key),
        (None, None) => return Ok(None),
        _ => {
            return Err(Error::IncompleteVendedCredentials {
                table: table.to_string(),
            });
        }
    };
    let expires_at = match properties.get(S3_SESSION_TOKEN_EXPIRES_AT_MS) {
        Some(millis) => Some(
            millis
                .parse::<u64>()
                .ok()
                .map(|millis| SystemTime::UNIX_EPOCH + Duration::from_millis(millis))
                .ok_or_else(|| Error::MalformedVendedExpiry {
                    table: table.to_string(),
                    value: millis.clone(),
                })?,
        ),
        None => None,
    };
    Ok(Some(VendedCredentials {
        credentials: S3Credentials {
            // The REST spec names the region `client.region`; `s3.region` is
            // the older spelling. The first wins, as it does in the client
            // library.
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
        },
        expires_at,
    }))
}

/// Split a file location into the root the store is opened on and the key of
/// the object under it: a bucket root for `s3://`, `s3a://` and `gs://`
/// locations, the filesystem root for a local path.
fn split_location(location: &str) -> Result<(String, ObjectPath)> {
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

#[cfg(test)]
mod tests {
    use object_storage::{DataFileLocation, LocalStore, StoreConnection, StoreError};

    use super::*;

    /// A store rooted at the `lake` bucket, backed by the local filesystem
    /// since only the root check is under test.
    fn lake_store() -> TableStore {
        TableStore {
            table: "db.t".to_string(),
            root: "s3://lake".to_string(),
            uri_prefix: "s3a://lake/".to_string(),
            store: Arc::new(LocalStore::new(std::env::temp_dir()).unwrap()),
            vended: None,
        }
    }

    /// A factory that opens nothing, for a table whose every store the catalog
    /// vends keys for.
    #[derive(Debug)]
    struct NoStoreFactory;

    impl ExternalStoreFactory for NoStoreFactory {
        fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
            Err(StoreError::Config(format!(
                "nothing vended for `{root_uri}`"
            )))
        }
    }

    /// Keys vended for the `lake` bucket at a fixed endpoint and region, so
    /// opening the store asks nothing of the network.
    fn vended_keys(access_key: &str, endpoint: &str) -> VendedCredentials {
        VendedCredentials {
            credentials: S3Credentials {
                region: Some("eu-west-1".to_string()),
                keys: S3Keys {
                    access_key: access_key.to_string(),
                    secret_key: "secret".to_string(),
                    session_token: Some(format!("{access_key}-token")),
                },
                endpoint: Some(endpoint.to_string()),
            },
            expires_at: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000)),
        }
    }

    fn access_key_of(store: &TableStore) -> String {
        match store.store.connection() {
            StoreConnection::S3 {
                credentials: Some(credentials),
                ..
            } => credentials.keys.access_key,
            _ => panic!("not a signing S3 store"),
        }
    }

    #[test]
    fn a_later_loads_keys_renew_the_store_a_query_holds() {
        let first = TableStore::open(
            "db.t",
            "s3://lake/db/t/metadata/v1.metadata.json",
            &NoStoreFactory,
            Some(vended_keys("first", "http://objects.example")),
        )
        .unwrap();
        let held = first.clone();

        let renewed = TableStore::open_or_renew(
            Some(&first),
            "db.t",
            "s3://lake/db/t/metadata/v2.metadata.json",
            &NoStoreFactory,
            Some(vended_keys("second", "http://objects.example")),
        )
        .unwrap();

        assert!(Arc::ptr_eq(&renewed.store, &held.store));
        assert_eq!(access_key_of(&held), "second");
        assert_eq!(
            renewed.keys_expire_at(),
            vended_keys("second", "").expires_at
        );
    }

    #[test]
    fn keys_vended_for_another_endpoint_open_a_store_of_their_own() {
        let first = TableStore::open(
            "db.t",
            "s3://lake/db/t/metadata/v1.metadata.json",
            &NoStoreFactory,
            Some(vended_keys("first", "http://objects.example")),
        )
        .unwrap();

        let reopened = TableStore::open_or_renew(
            Some(&first),
            "db.t",
            "s3://lake/db/t/metadata/v2.metadata.json",
            &NoStoreFactory,
            Some(vended_keys("second", "http://elsewhere.example")),
        )
        .unwrap();

        assert!(!Arc::ptr_eq(&reopened.store, &first.store));
        assert_eq!(access_key_of(&first), "first");
        assert_eq!(access_key_of(&reopened), "second");
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
    fn a_file_in_another_bucket_is_refused() {
        let store = lake_store();

        let error = store
            .data_file("s3://other-lake/data/f.parquet", 7)
            .unwrap_err();

        assert!(
            matches!(
                &error,
                Error::FileOutsideTableStore { file, root, .. }
                    if file == "s3://other-lake/data/f.parquet" && root == "s3://lake"
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

        let vended = read_vended_s3_credentials("db.t", &properties)
            .unwrap()
            .expect("both keys were vended");

        let credentials = vended.credentials;
        assert_eq!(credentials.keys.access_key, "access");
        assert_eq!(credentials.keys.secret_key, "secret");
        assert_eq!(credentials.keys.session_token.as_deref(), Some("token"));
        assert_eq!(credentials.region.as_deref(), Some("eu-west-1"));
        assert_eq!(
            credentials.endpoint.as_deref(),
            Some("http://objects.example")
        );
        assert_eq!(vended.expires_at, None);
    }

    #[test]
    fn the_keys_expiry_is_read_as_unix_milliseconds() {
        let properties = HashMap::from([
            ("s3.access-key-id".to_string(), "access".to_string()),
            ("s3.secret-access-key".to_string(), "secret".to_string()),
            (
                "s3.session-token-expires-at-ms".to_string(),
                "1800000000123".to_string(),
            ),
        ]);

        let vended = read_vended_s3_credentials("db.t", &properties)
            .unwrap()
            .expect("both keys were vended");

        assert_eq!(
            vended.expires_at,
            Some(SystemTime::UNIX_EPOCH + Duration::from_millis(1_800_000_000_123))
        );
    }

    #[test]
    fn an_expiry_that_is_not_a_timestamp_is_refused() {
        let properties = HashMap::from([
            ("s3.access-key-id".to_string(), "access".to_string()),
            ("s3.secret-access-key".to_string(), "secret".to_string()),
            (
                "s3.session-token-expires-at-ms".to_string(),
                "tomorrow".to_string(),
            ),
        ]);

        let error = read_vended_s3_credentials("db.t", &properties)
            .err()
            .expect("an expiry that is not a timestamp is refused");

        assert!(
            matches!(&error, Error::MalformedVendedExpiry { value, .. } if value == "tomorrow"),
            "{error}"
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

        let vended = read_vended_s3_credentials("db.t", &properties)
            .unwrap()
            .expect("both keys were vended");

        assert_eq!(vended.credentials.region.as_deref(), Some("us-east-1"));
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
