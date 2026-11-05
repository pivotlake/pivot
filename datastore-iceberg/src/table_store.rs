//! Opening the object store a catalog table's files live in, and asking it
//! for the size of the table's manifest list.
//!
//! Iceberg names every file by a full location (`s3://bucket/warehouse/...`).
//! A catalog table's files are read and written through one store, opened
//! on the root of its metadata file (the bucket, or the local filesystem)
//! with the credentials the catalog vended for the table when it vended
//! any, else through the factory that applies the process's credential
//! policy. A snapshot names its manifest list without a length, so the
//! store is asked for the size of the current one when the table is indexed.
//!
//! Vended credentials are a session's: short-lived. A later load of the table
//! that vends keys for the same root, region and endpoint renews the keys of
//! the store already open rather than opening another, so a query holding
//! the store signs its next request with the new keys.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::{
    CLIENT_REGION, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
};
use iceberg_tables::metadata::layout::{is_same_root, key_under_root};
use iceberg_tables::{MetadataFile, split_location};
use object_storage::{
    ExternalStoreFactory, ObjectStore, S3Credentials, S3Keys, S3Store, StoreScheme,
};
use planner::catalog::SchemaQualifiedTableName;

use crate::{Error, Result};

/// The store the files of the table `name` are read and written through,
/// opened on the root of `metadata_location`: with the credentials among
/// `storage_properties`, the table's storage properties as the catalog
/// client merged them, when the catalog vended any, else through
/// `store_factory`. Opened when the table is indexed rather than when a
/// query loads it, since opening a store may ask the bucket for its region.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    storage_properties: &HashMap<String, String>,
    metadata_location: &str,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<Arc<dyn ObjectStore>> {
    let vended = read_vended_s3_credentials(&name.to_string(), storage_properties)?;
    let (root, _) = split_location(metadata_location)?;
    open_store_at_root(&root, store_factory, vended)
}

/// Whether `store`, the one the last load of the table `name` was given,
/// signs with the credentials among `storage_properties` from now on. It
/// does when the catalog vended credentials, the store is on the root of
/// `metadata_location`, and it takes them: a query holding the store then
/// signs its next request with the new keys, so vended credentials are
/// renewed under it at every refresh. A store that does not is left as it
/// was, and the table needs one opened afresh.
pub(crate) fn renew_table_store(
    name: &SchemaQualifiedTableName,
    store: &dyn ObjectStore,
    storage_properties: &HashMap<String, String>,
    metadata_location: &str,
) -> Result<bool> {
    let Some(credentials) = read_vended_s3_credentials(&name.to_string(), storage_properties)?
    else {
        return Ok(false);
    };
    let (root, _) = split_location(metadata_location)?;
    Ok(is_same_root(&store.location_uri(), &root) && store.renew_s3_credentials(&credentials))
}

/// The size of the manifest list of the current snapshot `metadata_file`
/// holds, or `None` when it holds no current snapshot. The snapshot names
/// its manifest list without a length, and a ring read needs one, so
/// `store`, opened on the root of the metadata file's location, is asked
/// once, when the table is indexed, rather than on every query over it.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    metadata_file: &MetadataFile,
    store: &dyn ObjectStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = metadata_file.metadata.current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let (root, _) = split_location(&metadata_file.location)?;
    let key = key_under_root(name, &root, path)?;
    let size = store
        .object_size(&key)
        .map_err(iceberg_tables::Error::from)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
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

#[cfg(test)]
mod tests {
    use object_storage::{StoreConnection, StoreError};

    use super::*;

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

    /// The storage properties of a table the catalog vended keys for under
    /// `access_key`, at a fixed region and endpoint, so opening a store with
    /// them asks nothing of the network.
    fn vended_properties(access_key: &str) -> HashMap<String, String> {
        HashMap::from([
            ("s3.access-key-id".to_string(), access_key.to_string()),
            ("s3.secret-access-key".to_string(), "secret".to_string()),
            (
                "s3.session-token".to_string(),
                format!("{access_key}-token"),
            ),
            ("s3.region".to_string(), "eu-west-1".to_string()),
            (
                "s3.endpoint".to_string(),
                "http://objects.example".to_string(),
            ),
        ])
    }

    fn table_name() -> SchemaQualifiedTableName {
        SchemaQualifiedTableName::new("db".to_string(), "t".to_string())
    }

    /// The store of a table in the `lake` bucket, as a query holds it: opened
    /// with the keys the catalog vended under `access_key`.
    fn open_store_held_by_query(access_key: &str) -> Arc<dyn ObjectStore> {
        open_table_store(
            &table_name(),
            &vended_properties(access_key),
            "s3://lake/db/t/metadata/v1.metadata.json",
            &NoStoreFactory,
        )
        .unwrap()
    }

    fn access_key_of(store: &dyn ObjectStore) -> String {
        match store.connection() {
            StoreConnection::S3 {
                credentials: Some(credentials),
                ..
            } => credentials.keys.access_key,
            _ => panic!("not a signing S3 store"),
        }
    }

    #[test]
    fn a_later_loads_keys_renew_the_store_a_query_holds() {
        let held_by_query = open_store_held_by_query("first");

        let renewed = renew_table_store(
            &table_name(),
            held_by_query.as_ref(),
            &vended_properties("second"),
            "s3://lake/db/t/metadata/v2.metadata.json",
        )
        .unwrap();

        assert!(renewed);
        assert_eq!(access_key_of(held_by_query.as_ref()), "second");
    }

    #[test]
    fn keys_vended_for_a_table_moved_to_another_bucket_do_not_renew_the_store() {
        let held_by_query = open_store_held_by_query("first");

        let renewed = renew_table_store(
            &table_name(),
            held_by_query.as_ref(),
            &vended_properties("second"),
            "s3://other-lake/db/t/metadata/v2.metadata.json",
        )
        .unwrap();

        assert!(!renewed);
        assert_eq!(access_key_of(held_by_query.as_ref()), "first");
    }

    #[test]
    fn a_load_that_vends_no_keys_does_not_renew_the_store() {
        let held_by_query = open_store_held_by_query("first");

        let renewed = renew_table_store(
            &table_name(),
            held_by_query.as_ref(),
            &HashMap::from([("s3.region".to_string(), "eu-west-1".to_string())]),
            "s3://lake/db/t/metadata/v2.metadata.json",
        )
        .unwrap();

        assert!(!renewed);
        assert_eq!(access_key_of(held_by_query.as_ref()), "first");
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
