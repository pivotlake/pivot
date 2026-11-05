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
//! that vends credentials for the same place renews those of the store
//! already open rather than opening another, so a query holding the store
//! sends its next request with the new ones.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::{
    CLIENT_REGION, GCS_TOKEN, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY,
    S3_SESSION_TOKEN,
};
use iceberg_tables::metadata::layout::{is_same_root, key_under_root};
use iceberg_tables::{MetadataFile, split_location};
use object_storage::{
    ExternalStoreFactory, GcsStore, GcsToken, ObjectStore, S3Credentials, S3Keys, S3Store,
    StoreCredentials, StoreScheme,
};
use planner::catalog::SchemaQualifiedTableName;

use crate::{Error, Result};

/// When a vended GCS token stops being valid, in Unix milliseconds. Named by
/// the Iceberg reference implementation's GCP properties, next to the token.
const GCS_TOKEN_EXPIRES_AT: &str = "gcs.oauth2.token-expires-at";

/// The store a table's files are read and written through, opened on the
/// root of `metadata_location`: with the credentials the catalog vended for
/// the table when it vended any, else through `store_factory`.
/// `storage_properties` are the table's storage properties as the catalog
/// client merged them, the vended credentials among them. Opened when the
/// table is indexed rather than when a query loads it, since opening a store
/// may ask the bucket for its region.
pub(crate) fn open_table_store(
    storage_properties: &HashMap<String, String>,
    metadata_location: &str,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<Arc<dyn ObjectStore>> {
    let (root, _) = split_location(metadata_location)?;
    let vended = read_vended_credentials(&root, storage_properties)?;
    open_store_at_root(&root, store_factory, vended)
}

/// Whether `store`, the one the last load of the table was given,
/// authenticates from now on with the credentials in `storage_properties`:
/// the table's storage properties as the catalog client merged them at its
/// latest load, the vended credentials among them. It does when the catalog
/// vended credentials, the store is on the root of `metadata_location`, and
/// it takes them: a query holding the store then sends its next request with
/// the new ones, so vended credentials are renewed under it at every
/// refresh. A store that does not is left as it was, and the table needs one
/// opened afresh.
pub(crate) fn renew_table_store(
    store: &dyn ObjectStore,
    storage_properties: &HashMap<String, String>,
    metadata_location: &str,
) -> Result<bool> {
    let (root, _) = split_location(metadata_location)?;
    let Some(credentials) = read_vended_credentials(&root, storage_properties)? else {
        return Ok(false);
    };
    Ok(is_same_root(&store.location_uri(), &root) && store.renew_credentials(&credentials))
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
/// catalog vended any for it, else through `factory`, which applies the
/// process's credential policy.
pub(crate) fn open_store_at_root(
    root: &str,
    factory: &dyn ExternalStoreFactory,
    vended: Option<StoreCredentials>,
) -> Result<Arc<dyn ObjectStore>> {
    match vended {
        Some(StoreCredentials::S3(credentials)) => {
            Ok(Arc::new(S3Store::with_credentials(root, credentials)?))
        }
        Some(StoreCredentials::Gcs(token)) => Ok(Arc::new(GcsStore::with_token(root, token)?)),
        None => Ok(factory.open(root)?),
    }
}

/// The credentials a catalog vended for the files of a table under `root`,
/// read from `properties`, the table's storage properties as the client
/// merged them: S3 keys for a root on S3, a GCS token for one on GCS. `None`
/// when the catalog vended none of the kind the root takes. Only that kind is
/// read: what the catalog sent for another backend is not looked at.
pub(crate) fn read_vended_credentials(
    root: &str,
    properties: &HashMap<String, String>,
) -> Result<Option<StoreCredentials>> {
    Ok(match StoreScheme::from_uri(root)? {
        StoreScheme::S3 => read_vended_s3_credentials(properties)?.map(StoreCredentials::S3),
        StoreScheme::Gcs => read_vended_gcs_token(properties)?.map(StoreCredentials::Gcs),
        StoreScheme::Local => None,
    })
}

/// The GCS token a catalog vended for a table, read from the table's storage
/// properties: `gcs.oauth2.token`, with its expiry when the catalog sent one.
/// `None` when the catalog vended no token. An expiry without a token, or one
/// that is not a number of milliseconds, is an error rather than a silent
/// fall back to the process's credentials, which would read the table as
/// someone else.
fn read_vended_gcs_token(properties: &HashMap<String, String>) -> Result<Option<GcsToken>> {
    let invalid = |reason: String| Error::InvalidVendedGcsToken { reason };
    let expires_at_unix_ms = properties
        .get(GCS_TOKEN_EXPIRES_AT)
        .map(|expires_at| {
            expires_at.parse::<u64>().map_err(|_| {
                invalid(format!(
                    "`{GCS_TOKEN_EXPIRES_AT}` is `{expires_at}`, not a time in milliseconds"
                ))
            })
        })
        .transpose()?;
    match (properties.get(GCS_TOKEN), expires_at_unix_ms) {
        (Some(token), expires_at_unix_ms) => Ok(Some(GcsToken {
            token: token.clone(),
            expires_at_unix_ms,
        })),
        (None, None) => Ok(None),
        (None, Some(_)) => Err(invalid(format!(
            "`{GCS_TOKEN_EXPIRES_AT}` came without `{GCS_TOKEN}`"
        ))),
    }
}

/// The S3 credentials a catalog vended for a table, read from the table's
/// storage properties: the `s3.*` keys the load-table response carried, as
/// the client merged them. `None` when the catalog vended nothing. A key
/// without its pair is an error rather than a silent fall back to the
/// process's credentials, which would read the table as someone else.
fn read_vended_s3_credentials(
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
        _ => Err(Error::IncompleteVendedCredentials),
    }
}

#[cfg(test)]
mod tests {
    use object_storage::{DataFileLocation, ObjectPath, StoreConnection, StoreError};

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

    /// The store of a table in the `lake` bucket, as a query holds it: opened
    /// with the keys the catalog vended under `access_key`.
    fn open_store_held_by_query(access_key: &str) -> Arc<dyn ObjectStore> {
        open_table_store(
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
            held_by_query.as_ref(),
            &HashMap::from([("s3.region".to_string(), "eu-west-1".to_string())]),
            "s3://lake/db/t/metadata/v2.metadata.json",
        )
        .unwrap();

        assert!(!renewed);
        assert_eq!(access_key_of(held_by_query.as_ref()), "first");
    }

    /// The storage properties of a table the catalog vended the GCS token
    /// `token` for.
    fn vended_gcs_properties(token: &str) -> HashMap<String, String> {
        HashMap::from([("gcs.oauth2.token".to_string(), token.to_string())])
    }

    /// The `Authorization` header `store` sends with a read of one of its
    /// files.
    fn authorization_header_of(store: &dyn ObjectStore) -> String {
        let DataFileLocation::Remote {
            auth: Some(auth), ..
        } = store.source(&ObjectPath::new("part.parquet")).unwrap()
        else {
            panic!("not a store that authenticates its requests");
        };
        let headers = auth.authenticate("GET", "/part.parquet").unwrap();
        headers
            .into_iter()
            .find(|(name, _)| name == "Authorization")
            .expect("an authenticated request carries the header")
            .1
    }

    #[test]
    fn a_later_loads_gcs_token_renews_the_store_a_query_holds() {
        let held_by_query = open_table_store(
            &vended_gcs_properties("first"),
            "gs://lake/db/t/metadata/v1.metadata.json",
            &NoStoreFactory,
        )
        .unwrap();

        let renewed = renew_table_store(
            held_by_query.as_ref(),
            &vended_gcs_properties("second"),
            "gs://lake/db/t/metadata/v2.metadata.json",
        )
        .unwrap();

        assert!(renewed);
        assert_eq!(
            authorization_header_of(held_by_query.as_ref()),
            "Bearer second"
        );
    }

    #[test]
    fn a_vended_gcs_token_is_read_with_its_expiry() {
        let properties = HashMap::from([
            ("gcs.oauth2.token".to_string(), "token".to_string()),
            (
                "gcs.oauth2.token-expires-at".to_string(),
                "1767225600000".to_string(),
            ),
        ]);

        let token = read_vended_gcs_token(&properties)
            .unwrap()
            .expect("a token was vended");

        assert_eq!(token.token, "token");
        assert_eq!(token.expires_at_unix_ms, Some(1767225600000));
    }

    #[test]
    fn a_gcs_expiry_without_its_token_is_refused() {
        let properties = HashMap::from([(
            "gcs.oauth2.token-expires-at".to_string(),
            "1767225600000".to_string(),
        )]);

        let Err(error) = read_vended_gcs_token(&properties) else {
            panic!("an expiry without its token must be refused");
        };

        assert!(
            matches!(error, Error::InvalidVendedGcsToken { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_gcs_expiry_that_is_not_a_time_is_refused() {
        let properties = HashMap::from([
            ("gcs.oauth2.token".to_string(), "token".to_string()),
            (
                "gcs.oauth2.token-expires-at".to_string(),
                "tomorrow".to_string(),
            ),
        ]);

        let Err(error) = read_vended_gcs_token(&properties) else {
            panic!("an expiry that is not a time must be refused");
        };

        assert!(error.to_string().contains("tomorrow"), "{error}");
    }

    #[test]
    fn s3_keys_vended_for_a_table_on_gcs_open_no_store() {
        let Err(error) = open_table_store(
            &vended_properties("first"),
            "gs://lake/db/t/metadata/v1.metadata.json",
            &NoStoreFactory,
        ) else {
            panic!("keys of another backend must not open the table's store");
        };

        assert!(error.to_string().contains("nothing vended"), "{error}");
    }

    #[test]
    fn a_malformed_gcs_expiry_is_not_read_for_a_table_on_s3() {
        let mut properties = vended_properties("access");
        properties.insert(
            "gcs.oauth2.token-expires-at".to_string(),
            "tomorrow".to_string(),
        );

        let credentials = read_vended_credentials("s3://lake/", &properties).unwrap();

        assert!(matches!(
            credentials,
            Some(StoreCredentials::S3(credentials)) if credentials.keys.access_key == "access"
        ));
    }

    #[test]
    fn a_key_without_its_pair_is_not_read_for_a_table_on_gcs() {
        let mut properties = vended_gcs_properties("token");
        properties.insert("s3.access-key-id".to_string(), "access".to_string());

        let credentials = read_vended_credentials("gs://lake/", &properties).unwrap();

        assert!(matches!(
            credentials,
            Some(StoreCredentials::Gcs(token)) if token.token == "token"
        ));
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

        let credentials = read_vended_s3_credentials(&properties)
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

        let credentials = read_vended_s3_credentials(&properties)
            .unwrap()
            .expect("both keys were vended");

        assert_eq!(credentials.region.as_deref(), Some("us-east-1"));
    }

    #[test]
    fn properties_without_keys_vend_nothing() {
        let properties = HashMap::from([("s3.region".to_string(), "eu-west-1".to_string())]);

        let credentials = read_vended_s3_credentials(&properties).unwrap();

        assert!(credentials.is_none());
    }

    #[test]
    fn a_key_without_its_pair_is_refused() {
        let properties = HashMap::from([("s3.access-key-id".to_string(), "access".to_string())]);

        let Err(error) = read_vended_s3_credentials(&properties) else {
            panic!("a key without its pair must be refused");
        };

        assert!(
            matches!(error, Error::IncompleteVendedCredentials),
            "{error}"
        );
    }
}
