//! Opening the object store a catalog table's files live in.
//!
//! Iceberg names every file by a full location (`s3://bucket/warehouse/...`).
//! A catalog table's files are read and written through one store, opened
//! on the root of its metadata file (the bucket, or the local filesystem)
//! with the credentials the catalog vended for the table when it vended
//! any, else through the factory that applies the process's credential
//! policy.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::io::{
    CLIENT_REGION, S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
};
use object_storage::{
    ExternalStoreFactory, ObjectStore, S3Credentials, S3Keys, S3Store, StoreScheme,
};

use crate::{Error, Result};

/// The store for the files under `root`: with `vended` credentials when the
/// catalog vended any and the root is S3, else through `factory`, which
/// applies the process's credential policy.
pub fn open_store_at_root(
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
pub fn read_vended_s3_credentials(
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
    use super::*;

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
