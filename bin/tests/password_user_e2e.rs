//! End-to-end blackbox test of a user whose config entry spells its password
//! out: the file's `password` method must let that password in over SCRAM and
//! keep everything else out, built-in `pivot` name included.

mod common;

use std::sync::{Arc, OnceLock};

use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use common::{CatalogFixture, login, login_without_password, start_server_with_metastore};
use datastore_pivotlake::DEFAULT_REFRESH_INTERVAL;
use metastore_disk::{DiskMetastore, MetastoreConfig};
use tempfile::TempDir;

const PASSWORD: &str = "Password1337";

/// A server whose config defines `pivot` with a raw password. `Dispatch::spin_up`
/// is process-global, so the whole binary shares the one server.
fn password_user_server_port() -> u16 {
    static PORT: OnceLock<u16> = OnceLock::new();
    *PORT.get_or_init(|| {
        let data_dir = TempDir::new().unwrap();
        let yaml = format!(
            r#"datastores:
  default:
    kind: pivotlake
    location: "{}"
    default: true
users:
  pivot:
    auth:
      method: password
      password: {PASSWORD}
"#,
            data_dir.path().display(),
        );
        let config: MetastoreConfig = serde_yaml_ng::from_str(&yaml).unwrap();
        let metastore =
            Arc::new(DiskMetastore::open(config, None, DEFAULT_REFRESH_INTERVAL).unwrap());

        start_server_with_metastore(64, move |dispatch| {
            let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();
            let metastore: Arc<dyn Metastore> = metastore.clone();
            let catalog = Arc::new(
                PivotCatalog::new(
                    datastores,
                    metastore.default_datastore_name().to_string(),
                    metastore.clone(),
                )
                .unwrap(),
            );
            (CatalogFixture::with_data_dir(catalog, data_dir), metastore)
        })
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn the_configured_password_logs_in() {
    let port = password_user_server_port();

    let client = login(port, "pivot", PASSWORD).await.unwrap();

    client.simple_query("SELECT 1").await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_password_is_rejected() {
    let port = password_user_server_port();

    let error = login(port, "pivot", "not the password").await.unwrap_err();

    assert_eq!(error.as_db_error().unwrap().code().code(), "28P01");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_built_in_trust_of_pivot_is_taken_over_by_the_password() {
    let port = password_user_server_port();

    let result = login_without_password(port, "pivot").await;

    assert!(result.is_err(), "pivot logged in without its password");
}
