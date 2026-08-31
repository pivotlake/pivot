//! End-to-end blackbox test of CREATE USER: the statement arrives over the
//! postgres protocol, is applied to the metastore when it commits, and both
//! later logins and the rewritten metastore file see the user.

mod common;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use catalog::PivotCatalog;
use catalog::metastore::{DEFAULT_USER_NAME, Metastore};
use common::{CatalogFixture, login, login_without_password, start_server_with_metastore};
use datastore_pivot::DEFAULT_REFRESH_INTERVAL;
use metastore_disk::{DiskMetastore, MetastoreConfig};
use tempfile::TempDir;
use tokio_postgres::Client;

struct CreateUserServer {
    port: u16,
    metastore_path: PathBuf,
    _metastore_dir: TempDir,
}

/// A server over a real [`DiskMetastore`] with an initially empty metastore
/// file, so created users are persisted. `Dispatch::spin_up` is
/// process-global, so the whole binary shares the one server.
fn create_user_server() -> &'static CreateUserServer {
    static SERVER: OnceLock<CreateUserServer> = OnceLock::new();
    SERVER.get_or_init(|| {
        let data_dir = TempDir::new().unwrap();
        let metastore_dir = TempDir::new().unwrap();
        let metastore_path = metastore_dir.path().join("metastore.yaml");
        std::fs::write(&metastore_path, "users: {}\n").unwrap();
        let yaml = format!(
            r#"datastores:
  default:
    kind: pivot
    location: "{}"
    default: true
"#,
            data_dir.path().display(),
        );
        let config: MetastoreConfig = serde_yaml_ng::from_str(&yaml).unwrap();
        let metastore = Arc::new(
            DiskMetastore::open(config, Some(&metastore_path), DEFAULT_REFRESH_INTERVAL).unwrap(),
        );

        let server_metastore = metastore.clone();
        let port = start_server_with_metastore(64, move |dispatch| {
            let datastores = server_metastore
                .open_datastores(dispatch.dispatcher())
                .unwrap();
            let metastore: Arc<dyn Metastore> = server_metastore.clone();
            let catalog = Arc::new(
                PivotCatalog::new(
                    datastores,
                    server_metastore.default_datastore_name().to_string(),
                    metastore.clone(),
                )
                .unwrap(),
            );
            (CatalogFixture::with_data_dir(catalog, data_dir), metastore)
        });
        CreateUserServer {
            port,
            metastore_path,
            _metastore_dir: metastore_dir,
        }
    })
}

/// Connect as the built-in trusted user, who issues the CREATE USER statements.
async fn admin() -> Client {
    login_without_password(create_user_server().port, DEFAULT_USER_NAME)
        .await
        .unwrap()
}

#[tokio::test]
async fn a_created_user_logs_in_and_lands_in_the_metastore_file() {
    let server = create_user_server();
    let admin = admin().await;

    admin
        .simple_query("CREATE USER walt PASSWORD 'blue-1'")
        .await
        .unwrap();

    let client = login(server.port, "walt", "blue-1").await.unwrap();
    client.simple_query("SELECT 1").await.unwrap();
    assert!(login(server.port, "walt", "wrong").await.is_err());
    let file = std::fs::read_to_string(&server.metastore_path).unwrap();
    assert!(file.contains("walt"), "{file}");
    assert!(
        !file.contains("blue-1"),
        "the password itself is never stored: {file}"
    );
}

#[tokio::test]
async fn a_created_user_without_a_password_is_trusted() {
    let server = create_user_server();
    let admin = admin().await;

    admin.simple_query("CREATE USER jesse").await.unwrap();

    login_without_password(server.port, "jesse").await.unwrap();
}

#[tokio::test]
async fn creating_an_existing_user_is_refused() {
    let admin = admin().await;

    // Quoted because the built-in user's name, `pivot`, is also a keyword.
    let error = admin
        .simple_query(&format!("CREATE USER \"{DEFAULT_USER_NAME}\""))
        .await
        .unwrap_err();

    let message = error.as_db_error().unwrap().message();
    assert!(message.contains("already exists"), "{message}");
}
