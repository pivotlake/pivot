//! End-to-end blackbox test of a server whose metastore is a shared PostgreSQL
//! database: the datastore it serves and the users it authenticates are rows,
//! administered with plain SQL while the server is running. Environments
//! without Docker skip.

mod common;

use std::sync::{Arc, OnceLock};

use catalog::PivotCatalog;
use common::{CatalogFixture, select_rows, start_server_with_metastore};
use metastore::{Metastore, SCRAM_ITERATIONS, ScramVerifier, format_scram_verifier};
use metastore_postgres::{PostgresMetastore, PostgresMetastoreConfig, test_support};
use pgwire::api::auth::sasl::scram::gen_salted_password;
use tempfile::TempDir;
use tokio_postgres::{Client, NoTls};

const USER: &str = "analytics";
const PASSWORD: &str = "s3cret pässword";
const LIVE_USER: &str = "live_user";

fn verifier_for(password: &str) -> String {
    let salt = vec![7; 16];
    format_scram_verifier(&ScramVerifier {
        salted_password: gen_salted_password(password, &salt, SCRAM_ITERATIONS),
        salt,
    })
}

struct PostgresBackedServer {
    port: u16,
    /// The metastore database, for tests that administer rows while serving.
    metastore_url: String,
}

/// One server over a Postgres metastore holding a local default datastore and
/// one SCRAM user, or `None` without Docker. Plain `#[test]`s build this in a
/// synchronous context: testcontainers' `SyncRunner` and the blocking metastore
/// setup must not run inside a tokio runtime.
fn postgres_backed_server() -> &'static Option<PostgresBackedServer> {
    static SERVER: OnceLock<Option<PostgresBackedServer>> = OnceLock::new();
    SERVER.get_or_init(|| {
        let metastore_url = test_support::fresh_database_url("server_e2e")?;
        let data_dir = TempDir::new().unwrap();

        // First connection creates the schema; the datastore row must exist
        // before the server-facing connection reads the default's name.
        assert!(PostgresMetastore::connect(PostgresMetastoreConfig::new(&metastore_url)).is_err());
        let mut admin = postgres::Client::connect(&metastore_url, postgres::NoTls).unwrap();
        admin
            .execute(
                "INSERT INTO pivot_metastore.datastores (name, location, is_default) \
                 VALUES ('hot', $1, true)",
                &[&data_dir.path().to_str().unwrap()],
            )
            .unwrap();
        admin
            .execute(
                "INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier) \
                 VALUES ($1, 'scram-sha-256', $2)",
                &[&USER, &verifier_for(PASSWORD)],
            )
            .unwrap();

        let metastore: Arc<dyn Metastore> = Arc::new(
            PostgresMetastore::connect(PostgresMetastoreConfig::new(&metastore_url)).unwrap(),
        );
        let server_metastore = metastore.clone();
        let port = start_server_with_metastore(64, move |dispatch| {
            let datastores = server_metastore
                .open_datastores(dispatch.dispatcher())
                .unwrap();
            let catalog = Arc::new(
                PivotCatalog::new(
                    datastores,
                    server_metastore.default_datastore_name().to_string(),
                )
                .unwrap(),
            );
            (
                CatalogFixture::with_data_dir(catalog, data_dir),
                server_metastore,
            )
        });
        Some(PostgresBackedServer {
            port,
            metastore_url,
        })
    })
}

/// Drive an async body on a fresh runtime: plain `#[test]` keeps the fixture's
/// container startup and blocking clients out of any ambient tokio runtime.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

async fn login(port: u16, user: &str, password: &str) -> Result<Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user(user)
        .password(password)
        .dbname("test")
        .connect(NoTls)
        .await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

#[test]
fn a_datastore_and_user_defined_in_postgres_serve_queries() {
    let Some(server) = postgres_backed_server() else {
        return;
    };

    block_on(async {
        let client = login(server.port, USER, PASSWORD).await.unwrap();
        client
            .simple_query("CREATE TABLE events (v INT)")
            .await
            .unwrap();
        client
            .simple_query("INSERT INTO events VALUES (41), (1)")
            .await
            .unwrap();

        let rows = select_rows(&client, "SELECT SUM(v) FROM events").await;

        assert_eq!(rows, vec![vec![Some("42".to_string())]]);
    });
}

#[test]
fn users_administered_in_postgres_apply_to_later_logins() {
    let Some(server) = postgres_backed_server() else {
        return;
    };
    let mut admin = postgres::Client::connect(&server.metastore_url, postgres::NoTls).unwrap();

    block_on(async {
        assert!(
            login(server.port, LIVE_USER, "first password")
                .await
                .is_err()
        );
    });
    admin
        .execute(
            "INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier) \
             VALUES ($1, 'scram-sha-256', $2)",
            &[&LIVE_USER, &verifier_for("first password")],
        )
        .unwrap();
    block_on(async {
        login(server.port, LIVE_USER, "first password")
            .await
            .unwrap();
    });

    admin
        .execute(
            "UPDATE pivot_metastore.users SET scram_verifier = $1 WHERE name = $2",
            &[&verifier_for("second password"), &LIVE_USER],
        )
        .unwrap();

    block_on(async {
        assert!(
            login(server.port, LIVE_USER, "first password")
                .await
                .is_err()
        );
        login(server.port, LIVE_USER, "second password")
            .await
            .unwrap();
    });
}
