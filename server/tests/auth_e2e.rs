//! End-to-end blackbox test of PostgreSQL authentication: a `metastore` section
//! defines one SCRAM-SHA-256 user and one explicitly trusted user. Each
//! connection selects that user's method, while an unconfigured name is
//! rejected.
//!
//! The metastore used here is mutable so the suite can also prove that adding a
//! user and rotating its password affect later logins without a server restart.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use catalog::{Datastore, PivotCatalog};
use common::{CatalogFixture, select_rows, start_server_with_metastore};
use dispatch::DataFlowDispatcher;
use metastore::{Metastore, SCRAM_ITERATIONS, ScramVerifier, UserAuth};
use metastore_yaml::YamlMetastore;
use pgwire::api::auth::sasl::scram::gen_salted_password;
use tempfile::TempDir;
use tokio_postgres::{Client, NoTls};

const USER: &str = "analytics";
const TRUSTED_USER: &str = "reader";
const LIVE_USER: &str = "live-user";
const PASSWORD: &str = "s3cret pässword";

fn scram_auth(password: &str) -> UserAuth {
    let salt = vec![7; 16];
    UserAuth::ScramSha256(ScramVerifier {
        salted_password: gen_salted_password(password, &salt, SCRAM_ITERATIONS),
        salt,
    })
}

/// A metastore whose user map can change while the server is running. The
/// datastore half delegates to YAML; authentication reads the lock on every
/// login, mirroring a future metastore with live user administration.
struct MutableMetastore {
    inner: YamlMetastore,
    users: RwLock<HashMap<String, UserAuth>>,
}

impl MutableMetastore {
    fn set_user(&self, name: &str, auth: UserAuth) {
        self.users.write().unwrap().insert(name.to_string(), auth);
    }
}

impl Metastore for MutableMetastore {
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        self.inner.open_datastores(dispatcher)
    }

    fn default_datastore_name(&self) -> &str {
        self.inner.default_datastore_name()
    }

    fn user_auth(&self, username: &str) -> metastore::Result<Option<UserAuth>> {
        Ok(self.users.read().unwrap().get(username).cloned())
    }
}

/// Connect as `user`/`password`, returning the wire error when the login fails.
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

/// Connect without configuring a client password.
async fn login_without_password(port: u16, user: &str) -> Result<Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user(user)
        .dbname("test")
        .connect(NoTls)
        .await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

struct AuthServer {
    port: u16,
    metastore: Arc<MutableMetastore>,
}

/// A server whose metastore initially defines one SCRAM user and one trusted
/// user. `Dispatch::spin_up` is process-global, so the whole binary shares it.
fn authenticating_server() -> &'static AuthServer {
    static SERVER: OnceLock<AuthServer> = OnceLock::new();
    SERVER.get_or_init(|| {
        let data_dir = TempDir::new().unwrap();
        let yaml = format!(
            r#"datastores:
  default:
    kind: delta
    location: "{}"
    default: true
"#,
            data_dir.path().display(),
        );
        let inner = YamlMetastore::from_yaml(&yaml, "test").unwrap();
        let metastore = Arc::new(MutableMetastore {
            inner,
            users: RwLock::new(HashMap::from([
                (USER.to_string(), scram_auth(PASSWORD)),
                (TRUSTED_USER.to_string(), UserAuth::Trust),
            ])),
        });

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
            let metastore: Arc<dyn Metastore> = server_metastore;
            (CatalogFixture::with_data_dir(catalog, data_dir), metastore)
        });
        AuthServer { port, metastore }
    })
}

fn authenticating_server_port() -> u16 {
    authenticating_server().port
}

#[tokio::test(flavor = "multi_thread")]
async fn a_configured_user_with_its_password_can_query() {
    let port = authenticating_server_port();

    let client = login(port, USER, PASSWORD).await.unwrap();

    assert_eq!(
        select_rows(&client, "SELECT 1 AS ok").await,
        vec![vec![Some("1".to_string())]]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_explicitly_trusted_user_can_query_without_a_password() {
    let port = authenticating_server_port();

    let client = login_without_password(port, TRUSTED_USER).await.unwrap();

    assert_eq!(
        select_rows(&client, "SELECT 1 AS ok").await,
        vec![vec![Some("1".to_string())]]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_configured_user_with_the_wrong_password_is_rejected() {
    let port = authenticating_server_port();

    let error = login(port, USER, "not the password").await.unwrap_err();

    assert_eq!(error.as_db_error().unwrap().code().code(), "28P01");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unconfigured_user_is_rejected() {
    let port = authenticating_server_port();

    let error = login(port, "nobody", PASSWORD).await.unwrap_err();

    assert_eq!(error.as_db_error().unwrap().code().code(), "28P01");
}

/// An unconfigured user must fail the way a wrong password does, at the same
/// point in the handshake, or the difference names the accounts that exist.
#[tokio::test(flavor = "multi_thread")]
async fn an_unconfigured_user_fails_indistinguishably_from_a_wrong_password() {
    let port = authenticating_server_port();

    let unknown = login(port, "nobody", PASSWORD).await.unwrap_err();
    let wrong = login(port, USER, "not the password").await.unwrap_err();

    assert_eq!(
        unknown.as_db_error().unwrap().message(),
        wrong.as_db_error().unwrap().message()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn later_logins_see_added_users_and_rotated_passwords() {
    let server = authenticating_server();

    assert!(
        login(server.port, LIVE_USER, "first password")
            .await
            .is_err()
    );

    server
        .metastore
        .set_user(LIVE_USER, scram_auth("first password"));
    login(server.port, LIVE_USER, "first password")
        .await
        .unwrap();

    server
        .metastore
        .set_user(LIVE_USER, scram_auth("second password"));
    assert!(
        login(server.port, LIVE_USER, "first password")
            .await
            .is_err()
    );
    login(server.port, LIVE_USER, "second password")
        .await
        .unwrap();
}
