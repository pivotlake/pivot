//! A PostgreSQL container harness for integration tests, gated behind the
//! `test-support` feature and modelled on `datastore_delta::test_support`: the
//! container comes up once per test binary, and every entry point returns
//! `None` when Docker is unreachable so tests skip rather than fail offline.
//!
//! The `pivot_metastore` schema name is fixed, so concurrent tests are kept
//! apart by giving each its own database inside the shared container.

use std::sync::OnceLock;

use testcontainers::Container;
use testcontainers::runners::SyncRunner;
use testcontainers_modules::postgres::Postgres;

/// The container image's superuser credentials and maintenance database.
const ADMIN_USER: &str = "postgres";
const ADMIN_PASSWORD: &str = "postgres";
const ADMIN_DATABASE: &str = "postgres";

struct Harness {
    host: String,
    port: u16,
    _container: Container<Postgres>,
}

impl Harness {
    fn url(&self, database: &str) -> String {
        format!(
            "postgres://{ADMIN_USER}:{ADMIN_PASSWORD}@{}:{}/{database}",
            self.host, self.port
        )
    }
}

static HARNESS: OnceLock<Option<Harness>> = OnceLock::new();

fn harness() -> Option<&'static Harness> {
    HARNESS.get_or_init(start_postgres).as_ref()
}

/// Bring up PostgreSQL. Returns `None` (with a note) on any failure so the
/// tests skip rather than fail.
fn start_postgres() -> Option<Harness> {
    let container = match Postgres::default().start() {
        Ok(container) => container,
        Err(error) => {
            eprintln!("[test_support] skipping Postgres-backed tests, unavailable: {error}");
            return None;
        }
    };
    let host = container.get_host().ok()?.to_string();
    let port = container.get_host_port_ipv4(5432).ok()?;
    Some(Harness {
        host,
        port,
        _container: container,
    })
}

/// A connection URL to a freshly created database named `name`, or `None` when
/// Docker is unavailable. `name` must be a valid identifier, unique per caller.
pub fn fresh_database_url(name: &str) -> Option<String> {
    let harness = harness()?;
    let mut admin = postgres::Client::connect(&harness.url(ADMIN_DATABASE), postgres::NoTls)
        .expect("connect to the admin database");
    admin
        .batch_execute(&format!("CREATE DATABASE {name}"))
        .expect("create the test database");
    Some(harness.url(name))
}
