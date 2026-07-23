//! Aspirational end-to-end test: a Kafka Connect **JDBC PostgreSQL sink** writing
//! rows into pivot over the Postgres wire protocol.
//!
//! The pipeline it stands up is the real thing, nothing simulated:
//!
//! ```text
//!   record --> Kafka topic --> Kafka Connect (JdbcSinkConnector) --pgwire--> pivot
//! ```
//!
//! A single record is produced to a topic; a JDBC sink connector is pointed at a
//! live pivot server (`connection.url = jdbc:postgresql://.../test`) with
//! `auto.create=true`; the connector is expected to `CREATE TABLE "people"` and
//! `INSERT` the row; the test then reads the row back out of pivot with a plain
//! Postgres client.
//!
//! It is `#[ignore]`d because pivot does not support what the sink needs yet, so
//! it cannot pass today. It is checked in as an executable definition of "done":
//! drop the `#[ignore]` once the gaps below are closed and it should go green.
//!
//! What the connector needs that pivot is missing today:
//!   1. **Extended query protocol** (Parse/Bind/Execute). The JDBC driver drives
//!      everything through prepared statements; pivot registers a `NoopHandler`
//!      for extended queries, so the connection never gets past its first
//!      prepared statement. This is the hard blocker.
//!   2. **`INSERT` with an explicit column list** (`INSERT INTO t (a, b) VALUES
//!      (?, ?)`). The sink always names its columns; pivot only accepts
//!      positional `INSERT INTO t VALUES (...)`.
//!   3. **Server-side parameter binding** for the `?` placeholders the sink binds
//!      per row (part of the extended protocol above).
//!   4. **Catalog introspection** the JDBC driver's `DatabaseMetaData` issues to
//!      check whether the target table exists (`information_schema` / `pg_catalog`
//!      lookups) before `auto.create` fires.
//!   5. **Transactions** (`BEGIN`/`COMMIT`): the sink turns off autocommit and
//!      commits each batch. pivot has no multi-statement transactions, but now
//!      accepts `BEGIN`/`COMMIT`/`ROLLBACK` as no-ops (each statement still
//!      commits on its own), so this no longer blocks the connection.
//!   6. Only reached in richer configs, but worth noting: `ON CONFLICT ... DO
//!      UPDATE` for `insert.mode=upsert`, and `ALTER TABLE ... ADD COLUMN` for
//!      `auto.evolve=true`.
//!
//! Requires a working Docker daemon (Kafka + Kafka Connect containers) and pulls
//! the JDBC sink connector from Confluent Hub at container start, so it also
//! needs network egress the first time.

use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use catalog::ParquetCatalog;
use dispatch::Dispatch;
use server::Server;
use tempfile::TempDir;
use testcontainers::core::{ExecCommand, Host, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, ContainerRequest, GenericImage, ImageExt};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

/// The topic produced to and the table the sink is expected to auto-create.
const TOPIC: &str = "people";
/// User-defined Docker network so Connect can reach the broker by name.
const NETWORK: &str = "pivot-kafka-connect-e2e";
/// The hostname (backed by Docker's host-gateway) the Connect container uses to
/// reach the pivot server running on the test host.
const PIVOT_HOST: &str = "host.docker.internal";

const KAFKA_IMAGE: &str = "confluentinc/cp-kafka";
const CONNECT_IMAGE: &str = "confluentinc/cp-kafka-connect";
const CONFLUENT_TAG: &str = "7.6.1";
/// Confluent Hub coordinates of the JDBC sink connector, installed at Connect
/// start. The connector bundles the PostgreSQL JDBC driver.
const JDBC_CONNECTOR: &str = "confluentinc/kafka-connect-jdbc:10.7.6";

// --- pivot server ----------------------------------------------------------

/// Start a pivot server whose catalog is rooted at a local `root`, bound on
/// **all interfaces** (so the Connect container can reach it via host-gateway),
/// and return the chosen port once it is listening.
fn start_pivot_server(root: &str) -> u16 {
    let port = pick_free_port();
    let bind: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    let workers = core_affinity::get_core_ids().unwrap().len().clamp(1, 4);
    let root = root.to_string();
    thread::spawn(move || {
        let dispatch = Dispatch::spin_up(workers, 64, None);
        let catalog = Arc::new(ParquetCatalog::open(&root, dispatch.dispatcher()).unwrap());
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let server = Server::new(
                bind,
                dispatch,
                catalog,
                0,
                4,
                server::DEFAULT_CATALOG_REFRESH,
            );
            let _ = server.serve(Box::pin(std::future::pending::<()>())).await;
        });
    });
    wait_until_listening(SocketAddr::from(([127, 0, 0, 1], port)));
    port
}

fn pick_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_until_listening(addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(addr).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("pivot server failed to start listening on {addr}");
}

// --- container images ------------------------------------------------------

/// A single-node Kafka broker in KRaft mode (no ZooKeeper), reachable by other
/// containers on the shared network as `kafka:9092`.
fn kafka_image() -> ContainerRequest<GenericImage> {
    GenericImage::new(KAFKA_IMAGE, CONFLUENT_TAG)
        .with_wait_for(WaitFor::message_on_stdout("Kafka Server started"))
        .with_network(NETWORK)
        .with_container_name("kafka")
        .with_env_var("KAFKA_NODE_ID", "1")
        .with_env_var("KAFKA_PROCESS_ROLES", "broker,controller")
        .with_env_var("KAFKA_CONTROLLER_QUORUM_VOTERS", "1@kafka:9093")
        .with_env_var(
            "KAFKA_LISTENERS",
            "PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093",
        )
        .with_env_var("KAFKA_ADVERTISED_LISTENERS", "PLAINTEXT://kafka:9092")
        .with_env_var(
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
            "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
        )
        .with_env_var("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
        .with_env_var("KAFKA_INTER_BROKER_LISTENER_NAME", "PLAINTEXT")
        .with_env_var("KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR", "1")
        .with_env_var("KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR", "1")
        .with_env_var("KAFKA_TRANSACTION_STATE_LOG_MIN_ISR", "1")
        .with_env_var("KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS", "0")
        .with_env_var("KAFKA_AUTO_CREATE_TOPICS_ENABLE", "true")
        // A fixed base64 cluster id keeps KRaft format deterministic.
        .with_env_var("CLUSTER_ID", "MkU3OEVBNTcwNTJENDM2Qk")
}

/// A Kafka Connect worker that installs the JDBC sink connector on start and can
/// reach the pivot server on the host via [`PIVOT_HOST`].
fn connect_image() -> ContainerRequest<GenericImage> {
    GenericImage::new(CONNECT_IMAGE, CONFLUENT_TAG)
        .with_wait_for(WaitFor::message_on_stdout("Kafka Connect started"))
        .with_network(NETWORK)
        .with_container_name("connect")
        .with_host(PIVOT_HOST, Host::HostGateway)
        // Install the JDBC sink connector, then hand off to the standard entrypoint.
        .with_cmd([
            "bash".to_string(),
            "-c".to_string(),
            format!(
                "confluent-hub install --no-prompt {JDBC_CONNECTOR} && /etc/confluent/docker/run"
            ),
        ])
        .with_env_var("CONNECT_BOOTSTRAP_SERVERS", "kafka:9092")
        .with_env_var("CONNECT_GROUP_ID", "pivot-connect")
        .with_env_var("CONNECT_CONFIG_STORAGE_TOPIC", "connect-configs")
        .with_env_var("CONNECT_OFFSET_STORAGE_TOPIC", "connect-offsets")
        .with_env_var("CONNECT_STATUS_STORAGE_TOPIC", "connect-status")
        .with_env_var("CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR", "1")
        .with_env_var("CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR", "1")
        .with_env_var("CONNECT_STATUS_STORAGE_REPLICATION_FACTOR", "1")
        .with_env_var(
            "CONNECT_KEY_CONVERTER",
            "org.apache.kafka.connect.json.JsonConverter",
        )
        .with_env_var(
            "CONNECT_VALUE_CONVERTER",
            "org.apache.kafka.connect.json.JsonConverter",
        )
        .with_env_var("CONNECT_REST_ADVERTISED_HOST_NAME", "connect")
        .with_env_var(
            "CONNECT_PLUGIN_PATH",
            "/usr/share/java,/usr/share/confluent-hub-components",
        )
        // confluent-hub install downloads the connector, so allow a long start.
        .with_startup_timeout(Duration::from_secs(300))
}

// --- container helpers -----------------------------------------------------

/// Run `argv` inside `container` and return `(exit_code, stdout)`.
fn exec(container: &Container<GenericImage>, argv: Vec<String>) -> (i64, String) {
    let mut result = container.exec(ExecCommand::new(argv)).unwrap();
    let stdout = String::from_utf8_lossy(&result.stdout_to_vec().unwrap()).into_owned();
    let code = result.exit_code().unwrap().unwrap_or(-1);
    (code, stdout)
}

/// The one record produced to the topic: JSON envelope carrying its own schema,
/// the shape the `JsonConverter` (with `schemas.enable=true`) expects.
fn record_json() -> String {
    r#"{"schema":{"type":"struct","optional":false,"name":"people","fields":[{"type":"int64","optional":false,"field":"id"},{"type":"string","optional":false,"field":"name"}]},"payload":{"id":1,"name":"alice"}}"#
        .to_string()
}

/// Produce [`record_json`] to [`TOPIC`] from inside the broker container.
fn produce_record(kafka: &Container<GenericImage>) {
    let record = record_json();
    let (code, out) = exec(
        kafka,
        vec![
            "bash".into(),
            "-c".into(),
            format!(
                "echo '{record}' | kafka-console-producer --bootstrap-server kafka:9092 --topic {TOPIC}"
            ),
        ],
    );
    assert_eq!(code, 0, "kafka-console-producer failed: {out}");
}

/// The JDBC sink connector config, pointed at pivot on the host.
fn connector_config(pivot_port: u16) -> String {
    format!(
        r#"{{
          "name": "pivot-sink",
          "config": {{
            "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
            "tasks.max": "1",
            "topics": "{TOPIC}",
            "connection.url": "jdbc:postgresql://{PIVOT_HOST}:{pivot_port}/test?preferQueryMode=simple",
            "connection.user": "test",
            "connection.password": "test",
            "dialect.name": "PostgreSqlDatabaseDialect",
            "auto.create": "true",
            "insert.mode": "insert",
            "pk.mode": "none",
            "key.converter": "org.apache.kafka.connect.json.JsonConverter",
            "value.converter": "org.apache.kafka.connect.json.JsonConverter",
            "value.converter.schemas.enable": "true",
            "consumer.override.auto.offset.reset": "earliest"
          }}
        }}"#
    )
}

/// POST the sink connector config to the Connect REST API from inside the
/// Connect container. Asserts a 2xx create.
fn create_connector(connect: &Container<GenericImage>, config: &str) {
    let (code, out) = exec(
        connect,
        vec![
            "bash".into(),
            "-c".into(),
            format!(
                "curl -s -o /dev/null -w '%{{http_code}}' -X POST \
                 -H 'Content-Type: application/json' \
                 --data '{config}' http://localhost:8083/connectors"
            ),
        ],
    );
    assert_eq!(code, 0, "curl failed");
    assert!(out.starts_with('2'), "connector create returned HTTP {out}");
}

// --- pivot client ----------------------------------------------------------

async fn connect_client(port: u16) -> Client {
    let (client, conn) = tokio_postgres::Config::new()
        .host("127.0.0.1")
        .port(port)
        .user("test")
        .dbname("test")
        .connect(NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// Poll `SELECT id, name FROM people` until the produced row appears or the
/// deadline passes. The table not existing yet (or an empty result) just means
/// the sink hasn't landed the batch; keep waiting.
async fn wait_for_row(client: &Client, deadline: Instant) -> Vec<(String, String)> {
    while Instant::now() < deadline {
        if let Ok(messages) = client
            .simple_query("SELECT id, name FROM people ORDER BY id")
            .await
        {
            let rows: Vec<(String, String)> = messages
                .into_iter()
                .filter_map(|m| match m {
                    SimpleQueryMessage::Row(r) => {
                        Some((r.get(0)?.to_string(), r.get(1)?.to_string()))
                    }
                    _ => None,
                })
                .collect();
            if !rows.is_empty() {
                return rows;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Vec::new()
}

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(fut)
}

// --- the test --------------------------------------------------------------

#[test]
#[ignore = "pivot lacks the extended query protocol the JDBC sink requires; see module docs"]
fn kafka_connect_jdbc_sink_lands_rows() {
    let root = TempDir::new().unwrap();
    let pivot_port = start_pivot_server(root.path().to_str().unwrap());

    let kafka = kafka_image().start().unwrap();
    produce_record(&kafka);
    let connect = connect_image().start().unwrap();

    create_connector(&connect, &connector_config(pivot_port));

    let rows = block_on(async {
        let client = connect_client(pivot_port).await;
        wait_for_row(&client, Instant::now() + Duration::from_secs(120)).await
    });

    assert_eq!(rows, vec![("1".to_string(), "alice".to_string())]);
}
