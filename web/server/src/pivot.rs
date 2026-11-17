//! Reading a pivot server's system tables.

use std::time::{SystemTime, UNIX_EPOCH};

use chrono::DateTime;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage, SimpleQueryRow};
use tracing::{info, warn};

/// What this backend calls itself when it connects. Its own connection is left
/// out of what it reports: the page shows the server's clients, and a
/// connection that exists only to draw the page is not one of them.
const APPLICATION_NAME: &str = "pivot-web";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{}", Error::describe_postgres(.0))]
    Postgres(#[from] tokio_postgres::Error),
    #[error("the server returned no value for `{column}`")]
    MissingValue { column: &'static str },
    #[error("the server returned `{value}` for `{column}`, which does not parse")]
    MalformedValue { column: &'static str, value: String },
}

impl Error {
    /// A `tokio_postgres::Error` displays as a bare "db error" or "error
    /// connecting to server"; what went wrong is in its sources.
    fn describe_postgres(error: &tokio_postgres::Error) -> String {
        let mut description = error.to_string();
        let mut source = std::error::Error::source(error);
        while let Some(cause) = source {
            description.push_str(": ");
            description.push_str(&cause.to_string());
            source = cause.source();
        }
        description
    }
}

/// Reading the values of a row the server sent as text.
trait TextRow {
    fn read_text(&self, column: &'static str) -> Result<&str, Error>;

    fn parse_integer(&self, column: &'static str) -> Result<i64, Error>;

    /// Milliseconds since the Unix epoch.
    fn parse_instant_ms(&self, column: &'static str) -> Result<i64, Error>;
}

impl TextRow for SimpleQueryRow {
    fn read_text(&self, column: &'static str) -> Result<&str, Error> {
        self.try_get(column)?.ok_or(Error::MissingValue { column })
    }

    fn parse_integer(&self, column: &'static str) -> Result<i64, Error> {
        let text = self.read_text(column)?;
        text.parse().map_err(|_| Error::MalformedValue {
            column,
            value: text.to_string(),
        })
    }

    /// A pivot server renders a `TIMESTAMP WITH TIME ZONE` the way a
    /// PostgreSQL server does: `2026-11-17 19:00:41.123456+00`, the fraction
    /// present only when it is not zero.
    fn parse_instant_ms(&self, column: &'static str) -> Result<i64, Error> {
        let text = self.read_text(column)?;
        DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f%#z")
            .map(|instant| instant.timestamp_millis())
            .map_err(|_| Error::MalformedValue {
                column,
                value: text.to_string(),
            })
    }
}

/// One row of `system.connections`. Instants are milliseconds since the Unix
/// epoch, which is what the page's clock counts in.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Connection {
    id: i64,
    user_name: String,
    application_name: String,
    client_address: String,
    connected_at_ms: i64,
    state: String,
    state_changed_at_ms: i64,
    statement: String,
    copy_bytes: i64,
}

impl Connection {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        Ok(Self {
            id: row.parse_integer("id")?,
            user_name: row.read_text("user_name")?.to_string(),
            application_name: row.read_text("application_name")?.to_string(),
            client_address: row.read_text("client_address")?.to_string(),
            connected_at_ms: row.parse_instant_ms("connected_at")?,
            state: row.read_text("state")?.to_string(),
            state_changed_at_ms: row.parse_instant_ms("state_changed_at")?,
            statement: row.read_text("statement")?.to_string(),
            copy_bytes: row.parse_integer("copy_bytes")?,
        })
    }
}

/// The server's connections as they were at one moment.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionsReading {
    /// When the connections were read, by this backend's clock: what the
    /// instants in `connections` are measured against.
    read_at_ms: i64,
    /// The pivot server the connections belong to, as `host:port`.
    pivot_address: String,
    connections: Vec<Connection>,
}

/// The blocks of the buffer pool that are in one state on one NUMA node:
/// `system.memory_blocks`, grouped.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryUse {
    node: i64,
    /// `free`, `pinned`, or the name of the cache holding the blocks.
    state: String,
    block_count: i64,
    bytes: i64,
}

impl MemoryUse {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        Ok(Self {
            node: row.parse_integer("node")?,
            state: row.read_text("state")?.to_string(),
            block_count: row.parse_integer("block_count")?,
            bytes: row.parse_integer("bytes")?,
        })
    }
}

/// How the server's buffer pool was used at one moment.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryReading {
    /// One entry per NUMA node and state that has blocks, by node then state.
    usage: Vec<MemoryUse>,
}

/// The pivot server this backend describes, and the session it keeps open to
/// it.
pub struct PivotServer {
    host: String,
    port: u16,
    user: String,
    password: Option<String>,
    /// Opened on the first read and reopened after the server drops it, so the
    /// backend can start before the server and outlive its restarts.
    session: Mutex<Option<Client>>,
}

impl PivotServer {
    pub fn new(host: String, port: u16, user: String, password: Option<String>) -> Self {
        Self {
            host,
            port,
            user,
            password,
            session: Mutex::new(None),
        }
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub async fn read_connections(&self) -> Result<ConnectionsReading, Error> {
        let rows = self
            .run_query(&format!(
                "SELECT id, user_name, application_name, client_address, connected_at, state, \
                 state_changed_at, statement, copy_bytes \
                 FROM system.connections WHERE application_name <> '{APPLICATION_NAME}' \
                 ORDER BY id"
            ))
            .await?;
        let connections = rows
            .iter()
            .map(Connection::parse_row)
            .collect::<Result<_, _>>()?;

        let read_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the system clock reads a time after 1970");
        Ok(ConnectionsReading {
            read_at_ms: read_at.as_millis() as i64,
            pivot_address: self.address(),
            connections,
        })
    }

    /// The pool is grouped by the server, so a pool of any size comes back as
    /// a handful of rows.
    pub async fn read_memory(&self) -> Result<MemoryReading, Error> {
        let rows = self
            .run_query(
                "SELECT node, state, count(*) AS block_count, sum(bytes) AS bytes \
                 FROM system.memory_blocks GROUP BY node, state ORDER BY node, state",
            )
            .await?;
        let usage = rows
            .iter()
            .map(MemoryUse::parse_row)
            .collect::<Result<_, _>>()?;
        Ok(MemoryReading { usage })
    }

    /// Run `sql` on the session, opening one first if there is none to use,
    /// and return the rows it produced.
    async fn run_query(&self, sql: &str) -> Result<Vec<SimpleQueryRow>, Error> {
        let mut session = self.session.lock().await;
        let client = match session.as_ref() {
            Some(client) if !client.is_closed() => client,
            _ => session.insert(self.open_session().await?),
        };
        let messages = client.simple_query(sql).await?;
        Ok(messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(row),
                _ => None,
            })
            .collect())
    }

    async fn open_session(&self) -> Result<Client, Error> {
        let mut config = tokio_postgres::Config::new();
        config
            .host(&self.host)
            .port(self.port)
            .user(&self.user)
            .application_name(APPLICATION_NAME);
        if let Some(password) = &self.password {
            config.password(password);
        }
        let (client, connection) = config.connect(NoTls).await?;
        info!(pivot = %self.address(), user = %self.user, "connected to the pivot server");
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                warn!(%error, "the session with the pivot server ended");
            }
        });
        Ok(client)
    }
}
