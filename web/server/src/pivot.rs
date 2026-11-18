//! Reading a pivot server's system tables.

use std::net::IpAddr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use chrono::DateTime;
use futures_util::TryStreamExt;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls, SimpleQueryMessage, SimpleQueryRow};
use tracing::{info, warn};

/// What this backend calls itself when it connects. Its own connection is
/// reported with the rest and marked, so the page can tell the connection it
/// is read through from the server's clients.
const APPLICATION_NAME: &str = "pivot-web";

/// What the session that runs the page's own statements calls itself. A
/// separate session, so a slow statement does not hold up the readings the
/// other pages are drawn from, and under a name of its own, so the server
/// page shows it as the client it is, running the statement.
const QUERY_APPLICATION_NAME: &str = "pivot-web query";

/// How many rows of a statement's result are handed to the page. A result
/// can be any size, and the page is for looking at one, not for exporting it.
const QUERY_ROW_LIMIT: usize = 1000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{}", Error::describe_postgres(.0))]
    Postgres(#[from] tokio_postgres::Error),
    #[error("the server returned no value for `{column}`")]
    MissingValue { column: &'static str },
    #[error("the server returned `{value}` for `{column}`, which does not parse")]
    MalformedValue { column: &'static str, value: String },
    #[error("`{0}` is not a table id")]
    NotATableId(String),
}

impl Error {
    /// What the pivot server said was wrong with a statement, when that is
    /// what the error is: the statement's fault, not the connection's.
    pub fn read_statement_error(&self) -> Option<&str> {
        match self {
            Error::Postgres(error) => error.as_db_error().map(|refused| refused.message()),
            _ => None,
        }
    }
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

    /// The names of a comma-separated list, none for an empty string.
    fn parse_name_list(&self, column: &'static str) -> Result<Vec<String>, Error>;
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

    fn parse_name_list(&self, column: &'static str) -> Result<Vec<String>, Error> {
        Ok(self
            .read_text(column)?
            .split(',')
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect())
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
    /// Whether the connection is a web backend's, this one's or another's:
    /// one that calls itself what this backend does.
    is_web_backend: bool,
}

impl Connection {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        let application_name = row.read_text("application_name")?;
        Ok(Self {
            id: row.parse_integer("id")?,
            user_name: row.read_text("user_name")?.to_string(),
            application_name: application_name.to_string(),
            client_address: row.read_text("client_address")?.to_string(),
            connected_at_ms: row.parse_instant_ms("connected_at")?,
            state: row.read_text("state")?.to_string(),
            state_changed_at_ms: row.parse_instant_ms("state_changed_at")?,
            statement: row.read_text("statement")?.to_string(),
            copy_bytes: row.parse_integer("copy_bytes")?,
            is_web_backend: application_name == APPLICATION_NAME,
        })
    }
}

/// The machine a pivot server runs on. A server does not report it, so it is
/// known only when the server is on this backend's own machine.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Machine {
    hostname: String,
    cores: usize,
    memory_bytes: u64,
    architecture: &'static str,
    operating_system: &'static str,
}

impl Machine {
    /// This machine, if `pivot_host` names it: a loopback address reaches
    /// nothing else.
    fn read_if_local(pivot_host: &str) -> Option<Self> {
        let is_loopback = pivot_host == "localhost"
            || pivot_host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback());
        if !is_loopback {
            return None;
        }
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        Some(Self {
            hostname: sysinfo::System::host_name()
                .expect("the operating system names this machine"),
            cores: std::thread::available_parallelism()
                .expect("the operating system counts this machine's cores")
                .get(),
            memory_bytes: system.total_memory(),
            architecture: std::env::consts::ARCH,
            operating_system: std::env::consts::OS,
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
    /// The host of that address alone, which names the server when its
    /// machine is not known.
    pivot_host: String,
    /// The machine the server runs on, when it is this backend's.
    machine: Option<Machine>,
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

/// One row of `system.tables`, with how many files `system.table_files`
/// lists for the table.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Table {
    datastore: String,
    schema: String,
    name: String,
    id: String,
    /// The table's sort columns, in order.
    sorting_keys: Vec<String>,
    /// The table's partition columns, in order.
    partition_key: Vec<String>,
    total_rows: i64,
    bytes: i64,
    bytes_uncompressed: i64,
    file_count: i64,
}

impl Table {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        Ok(Self {
            datastore: row.read_text("datastore")?.to_string(),
            schema: row.read_text("schema")?.to_string(),
            name: row.read_text("name")?.to_string(),
            id: row.read_text("id")?.to_string(),
            sorting_keys: row.parse_name_list("sorting_keys")?,
            partition_key: row.parse_name_list("partition_key")?,
            total_rows: row.parse_integer("total_rows")?,
            bytes: row.parse_integer("bytes")?,
            bytes_uncompressed: row.parse_integer("bytes_uncompressed")?,
            // Counted by the files reading, which fills it in.
            file_count: 0,
        })
    }
}

/// The server's tables, the `system` datastore's own left out.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TablesReading {
    tables: Vec<Table>,
}

/// One row of `system.table_files`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TableFile {
    path: String,
    /// `column=value` pairs, or empty for an unpartitioned table.
    partition: String,
    bytes: i64,
    bytes_uncompressed: i64,
    /// The file's bounds on every column that records them: an object keyed
    /// by column name whose values each hold `min` and `max`.
    bounds: serde_json::Value,
}

impl TableFile {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        let bounds = row.read_text("min_max_stats")?;
        Ok(Self {
            path: row.read_text("path")?.to_string(),
            partition: row.read_text("partition")?.to_string(),
            bytes: row.parse_integer("bytes")?,
            bytes_uncompressed: row.parse_integer("bytes_uncompressed")?,
            bounds: serde_json::from_str(bounds).map_err(|_| Error::MalformedValue {
                column: "min_max_stats",
                value: bounds.to_string(),
            })?,
        })
    }
}

/// The sizes the pivot server's compacter selects merges by, which the page
/// needs to tell which of a table's files a routine round would merge. A
/// server does not report its datastores' settings, so they are this
/// backend's to be told.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
    /// `compact_bytes`: the size a merged file is cut at. A file below half
    /// of it is a small file.
    pub target_bytes: u64,
    /// `compact_merge_bytes`: the total a pile of small files merges at.
    pub merge_target_bytes: u64,
    /// `compact_min_files`: the count at which a pile below that total may
    /// merge anyway.
    pub min_files: u64,
    /// How many mutually overlapping files one layout merge rewrites.
    pub layout_clique_size: u64,
}

/// One table's data files as they were at one moment.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TableFilesReading {
    files: Vec<TableFile>,
    compaction: CompactionSettings,
}

/// One row of `system.columns`, with the table the column belongs to named
/// the way a statement names it.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TableColumn {
    datastore: String,
    schema: String,
    table: String,
    name: String,
    /// The column's SQL type, such as `BIGINT`.
    sql_type: String,
}

impl TableColumn {
    fn parse_row(row: &SimpleQueryRow) -> Result<Self, Error> {
        Ok(Self {
            datastore: row.read_text("datastore")?.to_string(),
            schema: row.read_text("schema")?.to_string(),
            table: row.read_text("table_name")?.to_string(),
            name: row.read_text("column_name")?.to_string(),
            sql_type: row.read_text("type")?.to_string(),
        })
    }
}

/// The columns of every table the server has, its own `system` tables among
/// them, each table's in their declared order.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnsReading {
    columns: Vec<TableColumn>,
}

/// What a statement run for the page produced.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatementResult {
    /// The result's column names, none for a statement that returns no rows.
    columns: Vec<String>,
    /// The result's rows as the server renders them in text, up to the row
    /// limit, a null being a null.
    rows: Vec<Vec<Option<String>>>,
    /// Whether the result had more rows than are in `rows`.
    is_cut_short: bool,
    /// How long the statement took, from sending it to its last row read.
    elapsed_ms: u64,
}

/// The pivot server this backend describes, and the sessions it keeps open
/// to it.
pub struct PivotServer {
    host: String,
    port: u16,
    user: String,
    password: Option<String>,
    compaction: CompactionSettings,
    /// The machine the pivot server runs on, read once: it does not change
    /// while this backend runs.
    machine: Option<Machine>,
    /// Opened on the first read and reopened after the server drops it, so the
    /// backend can start before the server and outlive its restarts.
    session: Mutex<Option<Client>>,
    /// The session the page's own statements run on, opened by the first of
    /// them.
    query_session: Mutex<Option<Client>>,
}

impl PivotServer {
    pub fn new(
        host: String,
        port: u16,
        user: String,
        password: Option<String>,
        compaction: CompactionSettings,
    ) -> Self {
        Self {
            machine: Machine::read_if_local(&host),
            host,
            port,
            user,
            password,
            compaction,
            session: Mutex::new(None),
            query_session: Mutex::new(None),
        }
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub async fn read_connections(&self) -> Result<ConnectionsReading, Error> {
        let rows = self
            .run_query(
                "SELECT id, user_name, application_name, client_address, connected_at, state, \
                 state_changed_at, statement, copy_bytes \
                 FROM system.connections ORDER BY id",
            )
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
            pivot_host: self.host.clone(),
            machine: self.machine.clone(),
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

    pub async fn read_tables(&self) -> Result<TablesReading, Error> {
        let table_rows = self
            .run_query(
                "SELECT datastore, schema, name, id, sorting_keys, partition_key, total_rows, \
                 bytes, bytes_uncompressed \
                 FROM system.tables WHERE datastore <> 'system' ORDER BY datastore, schema, name",
            )
            .await?;
        let mut tables = table_rows
            .iter()
            .map(Table::parse_row)
            .collect::<Result<Vec<_>, _>>()?;

        let count_rows = self
            .run_query(
                "SELECT table_id, count(*) AS file_count FROM system.table_files GROUP BY table_id",
            )
            .await?;
        for row in &count_rows {
            let table_id = row.read_text("table_id")?;
            // A table created between the two queries is counted but not listed.
            if let Some(table) = tables.iter_mut().find(|table| table.id == table_id) {
                table.file_count = row.parse_integer("file_count")?;
            }
        }
        Ok(TablesReading { tables })
    }

    pub async fn read_columns(&self) -> Result<ColumnsReading, Error> {
        let rows = self
            .run_query(
                "SELECT t.datastore, t.schema, t.name AS table_name, c.name AS column_name, c.type \
                 FROM system.columns c JOIN system.tables t ON c.table_id = t.id \
                 ORDER BY t.datastore, t.schema, t.name, c.position",
            )
            .await?;
        let columns = rows
            .iter()
            .map(TableColumn::parse_row)
            .collect::<Result<_, _>>()?;
        Ok(ColumnsReading { columns })
    }

    /// The files of the table whose `system.tables.id` is `table_id`.
    pub async fn read_table_files(&self, table_id: &str) -> Result<TableFilesReading, Error> {
        // The id goes into the statement's text, so it is held to the
        // characters an id is made of.
        let is_table_id = !table_id.is_empty()
            && table_id
                .chars()
                .all(|character| character.is_ascii_hexdigit() || character == '-');
        if !is_table_id {
            return Err(Error::NotATableId(table_id.to_string()));
        }
        let rows = self
            .run_query(&format!(
                "SELECT path, partition, bytes, bytes_uncompressed, min_max_stats \
                 FROM system.table_files WHERE table_id = '{table_id}'"
            ))
            .await?;
        let files = rows
            .iter()
            .map(TableFile::parse_row)
            .collect::<Result<_, _>>()?;
        Ok(TableFilesReading {
            files,
            compaction: self.compaction,
        })
    }

    /// Run `sql`, a statement the page's user wrote, and return what it
    /// produced, its rows up to the row limit. One statement runs at a time.
    pub async fn run_statement(&self, sql: &str) -> Result<StatementResult, Error> {
        let mut session = self.query_session.lock().await;
        let client = match session.as_ref() {
            Some(client) if !client.is_closed() => client,
            _ => session.insert(self.open_session(QUERY_APPLICATION_NAME).await?),
        };

        let started = Instant::now();
        let messages = client.simple_query_raw(sql).await?;
        tokio::pin!(messages);
        let mut result = StatementResult {
            columns: Vec::new(),
            rows: Vec::new(),
            is_cut_short: false,
            elapsed_ms: 0,
        };
        while let Some(message) = messages.try_next().await? {
            match message {
                SimpleQueryMessage::RowDescription(columns) => {
                    result.columns = columns
                        .iter()
                        .map(|column| column.name().to_string())
                        .collect();
                }
                SimpleQueryMessage::Row(row) => {
                    // The rest of the result is left unread: the session
                    // discards it as it arrives.
                    if result.rows.len() == QUERY_ROW_LIMIT {
                        result.is_cut_short = true;
                        break;
                    }
                    result.rows.push(
                        (0..row.len())
                            .map(|column| {
                                row.try_get(column).map(|value| value.map(str::to_string))
                            })
                            .collect::<Result<_, _>>()?,
                    );
                }
                _ => {}
            }
        }
        result.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Run `sql` on the session, opening one first if there is none to use,
    /// and return the rows it produced.
    async fn run_query(&self, sql: &str) -> Result<Vec<SimpleQueryRow>, Error> {
        let mut session = self.session.lock().await;
        let client = match session.as_ref() {
            Some(client) if !client.is_closed() => client,
            _ => session.insert(self.open_session(APPLICATION_NAME).await?),
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

    /// Log in to the pivot server under `application_name`.
    async fn open_session(&self, application_name: &str) -> Result<Client, Error> {
        let mut config = tokio_postgres::Config::new();
        config
            .host(&self.host)
            .port(self.port)
            .user(&self.user)
            .application_name(application_name);
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
