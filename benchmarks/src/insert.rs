//! Insert-throughput runner: times the `INSERT ... VALUES` write path.
//!
//! The `clickbench-insert` suite measures ingestion instead of queries. Its
//! `setup.sql` declares the hits schema over the parquet under `--source` as
//! `hits`, and once more per batch size as an empty table to write into. Each
//! batch size is its own measurement: a streamed `SELECT` over `hits` hands out
//! that many rows at a time, and every iteration renders a batch as one
//! `INSERT INTO <target> VALUES (...), (...)` and times how long the server
//! takes to run it. So the numbers cover parsing a large multi-row statement,
//! building the batch, and committing the file it writes: what a client
//! streaming rows in over the wire actually pays.
//!
//! Reading the rows back out of the server keeps this crate free of any Parquet
//! reader of its own, and the timing excludes it: only the statement is timed.

use std::pin::Pin;
use std::thread::sleep;
use std::time::{Duration, Instant};

use futures_util::TryStreamExt;
use tokio_postgres::{Client, SimpleQueryMessage, SimpleQueryRow, SimpleQueryStream};

use crate::runner::{Error, QueryRun, Result, RunOptions};

/// The suite this runs for. Its `setup.sql` declares the tables named below, so
/// the two files are read together.
pub const INSERT_SUITE: &str = "clickbench-insert";

/// Table bound to `--source`. Read, untimed, to supply the rows to insert.
const SOURCE_TABLE: &str = "hits";

/// One measured batch size.
struct BatchSize {
    /// Recorded like a query ID, so the baseline, comparison table and `--show`
    /// treat each size as its own result: cold is its first statement, hot the
    /// mean of the rest.
    id: &'static str,
    rows: usize,
    /// Table this size's statements write into. Each size gets its own, so every
    /// one starts from an empty table and a fresh Delta log rather than on top
    /// of what the size before it wrote.
    table: &'static str,
}

/// The batch sizes the suite measures. A thousand-row statement and a
/// ten-thousand-row one load the write path differently enough (statement text
/// to parse, batch to build, rows per commit) that one number covering both
/// would hide which of them moved.
const BATCH_SIZES: [BatchSize; 3] = [
    BatchSize {
        id: "insert-1k",
        rows: 1_000,
        table: "hits_inserted_1k",
    },
    BatchSize {
        id: "insert-5k",
        rows: 5_000,
        table: "hits_inserted_5k",
    },
    BatchSize {
        id: "insert-10k",
        rows: 10_000,
        table: "hits_inserted_10k",
    },
];

/// The hits schema's `VARCHAR` columns, whose values go into the statement as
/// quoted literals. Every other column is numeric and takes its value bare.
/// Matched against the names the server sends in the row description, so a
/// column that stops matching the schema shows up as a type error on the first
/// statement rather than as a silently mistyped literal.
const TEXT_COLUMNS: [&str; 28] = [
    "Title",
    "URL",
    "Referer",
    "FlashMinor2",
    "UserAgentMinor",
    "MobilePhoneModel",
    "Params",
    "SearchPhrase",
    "PageCharset",
    "OriginalURL",
    "HitColor",
    "BrowserLanguage",
    "BrowserCountry",
    "SocialNetwork",
    "SocialAction",
    "SocialSourcePage",
    "ParamOrderID",
    "ParamCurrency",
    "OpenstatServiceName",
    "OpenstatCampaignID",
    "OpenstatAdID",
    "OpenstatSourceID",
    "UTMSource",
    "UTMMedium",
    "UTMCampaign",
    "UTMContent",
    "UTMTerm",
    "FromTag",
];

/// Which of a row's columns are quoted, by position, read off the row
/// description the source scan opens with.
fn quoting_mask(columns: &[tokio_postgres::SimpleColumn]) -> Vec<bool> {
    columns
        .iter()
        .map(|column| {
            TEXT_COLUMNS
                .iter()
                .any(|text| text.eq_ignore_ascii_case(column.name()))
        })
        .collect()
}

/// Take the next `batch_size` rows off the source scan, or fewer if it ends.
///
/// The rows arrive as one streamed `SELECT`, so successive batches are
/// successive stretches of the source with nothing repeated or skipped, and only
/// one batch is ever held in memory. Paging with `LIMIT`/`OFFSET` instead would
/// give neither: the scan is parallel and unordered, so each page is an
/// arbitrary set of rows that may overlap the page before it.
///
/// The row description leading the rows fills `quoted` the first time it
/// arrives.
async fn take_batch(
    rows: &mut Pin<Box<SimpleQueryStream>>,
    batch_size: usize,
    quoted: &mut Vec<bool>,
) -> Result<Vec<SimpleQueryRow>> {
    let mut batch = Vec::with_capacity(batch_size);
    while batch.len() < batch_size {
        match rows.try_next().await? {
            Some(SimpleQueryMessage::Row(row)) => batch.push(row),
            Some(SimpleQueryMessage::RowDescription(columns)) => *quoted = quoting_mask(&columns),
            Some(_) => continue,
            None => break,
        }
    }
    Ok(batch)
}

/// Render one batch as a single `INSERT INTO <table> VALUES (...), (...)`.
fn build_insert(table: &str, quoted: &[bool], rows: &[SimpleQueryRow]) -> String {
    let mut sql = format!("INSERT INTO {table} VALUES ");
    for (row_index, row) in rows.iter().enumerate() {
        if row_index > 0 {
            sql.push(',');
        }
        sql.push('(');
        for (column_index, quoted) in quoted.iter().enumerate() {
            if column_index > 0 {
                sql.push(',');
            }
            match row.get(column_index) {
                None => sql.push_str("NULL"),
                Some(value) if *quoted => {
                    sql.push('\'');
                    sql.push_str(&value.replace('\'', "''"));
                    sql.push('\'');
                }
                Some(value) => sql.push_str(value),
            }
        }
        sql.push(')');
    }
    sql
}

/// Read `table`'s row count over the wire.
async fn count_rows(client: &Client, id: &str, table: &str) -> Result<u64> {
    let sql = format!("SELECT count(*) FROM {table}");
    let messages = client.simple_query(&sql).await?;
    let count = messages.iter().find_map(|message| match message {
        SimpleQueryMessage::Row(row) => row.get(0).and_then(|value| value.parse::<u64>().ok()),
        _ => None,
    });
    count.ok_or_else(|| Error::NonRowMessage {
        id: id.to_string(),
        message: format!("`{sql}` returned no countable row"),
    })
}

/// Measure one batch size: `opts.iterations` statements of `size.rows` rows
/// each, walking through the source data, reporting the per-statement
/// wall-clock. The rows are verified to have landed before the timings are
/// reported, so a run that lost rows can't look like a fast one.
async fn run_batch_size(
    client: &Client,
    server_port: u16,
    size: &BatchSize,
    opts: &RunOptions,
) -> Result<QueryRun> {
    let mut iterations_ms = Vec::with_capacity(opts.iterations as usize);

    // The source scan gets a connection of its own: it stays open for this
    // size's statements, handing out one batch at a time, while the timed
    // statements go over `client`. It asks for exactly the rows the size
    // consumes, so a source far larger than the benchmark costs nothing to read.
    let wanted_rows = size.rows * opts.iterations as usize;
    let reader = crate::runner::connect(server_port).await?;
    let mut source_rows = Box::pin(
        reader
            .simple_query_raw(&format!("SELECT * FROM {SOURCE_TABLE} LIMIT {wanted_rows}"))
            .await?,
    );
    let mut quoted = Vec::new();

    println!(
        "=== Query {} === ({} rows per statement into {})",
        size.id, size.rows, size.table
    );
    for iteration in 0..opts.iterations {
        let batch = take_batch(&mut source_rows, size.rows, &mut quoted).await?;
        if batch.len() < size.rows {
            return Err(Error::SourceExhausted {
                table: SOURCE_TABLE.to_string(),
                read: iteration as usize * size.rows + batch.len(),
                wanted: wanted_rows,
            });
        }
        let sql = build_insert(size.table, &quoted, &batch);

        let start = Instant::now();
        client.simple_query(&sql).await?;
        let elapsed = start.elapsed().as_millis();

        iterations_ms.push(elapsed);
        println!(
            "[{}/{}] Query {} — {elapsed}ms",
            iteration + 1,
            opts.iterations,
            size.id
        );
        if let Some(ms) = opts.sleep_ms
            && iteration + 1 < opts.iterations
        {
            sleep(Duration::from_millis(ms));
        }
    }

    let expected = size.rows as u64 * u64::from(opts.iterations);
    let actual = count_rows(client, size.id, size.table).await?;
    if actual != expected {
        return Err(Error::InsertCountMismatch {
            table: size.table.to_string(),
            expected,
            actual,
        });
    }

    let total_ms: u128 = iterations_ms.iter().sum();
    println!(
        "inserted {expected} rows in {total_ms}ms ({:.0} rows/sec)",
        expected as f64 * 1000.0 / total_ms.max(1) as f64
    );

    Ok(QueryRun {
        id: size.id.to_string(),
        iterations_ms,
    })
}

/// Measure every batch size the suite ships, smallest first, honouring
/// `--query` the way a query suite does so a single size can be run alone.
pub async fn run_inserts(
    client: &Client,
    server_port: u16,
    opts: &RunOptions,
) -> Result<Vec<QueryRun>> {
    let mut runs = Vec::with_capacity(BATCH_SIZES.len());
    for size in &BATCH_SIZES {
        if let Some(filter) = &opts.query_filter
            && !filter.iter().any(|wanted| wanted == size.id)
        {
            continue;
        }
        runs.push(run_batch_size(client, server_port, size, opts).await?);
    }
    Ok(runs)
}
