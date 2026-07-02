//! `GET /api/overview`: a single poll-friendly snapshot of the whole engine -
//! every table (metadata + live row count and insert rate), compaction
//! counters, and system metrics.

use arrow::util::display::{ArrayFormatter, FormatOptions};
use axum::Json;
use axum::extract::State;
use catalog::ParquetCatalog;
use serde::Serialize;

use super::IntrospectState;
use super::json::ColumnOut;
use super::system::{SystemInfo, collect_system};

#[derive(Serialize)]
pub(super) struct Overview {
    store: String,
    tables: Vec<TableOut>,
    compaction: Option<CompactionOut>,
    system: SystemInfo,
    connected: bool,
}

#[derive(Serialize)]
struct TableOut {
    name: String,
    location: String,
    columns: Vec<ColumnOut>,
    file_count: usize,
    total_bytes: u64,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    row_count: Option<i64>,
    rows_per_sec: Option<f64>,
    /// This table's compaction counters, if a compacter is running.
    compaction: Option<CompactionOut>,
}

#[derive(Serialize, Default)]
struct CompactionOut {
    compactions: u64,
    files_merged_in: u64,
    files_written: u64,
    bytes_written: u64,
    last_run_unix_ms: u64,
}

fn compaction_out(stats: &compact::CompactStatsSnapshot) -> CompactionOut {
    CompactionOut {
        compactions: stats.compactions,
        files_merged_in: stats.files_merged_in,
        files_written: stats.files_written,
        bytes_written: stats.bytes_written,
        last_run_unix_ms: stats.last_run_unix_ms,
    }
}

/// Table metadata gathered from the catalog (no row count / rate yet).
struct TableMeta {
    name: String,
    location: String,
    columns: Vec<ColumnOut>,
    file_count: usize,
    total_bytes: u64,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
}

pub(super) async fn overview(State(state): State<IntrospectState>) -> Json<Overview> {
    let store = state.catalog.store_description();

    // Catalog snapshot (manifest reads may touch object storage) off the runtime.
    let metas = {
        let catalog = state.catalog.clone();
        tokio::task::spawn_blocking(move || collect_tables(&catalog))
            .await
            .unwrap_or_default()
    };

    let compaction_snap = state.compacter.as_ref().map(|c| c.snapshot());

    let mut tables = Vec::with_capacity(metas.len());
    for meta in metas {
        let row_count = count_rows(&state, &meta.name).await;
        // Insert rate from the row-count deltas between polls.
        let rows_per_sec =
            row_count.and_then(|rows| state.record_rate(&meta.name, rows.max(0) as u64));
        let compaction = compaction_snap
            .as_ref()
            .and_then(|c| c.per_table.get(&meta.name))
            .map(compaction_out);
        tables.push(TableOut {
            name: meta.name,
            location: meta.location,
            columns: meta.columns,
            file_count: meta.file_count,
            total_bytes: meta.total_bytes,
            partition_by: meta.partition_by,
            sort_by: meta.sort_by,
            row_count,
            rows_per_sec,
            compaction,
        });
    }

    // Roll the per-table counters up for the summary card; "last run" is the
    // last full sweep.
    let compaction = compaction_snap.as_ref().map(|snap| {
        let mut out = CompactionOut::default();
        for stats in snap.per_table.values() {
            out.compactions += stats.compactions;
            out.files_merged_in += stats.files_merged_in;
            out.files_written += stats.files_written;
            out.bytes_written += stats.bytes_written;
        }
        out.last_run_unix_ms = snap.last_sweep_unix_ms;
        out
    });

    Json(Overview {
        store,
        tables,
        compaction,
        system: collect_system(&state.system, state.pid),
        connected: true,
    })
}

fn collect_tables(catalog: &ParquetCatalog) -> Vec<TableMeta> {
    catalog
        .tables()
        .into_iter()
        .map(|table| {
            let name = table.name().to_string();
            let columns = table
                .columns()
                .into_iter()
                .map(|c| ColumnOut {
                    name: c.name,
                    col_type: c.col_type.to_string(),
                })
                .collect();
            // Count + total size only; the file *list* is paginated separately
            // (`/api/tables/{name}/files`) so the polled overview stays small
            // even for a table with thousands of files.
            let files = catalog.table_files(&name).unwrap_or_default();
            let total_bytes = files.iter().map(|f| f.size).sum();
            let file_count = files.len();
            TableMeta {
                name,
                location: table.location().to_string(),
                columns,
                file_count,
                total_bytes,
                partition_by: table.partition_by().to_vec(),
                sort_by: table.sort_by().to_vec(),
            }
        })
        .collect()
}

/// Unfiltered `COUNT(*)` for one table - answered from Parquet footers, so it's
/// cheap to poll. `None` if the query fails (e.g. the table was just dropped).
async fn count_rows(state: &IntrospectState, table: &str) -> Option<i64> {
    let sql = format!("SELECT COUNT(*) FROM \"{}\"", table.replace('"', "\"\""));
    let batches =
        crate::query_handler::execute_sql(state.catalog_dyn.clone(), state.dispatcher.clone(), sql)
            .await
            .ok()?;
    let batch = batches.first()?;
    if batch.num_rows() == 0 {
        return None;
    }
    let opts = FormatOptions::default();
    let formatter = ArrayFormatter::try_new(batch.column(0).as_ref(), &opts).ok()?;
    formatter.value(0).to_string().parse::<i64>().ok()
}
