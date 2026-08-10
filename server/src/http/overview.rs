//! `GET /api/overview`: a single poll-friendly snapshot of the default Delta
//! datastore: every table (metadata + live row count) and system metrics.

use arrow::util::display::{ArrayFormatter, FormatOptions};
use axum::Json;
use axum::extract::State;
use datastore_delta::DeltaDatastore;
use planner::catalog::SchemaQualifiedTableName;
use serde::Serialize;

use super::IntrospectState;
use super::json::ColumnOut;
use super::system::{SystemInfo, collect_system};

#[derive(Serialize)]
pub(super) struct Overview {
    store: String,
    tables: Vec<TableOut>,
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
}

/// BoundTable metadata gathered from the catalog (no row count yet).
struct TableMeta {
    name: SchemaQualifiedTableName,
    location: String,
    columns: Vec<ColumnOut>,
    file_count: usize,
    total_bytes: u64,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
}

pub(super) async fn overview(State(state): State<IntrospectState>) -> Json<Overview> {
    // The default datastore always exists (`PivotCatalog` requires it). This
    // overview reads Parquet-level detail, so recover the concrete backend; the
    // dashboard serves only Delta datastores.
    let datastore = state
        .catalog
        .default_datastore()
        .clone()
        .into_any_arc()
        .downcast::<DeltaDatastore>()
        .expect("the dashboard serves only Delta datastores");
    let store = datastore.store_description();

    // The datastore's accessors are async and hop to the blocking pool for
    // their own store reads, so the snapshot is gathered right here.
    let metas = collect_tables(&datastore).await;

    let mut tables = Vec::with_capacity(metas.len());
    for meta in metas {
        let row_count = count_rows(&state, &meta.name).await;
        tables.push(TableOut {
            name: format_table_label(&meta.name),
            location: meta.location,
            columns: meta.columns,
            file_count: meta.file_count,
            total_bytes: meta.total_bytes,
            partition_by: meta.partition_by,
            sort_by: meta.sort_by,
            row_count,
        });
    }

    Json(Overview {
        store,
        tables,
        system: collect_system(&state.system, state.pid),
        connected: true,
    })
}

async fn collect_tables(datastore: &DeltaDatastore) -> Vec<TableMeta> {
    let mut metas = Vec::new();
    for (name, table) in datastore.tables().await {
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
        let files = datastore.table_files(&name).await.unwrap_or_default();
        let total_bytes = files.iter().map(|f| f.size).sum();
        let file_count = files.len();
        metas.push(TableMeta {
            name,
            location: table.location().to_string(),
            columns,
            file_count,
            total_bytes,
            partition_by: table.partition_by().to_vec(),
            sort_by: table.sort_by().to_vec(),
        });
    }
    metas
}

/// How a table is labelled in the dashboard: bare in the default schema, where
/// the qualifier would carry no information, and schema-qualified elsewhere so
/// equal table names stay distinguishable.
fn format_table_label(name: &SchemaQualifiedTableName) -> String {
    if name.schema == planner::DEFAULT_SCHEMA_NAME {
        name.table.clone()
    } else {
        name.to_string()
    }
}

/// Escape an identifier for embedding in a double-quoted SQL name. Doubles the
/// quotes within it; the caller supplies the surrounding pair.
fn escape_identifier_quotes(name: &str) -> String {
    name.replace('"', "\"\"")
}

/// Unfiltered `COUNT(*)` for one table - answered from Parquet footers, so it's
/// cheap to poll. `None` if the query fails (e.g. the table was just dropped).
async fn count_rows(state: &IntrospectState, table: &SchemaQualifiedTableName) -> Option<i64> {
    let sql = format!(
        "SELECT COUNT(*) FROM \"{}\".\"{}\"",
        escape_identifier_quotes(&table.schema),
        escape_identifier_quotes(&table.table)
    );
    let batches = crate::query_handler::execute_sql(
        state.catalog.clone(),
        state.dispatcher.clone(),
        state.plan_cache.clone(),
        sql,
    )
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
