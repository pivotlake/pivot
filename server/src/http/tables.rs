//! Paginated per-table internals: the data-file list
//! (`/api/datastores/{datastore}/tables/{name}/files`) and the Parquet row-group
//! stats (`/api/tables/{name}/rowgroups`). The unqualified files and row-group
//! routes address the default datastore for compatibility. Both page so a table
//! with thousands of files/row groups streams to the UI a slice at a time
//! (infinite scroll).

use axum::Json;
use axum::extract::{Path, Query, State};
use datastore_delta::DeltaDatastore;
use planner::catalog::SchemaQualifiedTableName;
use serde::{Deserialize, Serialize};

use super::IntrospectState;
use super::json::{ColumnOut, batches_to_json};

#[derive(Deserialize)]
pub(super) struct Page {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Restrict row groups to a single file, by its (stable) path/name.
    #[serde(default)]
    file: Option<String>,
}

fn default_limit() -> usize {
    50
}

#[derive(Serialize)]
struct FileOut {
    path: String,
    size: u64,
}

#[derive(Serialize, Default)]
pub(super) struct FilesPage {
    items: Vec<FileOut>,
    total: usize,
}

/// One page of a table's data files (path + size), straight from the catalog.
/// Paginated so a table with thousands of files streams to the client a slice
/// at a time (infinite scroll) instead of all at once.
pub(super) async fn default_files_page(
    State(state): State<IntrospectState>,
    Path(name): Path<String>,
    Query(page): Query<Page>,
) -> Json<FilesPage> {
    let datastore = state.catalog.default_datastore_name().to_string();
    files_page_for(state, datastore, name, page).await
}

/// One page from an explicitly named datastore. The concrete backend owns the
/// format-specific file enumeration; this handler only selects and paginates.
pub(super) async fn files_page(
    State(state): State<IntrospectState>,
    Path((datastore, name)): Path<(String, String)>,
    Query(page): Query<Page>,
) -> Json<FilesPage> {
    files_page_for(state, datastore, name, page).await
}

async fn files_page_for(
    state: IntrospectState,
    datastore: String,
    name: String,
    page: Page,
) -> Json<FilesPage> {
    let Some(datastore) = state.catalog.get_datastore(&datastore).cloned() else {
        return Json(FilesPage::default());
    };
    // File enumeration is Parquet-specific; recover the concrete backend.
    let Ok(datastore) = datastore.into_any_arc().downcast::<DeltaDatastore>() else {
        return Json(FilesPage::default());
    };
    let limit = page.limit.min(500);
    let offset = page.offset;
    let result = tokio::task::spawn_blocking(move || {
        // The path segment is a table name, taken as written: this route
        // addresses the default schema only.
        let name = SchemaQualifiedTableName::in_default_schema(name);
        let Ok(Some(ordered)) = datastore.table_data_files(&name) else {
            return FilesPage::default();
        };
        let total = ordered.len();
        let items = ordered
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|file| FileOut {
                path: file.path,
                size: file.size,
            })
            .collect();
        FilesPage { items, total }
    })
    .await
    .unwrap_or_default();
    Json(result)
}

#[derive(Serialize, Default)]
pub(super) struct RowGroupsPage {
    columns: Vec<ColumnOut>,
    rows: Vec<Vec<Option<String>>>,
    has_more: bool,
}

/// One page of a table's Parquet row-group stats (via the `metadata()` table
/// function), ordered for stable paging. `has_more` is true when the page came
/// back full, so the client can keep scrolling.
pub(super) async fn rowgroups_page(
    State(state): State<IntrospectState>,
    Path(name): Path<String>,
    Query(page): Query<Page>,
) -> Json<RowGroupsPage> {
    let limit = page.limit.min(500);
    // Filter by the file's stable path (`file_name`), not a positional index, so
    // it stays correct while INSERT/compaction add and remove files. The
    // `file_name` column itself is not selected - it'd be the same on every row.
    let filter = page
        .file
        .as_ref()
        .map(|f| format!(" WHERE file_name = '{}'", f.replace('\'', "''")))
        .unwrap_or_default();
    let sql = format!(
        "SELECT file_index, row_group_index, num_rows, num_columns, compressed_bytes \
         FROM metadata('{}'){} ORDER BY file_index, row_group_index LIMIT {} OFFSET {}",
        name.replace('\'', "''"),
        filter,
        limit,
        page.offset,
    );
    match crate::query_handler::execute_sql(
        state.catalog.clone(),
        state.dispatcher.clone(),
        state.plan_cache.clone(),
        sql,
    )
    .await
    {
        Ok(batches) => {
            let (columns, rows) = batches_to_json(&batches);
            let has_more = rows.len() >= limit;
            Json(RowGroupsPage {
                columns,
                rows,
                has_more,
            })
        }
        Err(_) => Json(RowGroupsPage::default()),
    }
}
