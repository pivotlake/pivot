//! Paginated per-table internals: the data-file list (`/api/tables/{name}/files`)
//! and the Parquet row-group stats (`/api/tables/{name}/rowgroups`). Both page
//! so a table with thousands of files/row groups streams to the UI a slice at a
//! time (infinite scroll) instead of all at once.

use axum::Json;
use axum::extract::{Path, Query, State};
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
pub(super) async fn files_page(
    State(state): State<IntrospectState>,
    Path(name): Path<String>,
    Query(page): Query<Page>,
) -> Json<FilesPage> {
    let catalog = state.catalog.clone();
    let limit = page.limit.min(500);
    let offset = page.offset;
    let result = tokio::task::spawn_blocking(move || {
        let Some(mut table) = catalog.table_handle(&name) else {
            return FilesPage::default();
        };
        let _ = table.refresh();
        // `file_partitions` is in manifest order, which is the same order
        // `metadata()` assigns `file_index` - so a file's position here is its
        // `file_index`, letting the UI filter row groups by it. `file_refs`
        // supplies the sizes.
        let sizes: std::collections::HashMap<String, u64> = table
            .file_refs()
            .into_iter()
            .map(|f| (f.path.as_str().to_string(), f.size))
            .collect();
        let ordered = table.file_partitions();
        let total = ordered.len();
        let items = ordered
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(path, _)| {
                let path = path.as_str().to_string();
                let size = sizes.get(&path).copied().unwrap_or(0);
                FileOut { path, size }
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
    match crate::query_handler::execute_sql(state.catalogs.clone(), state.dispatcher.clone(), sql)
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
