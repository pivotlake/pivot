//! `POST /api/query`: run an arbitrary SQL string from the console on the same
//! planner + dispatch pool as every other query and return the result as JSON.
//! This is the only endpoint that is not read-only.

use std::time::Instant;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use super::IntrospectState;
use super::json::{ColumnOut, batches_to_json};

#[derive(Deserialize)]
pub(super) struct QueryRequest {
    sql: String,
}

#[derive(Serialize, Default)]
pub(super) struct QueryResponse {
    columns: Vec<ColumnOut>,
    rows: Vec<Vec<Option<String>>>,
    row_count: usize,
    elapsed_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub(super) async fn query(
    State(state): State<IntrospectState>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let started = Instant::now();
    let result = crate::query_handler::execute_sql(
        state.catalog.clone(),
        state.dispatcher.clone(),
        state.plan_cache.clone(),
        req.sql,
    )
    .await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    Json(match result {
        Ok(batches) => {
            let (columns, rows) = batches_to_json(&batches);
            QueryResponse {
                row_count: rows.len(),
                columns,
                rows,
                elapsed_ms,
                error: None,
            }
        }
        Err(e) => QueryResponse {
            error: Some(e),
            elapsed_ms,
            ..Default::default()
        },
    })
}
