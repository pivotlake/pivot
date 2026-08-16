//! Dataflow compilation, result collection, and command classification.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int64Array, RecordBatch};
use dispatch::{CancelToken, DataFlowHandle, DataFlowStats};

use super::{Error, Result, ResultColumn};

pub(super) enum StatementHandle<T> {
    Batches(DataFlowHandle<T>),
    Insert(DataFlowHandle<std::result::Result<usize, String>>),
}

impl<T> StatementHandle<T> {
    fn cancel_token(&self) -> CancelToken {
        match self {
            Self::Batches(handle) => handle.cancel_token(),
            Self::Insert(handle) => handle.cancel_token(),
        }
    }

    fn collect_with_stats(
        self,
    ) -> std::result::Result<(StatementResults<T>, DataFlowStats), dispatch::DataFlowError> {
        match self {
            Self::Batches(handle) => handle
                .collect_with_stats()
                .map(|(batches, stats)| (StatementResults::Batches(batches), stats)),
            Self::Insert(handle) => handle
                .collect_with_stats()
                .map(|(outputs, stats)| (StatementResults::Insert(outputs), stats)),
        }
    }
}

pub(super) enum StatementResults<T> {
    Batches(Vec<T>),
    Insert(Vec<std::result::Result<usize, String>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StatementKind {
    Query,
    Insert,
    CreateTable,
    CreateSchema,
    CreateUser,
    DropUser,
    DropTable,
    DropSchema,
}

impl StatementKind {
    pub(super) fn from_plan(plan: &planner::Plan) -> Self {
        match &plan.root.operator {
            planner::Operator::Insert(_) => Self::Insert,
            planner::Operator::CreateTable(_) => Self::CreateTable,
            planner::Operator::CreateSchema(_) => Self::CreateSchema,
            planner::Operator::CreateUser(_) => Self::CreateUser,
            planner::Operator::DropUser(_) => Self::DropUser,
            planner::Operator::DropTable(_) => Self::DropTable,
            planner::Operator::DropSchema(_) => Self::DropSchema,
            _ => Self::Query,
        }
    }
}

pub(super) fn result_columns(plan: &planner::Plan) -> Result<Vec<ResultColumn>> {
    let types = plan.root.output_types()?;
    Ok(types
        .iter()
        .enumerate()
        .map(|(index, pivot_type)| ResultColumn {
            name: plan
                .output_names
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("column{}", index + 1)),
            data_type: planner::types::physical_arrow_type(pivot_type),
        })
        .collect())
}

pub(super) fn affected_rows(outputs: Vec<std::result::Result<usize, String>>) -> Result<usize> {
    if outputs.len() != 1 {
        return Err(Error::InvalidInsertResult(format!(
            "expected one affected-row output, got {}",
            outputs.len()
        )));
    }
    outputs
        .into_iter()
        .next()
        .expect("output length checked above")
        .map_err(Error::InvalidInsertResult)
}

pub(super) fn affected_rows_from_record_batch(
    batch: RecordBatch,
) -> std::result::Result<usize, String> {
    if batch.num_rows() != 1 || batch.num_columns() != 1 {
        return Err(format!(
            "expected one row and one column, got {} rows and {} columns",
            batch.num_rows(),
            batch.num_columns()
        ));
    }
    let counts = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("expected Int64, got {}", batch.column(0).data_type()))?;
    if counts.is_null(0) {
        return Err("count is null".to_string());
    }
    usize::try_from(counts.value(0)).map_err(|_| "count is negative or too large".to_string())
}

pub(super) async fn collect_handle<T>(
    handle: StatementHandle<T>,
) -> Result<(StatementResults<T>, DataFlowStats, Duration)>
where
    T: Send + 'static,
{
    let guard = CancelOnDrop::new(handle.cancel_token());
    let started = Instant::now();
    let (outputs, flow) = tokio::task::spawn_blocking(move || handle.collect_with_stats())
        .await
        .map_err(Error::WorkerPanic)??;
    let elapsed = started.elapsed();
    guard.defuse();
    Ok((outputs, flow, elapsed))
}

pub(super) async fn execute_compact(
    catalog: &Arc<catalog::PivotCatalog>,
    transaction: &dyn planner::catalog::CatalogTransaction,
    request: &planner::Compact,
) -> Result<u64> {
    let datastore_name = request
        .datastore
        .as_deref()
        .unwrap_or_else(|| catalog.default_datastore_name());
    let table = planner::catalog::SchemaQualifiedTableName::new(
        request
            .schema
            .as_deref()
            .unwrap_or(planner::DEFAULT_SCHEMA_NAME),
        request.table.as_str(),
    );
    Ok(transaction
        .compact(datastore_name, &table, request.final_sweep)
        .await?)
}

struct CancelOnDrop {
    token: Option<CancelToken>,
}

impl CancelOnDrop {
    fn new(token: CancelToken) -> Self {
        Self { token: Some(token) }
    }

    fn defuse(mut self) {
        self.token.take();
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            token.cancel();
        }
    }
}
