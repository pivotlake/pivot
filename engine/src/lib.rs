//! Transport-neutral SQL execution for Pivot.
//!
//! [`Engine`] owns the planner cache and coordinates one statement transaction
//! across planning, dataflow execution, cancellation, and commit or rollback.
//! Frontends choose how query batches are converted before they leave a
//! dispatch worker. The PostgreSQL frontend turns them into wire rows, while a
//! local frontend can copy Arrow batches or turn cells into terminal text.

use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::ipc::reader::StreamDecoder;
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::DataType;
use bytes::Bytes;
use dispatch::{
    CancelToken, ChannelInputFull, ChannelInputSender, DataFlowHandle, DataFlowStats, OutputBatch,
};
use lru::LruCache;
use thiserror::Error;
use tokio::sync::Notify;
use tokio::task::JoinError;

thread_local! {
    /// One planner and its non-Send DuckDB context per blocking-pool thread.
    static PLANNER: RefCell<Option<planner::Planner>> = const { RefCell::new(None) };
}

const PLAN_CACHE_QUERY_CAPACITY: usize = 128;

/// Options that affect execution but not the statement's result.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecuteOptions {
    /// Collect the dispatch IO and CPU breakdown.
    pub collect_stats: bool,
    /// Mark the statement's dataflows for exclusive perf profiling. This is a
    /// no-op unless the crate is built with its `perf` feature.
    pub profile: bool,
}

/// One result column resolved during planning. This remains available for an
/// empty result, where no output batch exists to carry a schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultColumn {
    pub name: String,
    pub data_type: DataType,
}

/// A statement that completed without returning a row set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Insert { rows: usize },
    CreateTable,
    DropTable,
    CreateSchema,
    CreateUser,
    Compact,
}

impl Command {
    /// PostgreSQL-style command-completion text.
    pub fn tag(&self) -> String {
        match self {
            Self::Insert { rows } => format!("INSERT 0 {rows}"),
            Self::CreateTable => "CREATE TABLE".to_string(),
            Self::DropTable => "DROP TABLE".to_string(),
            Self::CreateSchema => "CREATE SCHEMA".to_string(),
            Self::CreateUser => "CREATE USER".to_string(),
            Self::Compact => "COMPACT".to_string(),
        }
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tag())
    }
}

/// Semantic output from one statement. `T` is the frontend-selected material
/// produced for each query batch on a dispatch worker.
#[derive(Debug)]
pub enum StatementOutput<T> {
    Rows {
        columns: Vec<ResultColumn>,
        batches: Vec<T>,
    },
    Command(Command),
    /// A parsed `SET` or `RESET`. Frontends own session variables, so the
    /// engine returns the name and optional value rather than applying it.
    Set {
        name: String,
        value: Option<String>,
    },
    /// A `COPY ... FROM STDIN` with its ingest dataflow already running: the
    /// rows arrive over the frontend's own protocol, so the frontend feeds
    /// [`CopyIngest::push`] and then either [`CopyIngest::finish`]es (which
    /// commits) or drops the ingest (which aborts and rolls back). A frontend
    /// with no copy-in channel just drops it.
    CopyFromStdin(Box<CopyIngest>),
}

/// Phase timings and aggregate dataflow work for one statement.
#[derive(Clone, Debug, Default)]
pub struct ExecutionStats {
    pub plan: Duration,
    pub compile: Duration,
    pub execute: Duration,
    pub flow: DataFlowStats,
}

impl ExecutionStats {
    /// Render the server's compact `pivot_stats` notice.
    pub fn summary(&self) -> String {
        let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        format!(
            "stats: plan={:.1}ms compile={:.1}ms exec={:.1}ms | \
             disk={} ops/{:.1}MiB/read={:.1}ms/write={:.1}ms  \
             http={} ops/{:.1}MiB/get={:.1}ms/upload={:.1}ms  \
             http-disk-cache={} ops/{:.1}MiB/{:.1}ms  cpu={:.1}ms",
            ms(self.plan),
            ms(self.compile),
            ms(self.execute),
            self.flow.disk_requests,
            mib(self.flow.disk_bytes),
            ms(self.flow.disk_read_time),
            ms(self.flow.disk_write_time),
            self.flow.http_requests,
            mib(self.flow.http_bytes),
            ms(self.flow.http_get_time),
            ms(self.flow.http_upload_time),
            self.flow.disk_cache_requests,
            mib(self.flow.disk_cache_bytes),
            ms(self.flow.disk_cache_time),
            ms(self.flow.cpu),
        )
    }
}

/// A completed statement and its execution measurements.
#[derive(Debug)]
pub struct Execution<T> {
    pub output: StatementOutput<T>,
    pub stats: ExecutionStats,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Plan(#[from] planner::Error),
    #[error("compile error: {0}")]
    Compile(#[from] planner::compile::Error),
    #[error(transparent)]
    DataFlow(#[from] dispatch::DataFlowError),
    #[error(transparent)]
    Catalog(#[from] planner::catalog::Error),
    #[error("waiter thread panicked: {0}")]
    WorkerPanic(JoinError),
    #[error("planner thread panicked: {0}")]
    PlannerPanic(JoinError),
    #[error("invalid INSERT row-count result: {0}")]
    InvalidInsertResult(String),
    #[error("{0}")]
    Copy(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Shared SQL execution state for every frontend of one Pivot instance.
#[derive(Clone)]
pub struct Engine {
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
}

impl Engine {
    pub fn new(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self {
            catalog,
            dispatcher,
            plan_cache: Arc::new(PlanCache::default()),
        }
    }

    /// Execute a statement and copy its Arrow batches out of dispatch memory.
    pub async fn execute(
        &self,
        sql: String,
        options: ExecuteOptions,
    ) -> Result<Execution<RecordBatch>> {
        self.execute_with_output::<RecordBatch>(sql, options).await
    }

    /// Execute a statement with a frontend-selected output batch type.
    ///
    /// [`OutputBatch::from_record_batch`] runs on each dispatch worker before
    /// the result leaves dispatch memory.
    pub async fn execute_with_output<T>(
        &self,
        sql: String,
        options: ExecuteOptions,
    ) -> Result<Execution<T>>
    where
        T: OutputBatch,
    {
        let transaction = self.catalog.begin_transaction();
        let result = self
            .execute_in_transaction::<T>(sql, options, transaction.clone())
            .await;
        match result {
            // A COPY FROM STDIN hands the transaction to whoever drives the
            // ingest instead of resolving it here.
            Ok(execution) if matches!(execution.output, StatementOutput::CopyFromStdin(_)) => {
                Ok(execution)
            }
            Ok(execution) => {
                transaction.commit().await?;
                Ok(execution)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }

    /// Plan a statement without executing it and report its result shape:
    /// `Some` with the row description for statements that return rows,
    /// `None` for those that do not (commands, SET, COMPACT, COPY FROM
    /// STDIN). Planning runs in its own transaction, rolled back before
    /// returning.
    pub async fn describe(&self, sql: &str) -> Result<Option<Vec<ResultColumn>>> {
        let transaction = self.catalog.begin_transaction();
        let plan = plan_query(
            &self.catalog,
            transaction.clone(),
            self.plan_cache.as_ref(),
            sql,
        )
        .await;
        transaction.rollback();
        let plan = plan?;

        if plan.as_set_variable().is_some()
            || plan.as_compact().is_some()
            || plan.as_copy_from_stdin().is_some()
        {
            return Ok(None);
        }
        match StatementKind::from_plan(&plan) {
            StatementKind::Query => Ok(Some(result_columns(&plan)?)),
            _ => Ok(None),
        }
    }

    async fn execute_in_transaction<T>(
        &self,
        sql: String,
        options: ExecuteOptions,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    ) -> Result<Execution<T>>
    where
        T: OutputBatch,
    {
        let started = Instant::now();
        let plan = plan_query(
            &self.catalog,
            transaction.clone(),
            self.plan_cache.as_ref(),
            &sql,
        )
        .await?;
        let plan_time = started.elapsed();

        if let Some(set) = plan.as_set_variable() {
            return Ok(Execution {
                output: StatementOutput::Set {
                    name: set.name.clone(),
                    value: set.value.clone(),
                },
                stats: ExecutionStats {
                    plan: plan_time,
                    ..ExecutionStats::default()
                },
            });
        }

        if let Some(request) = plan.as_compact() {
            let started = Instant::now();
            execute_compact(&self.catalog, request).await?;
            return Ok(Execution {
                output: StatementOutput::Command(Command::Compact),
                stats: ExecutionStats {
                    plan: plan_time,
                    execute: started.elapsed(),
                    ..ExecutionStats::default()
                },
            });
        }

        // A COPY FROM STDIN never compiles through the plan: its rows arrive
        // later over the frontend's protocol. Launch its ingest dataflow here
        // and hand the running exchange back; the frontend feeds it and
        // resolves it (or drops it, which aborts).
        if let Some(statement) = plan.as_copy_from_stdin() {
            let ingest = self
                .launch_copy_ingest(statement.clone(), transaction)
                .await?;
            return Ok(Execution {
                output: StatementOutput::CopyFromStdin(Box::new(ingest)),
                stats: ExecutionStats {
                    plan: plan_time,
                    ..ExecutionStats::default()
                },
            });
        }

        let statement_kind = StatementKind::from_plan(&plan);
        let columns = result_columns(&plan)?;
        let dispatcher = self.dispatcher.clone();
        #[cfg(feature = "perf")]
        let dispatcher = dispatcher.with_profiling(options.profile);

        let started = Instant::now();
        let compile_transaction = transaction;
        let handle = tokio::task::spawn_blocking(move || -> Result<StatementHandle<T>> {
            let spec = plan.compile(&dispatcher, compile_transaction.as_ref())?;
            Ok(match (statement_kind, options.collect_stats) {
                (StatementKind::Insert, true) => {
                    let output = spec.map(|| affected_rows_from_record_batch);
                    StatementHandle::Insert(output.execute_with_stats())
                }
                (StatementKind::Insert, false) => {
                    let output = spec.map(|| affected_rows_from_record_batch);
                    StatementHandle::Insert(output.execute())
                }
                (_, true) => StatementHandle::Batches(spec.execute_with_stats_as::<T>()),
                (_, false) => StatementHandle::Batches(spec.execute_as::<T>()),
            })
        })
        .await
        .map_err(Error::PlannerPanic)??;
        let compile_time = started.elapsed();

        let (results, flow, execute_time) = collect_handle(handle).await?;
        let output = match (statement_kind, results) {
            (StatementKind::Query, StatementResults::Batches(batches)) => {
                StatementOutput::Rows { columns, batches }
            }
            (StatementKind::Insert, StatementResults::Insert(outputs)) => {
                StatementOutput::Command(Command::Insert {
                    rows: affected_rows(outputs)?,
                })
            }
            (StatementKind::CreateTable, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateTable)
            }
            (StatementKind::DropTable, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::DropTable)
            }
            (StatementKind::CreateSchema, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateSchema)
            }
            (StatementKind::CreateUser, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateUser)
            }
            _ => unreachable!("statement kind and worker output must agree"),
        };
        Ok(Execution {
            output,
            stats: ExecutionStats {
                plan: plan_time,
                compile: compile_time,
                execute: execute_time,
                flow,
            },
        })
    }
}

enum StatementHandle<T> {
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

enum StatementResults<T> {
    Batches(Vec<T>),
    Insert(Vec<std::result::Result<usize, String>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatementKind {
    Query,
    Insert,
    CreateTable,
    DropTable,
    CreateSchema,
    CreateUser,
}

impl StatementKind {
    fn from_plan(plan: &planner::Plan) -> Self {
        match &plan.root.operator {
            planner::Operator::Insert(_) => Self::Insert,
            planner::Operator::CreateTable(_) => Self::CreateTable,
            planner::Operator::DropTable(_) => Self::DropTable,
            planner::Operator::CreateSchema(_) => Self::CreateSchema,
            planner::Operator::CreateUser(_) => Self::CreateUser,
            _ => Self::Query,
        }
    }
}

fn result_columns(plan: &planner::Plan) -> Result<Vec<ResultColumn>> {
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

fn affected_rows(outputs: Vec<std::result::Result<usize, String>>) -> Result<usize> {
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

fn affected_rows_from_record_batch(batch: RecordBatch) -> std::result::Result<usize, String> {
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

async fn collect_handle<T>(
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

async fn execute_compact(
    catalog: &Arc<catalog::PivotCatalog>,
    request: &planner::Compact,
) -> Result<u64> {
    let datastore_name = request
        .datastore
        .as_deref()
        .unwrap_or_else(|| catalog.default_datastore_name());
    let datastore = catalog
        .iter_datastores()
        .find(|(name, _)| name.as_str() == datastore_name)
        .map(|(_, datastore)| Arc::clone(datastore))
        .ok_or_else(|| {
            planner::catalog::Error::Other(
                format!("COMPACT: no datastore named `{datastore_name}`").into(),
            )
        })?;
    let table = planner::catalog::SchemaQualifiedTableName::new(
        request
            .schema
            .as_deref()
            .unwrap_or(planner::DEFAULT_SCHEMA_NAME),
        request.table.as_str(),
    );
    Ok(datastore.compact(&table, request.final_sweep).await?)
}

fn with_planner<R>(
    catalog: &Arc<catalog::PivotCatalog>,
    function: impl FnOnce(&mut planner::Planner) -> R,
) -> Result<R, planner::Error> {
    PLANNER.with_borrow_mut(|slot| {
        let planner = match slot {
            Some(planner) => planner,
            None => {
                let datastore_names = catalog
                    .iter_datastores()
                    .map(|(name, _)| name.clone())
                    .collect();
                slot.insert(planner::Planner::from_datastore_names(
                    datastore_names,
                    catalog.default_datastore_name().to_string(),
                )?)
            }
        };
        Ok(function(planner))
    })
}

async fn plan_query(
    catalog: &Arc<catalog::PivotCatalog>,
    transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    plan_cache: &PlanCache,
    query: &str,
) -> Result<Arc<planner::Plan>> {
    if let Some(plan) = plan_cache.get(query, transaction.as_ref()) {
        return Ok(plan);
    }

    let catalog = catalog.clone();
    let cache_key = query.to_string();
    let query = cache_key.clone();
    let plan = tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
        with_planner(&catalog, |planner| {
            Ok(Arc::new(planner.plan(&query, transaction)?))
        })?
    })
    .await
    .map_err(Error::PlannerPanic)??;
    if plan.is_cacheable() {
        plan_cache.insert(cache_key, plan.clone());
    }
    Ok(plan)
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

struct PlanCache {
    inner: Mutex<LruCache<String, Arc<planner::Plan>>>,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(PLAN_CACHE_QUERY_CAPACITY)
    }
}

impl PlanCache {
    fn new(query_capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(query_capacity).expect("plan cache capacity must be non-zero"),
            )),
        }
    }

    fn get(
        &self,
        query: &str,
        transaction: &dyn planner::catalog::CatalogTransaction,
    ) -> Option<Arc<planner::Plan>> {
        let mut inner = self.inner.lock().unwrap();
        let plan = inner.get(query)?.clone();
        if plan.has_matching_table_revisions(transaction) {
            Some(plan)
        } else {
            let _ = inner.pop(query);
            None
        }
    }

    fn insert(&self, query: String, plan: Arc<planner::Plan>) {
        debug_assert!(plan.is_cacheable());
        self.inner.lock().unwrap().put(query, plan);
    }
}

/// A running `COPY ... FROM STDIN` ingest, the engine's whole copy surface:
/// the frontend feeds it protocol bytes and finishes or drops it, and every
/// dispatch-facing concern (decoding, backpressure, cancellation, the
/// statement's transaction) stays inside.
pub struct CopyIngest {
    /// The sequential half: reassembles the protocol's byte frames into Arrow
    /// IPC messages and yields client-schema batches.
    decoder: StreamDecoder,
    sender: ChannelInputSender<RecordBatch>,
    /// Signalled by workers claiming batches; [`push`](Self::push) sleeps on
    /// it while the queue is full.
    space_freed: Arc<Notify>,
    cancel: CancelToken,
    /// Taken by [`finish`](Self::finish); still present on drop marks an
    /// unfinished ingest, which the drop aborts.
    handle: Option<DataFlowHandle<std::result::Result<usize, String>>>,
    /// The transaction the statement was planned in, which its bound table
    /// stages into. Taken and committed by [`finish`](Self::finish); rolled
    /// back on drop otherwise.
    transaction: Option<Arc<dyn planner::catalog::CatalogTransaction>>,
    column_count: usize,
}

impl fmt::Debug for CopyIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyIngest")
            .field("column_count", &self.column_count)
            .finish_non_exhaustive()
    }
}

impl CopyIngest {
    /// Columns each incoming batch must carry (for the frontend's
    /// copy-in response).
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Decode one protocol frame and feed the batches it completes to the
    /// running dataflow, sleeping while the queue is full. The decoder
    /// carries partial messages across frames, so a batch may span any number
    /// of frames (and one frame may complete several). An error means the
    /// stream is malformed; the copy cannot proceed and should be dropped.
    ///
    /// The frame arrives as refcounted [`Bytes`], so handing it to the
    /// decoder copies nothing: a message contained in one frame is sliced in
    /// place, and the decoded batches keep the frame's allocation alive.
    pub async fn push(&mut self, bytes: Bytes) -> Result<()> {
        let mut buffer = arrow::buffer::Buffer::from(bytes);
        loop {
            let batch = self
                .decoder
                .decode(&mut buffer)
                .map_err(|e| Error::Copy(format!("decoding COPY arrow stream: {e}")))?;
            let Some(mut batch) = batch else {
                return Ok(());
            };
            loop {
                // A dead dataflow claims nothing again; stop feeding it. Its
                // own error surfaces when the flow is collected at finish.
                if self.cancel.is_cancelled() {
                    return Ok(());
                }
                // Register before trying so a claim racing the failed send
                // cannot lose the notification. The timeout lets a cancelled
                // dataflow that will never claim again be observed promptly.
                let notified = self.space_freed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                match self.sender.try_send(batch) {
                    Ok(()) => break,
                    Err(ChannelInputFull(returned)) => {
                        batch = returned;
                        let _ = tokio::time::timeout(Duration::from_millis(50), notified).await;
                    }
                }
            }
        }
    }

    /// Verify the stream ended cleanly, drain the dataflow, and commit,
    /// returning the ingested-row count. Consuming `self` makes completion
    /// single-use; every failure path drops the remains, which aborts.
    pub async fn finish(mut self) -> Result<usize> {
        // Complete batches were sent as they decoded, so there is nothing to
        // flush; a stream cut mid-message is an error.
        self.decoder
            .finish()
            .map_err(|e| Error::Copy(format!("COPY arrow stream ended mid-message: {e}")))?;
        self.sender.close();
        let handle = self.handle.take().expect("a copy finishes only once");
        let outputs = tokio::task::spawn_blocking(move || handle.collect())
            .await
            .map_err(Error::WorkerPanic)??;
        let count = affected_rows(outputs)?;
        let transaction = self
            .transaction
            .take()
            .expect("the transaction resolves only here");
        transaction.commit().await?;
        Ok(count)
    }
}

/// Dropping an unfinished ingest aborts it: cancel the dataflow, stop the
/// queue, and roll back the statement's transaction, discarding whatever the
/// copy staged. A finished ingest already took both fields, so its drop does
/// nothing.
impl Drop for CopyIngest {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.cancel.cancel();
            self.sender.close();
        }
        if let Some(transaction) = self.transaction.take() {
            transaction.rollback();
        }
    }
}

impl Engine {
    /// Launch the ingest dataflow for a planned COPY FROM STDIN: a batch
    /// channel fanning out to per-worker schema-conformance stages, feeding
    /// the target table's insert sink, staging into the statement's
    /// transaction. The returned [`CopyIngest`] owns the whole exchange.
    async fn launch_copy_ingest(
        &self,
        statement: planner::CopyFromStdin,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    ) -> Result<CopyIngest> {
        let column_count = if statement.columns.is_empty() {
            statement.table.columns().len()
        } else {
            statement.columns.len()
        };
        let space_freed = Arc::new(Notify::new());
        let on_claim = {
            let space_freed = space_freed.clone();
            Box::new(move || space_freed.notify_one()) as Box<dyn Fn() + Send + Sync>
        };
        let dispatcher = self.dispatcher.clone();
        // Compile and launch on the blocking pool: the insert sink's write
        // preparation may touch the store.
        let (sender, handle) =
            tokio::task::spawn_blocking(move || -> Result<(ChannelInputSender<RecordBatch>, _)> {
                let (sender, spec) = statement.compile_ingest(&dispatcher, on_claim)?;
                Ok((
                    sender,
                    spec.map(|| affected_rows_from_record_batch).execute(),
                ))
            })
            .await
            .map_err(Error::PlannerPanic)??;
        Ok(CopyIngest {
            decoder: StreamDecoder::new(),
            cancel: handle.cancel_token(),
            sender,
            space_freed,
            handle: Some(handle),
            transaction: Some(transaction),
            column_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use planner::DEFAULT_DATASTORE_NAME;
    use planner::catalog::{BoundTable, Column, TableReference, TableRevision};

    use super::PlanCache;
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct RevisionTable {
        reference: TableReference,
        revision: TableRevision,
    }

    impl BoundTable for RevisionTable {
        fn table_reference(&self) -> TableReference {
            self.reference.clone()
        }

        fn table_revision(&self) -> TableRevision {
            self.revision.clone()
        }

        fn compile_scan(
            &self,
            _dispatcher: &dispatch::DataFlowDispatcher,
            _projection: dispatch::Projection,
            _dynamic_filters: Vec<planner::catalog::DynamicScanPredicate>,
            _emit_row_group_metadata: bool,
        ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
            unreachable!("cache tests do not compile plans")
        }

        fn columns(&self) -> Vec<Column> {
            Vec::new()
        }

        fn clone_box(&self) -> Box<dyn BoundTable> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct RevisionTransaction {
        revisions: HashMap<TableReference, TableRevision>,
    }

    #[async_trait]
    impl planner::catalog::CatalogTransaction for RevisionTransaction {
        fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
            schema == planner::DEFAULT_SCHEMA_NAME
        }

        fn bind_table(&self, _reference: &TableReference) -> Option<Box<dyn BoundTable>> {
            None
        }

        fn table_revision(&self, reference: &TableReference) -> Option<TableRevision> {
            self.revisions.get(reference).cloned()
        }
    }

    fn table() -> TableReference {
        TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: planner::DEFAULT_SCHEMA_NAME.to_string(),
            table: "events".to_string(),
        }
    }

    fn revision(identity: &str, version: u64) -> TableRevision {
        TableRevision {
            identity: identity.to_string(),
            version,
        }
    }

    fn transaction(identity: &str, version: u64) -> RevisionTransaction {
        RevisionTransaction {
            revisions: HashMap::from([(table(), revision(identity, version))]),
        }
    }

    fn plan(identity: &str, version: u64) -> Arc<planner::Plan> {
        Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(RevisionTable {
                        reference: table(),
                        revision: revision(identity, version),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
        })
    }

    #[test]
    fn cache_hit_requires_the_same_identity_and_version() {
        let cache = PlanCache::new(2);
        let cached = plan("table-id", 7);
        cache.insert("SELECT * FROM events".to_string(), cached.clone());

        let hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 7))
            .unwrap();
        let changed_version = cache.get("SELECT * FROM events", &transaction("table-id", 8));
        cache.insert("SELECT * FROM events".to_string(), cached.clone());
        let changed_identity = cache.get("SELECT * FROM events", &transaction("new-table-id", 7));

        assert!(Arc::ptr_eq(&hit, &cached));
        assert!(changed_version.is_none());
        assert!(changed_identity.is_none());
        assert!(cache.inner.lock().unwrap().is_empty());
    }

    #[test]
    fn inserting_a_new_revision_replaces_the_previous_plan() {
        let cache = PlanCache::new(2);
        cache.insert("SELECT * FROM events".to_string(), plan("table-id", 7));
        let current = plan("table-id", 8);

        cache.insert("SELECT * FROM events".to_string(), current.clone());
        let hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 8))
            .unwrap();

        assert!(Arc::ptr_eq(&hit, &current));
        assert_eq!(cache.inner.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_revision_from_another_schema_does_not_validate_a_cached_plan() {
        let analytics = TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: "analytics".to_string(),
            table: "events".to_string(),
        };
        let cached = Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(RevisionTable {
                        reference: analytics.clone(),
                        revision: revision("table-id", 7),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
        });
        let cache = PlanCache::new(2);
        cache.insert("SELECT * FROM analytics.events".to_string(), cached);
        let other_schema = RevisionTransaction {
            revisions: HashMap::from([(table(), revision("table-id", 7))]),
        };

        let hit = cache.get("SELECT * FROM analytics.events", &other_schema);

        assert!(hit.is_none());
    }

    #[test]
    fn inserting_over_capacity_evicts_the_least_recently_used_query() {
        let cache = PlanCache::new(2);
        let first = plan("first", 1);
        cache.insert("first query".to_string(), first.clone());
        cache.insert("second query".to_string(), plan("second", 1));
        cache.get("first query", &transaction("first", 1)).unwrap();

        let third = plan("third", 1);
        cache.insert("third query".to_string(), third.clone());

        assert!(
            cache
                .get("second query", &transaction("second", 1))
                .is_none()
        );
        assert!(Arc::ptr_eq(
            &cache.get("first query", &transaction("first", 1)).unwrap(),
            &first
        ));
        assert!(Arc::ptr_eq(
            &cache.get("third query", &transaction("third", 1)).unwrap(),
            &third
        ));
    }
}
