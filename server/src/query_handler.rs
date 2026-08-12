//! PostgreSQL wire adapters around the transport-neutral query engine.

use std::fmt::Debug;
use std::sync::Arc;

use crate::arrow_to_pgwire::{PGRowBatch, build_field_info};
use crate::auth::Authenticator;
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use engine::{Command, ExecuteOptions, Execution, StatementOutput};
use futures::{Sink, SinkExt, stream};
use metastore::Metastore;
use pgwire::api::auth::StartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{QueryResponse, Response, Tag};
use pgwire::api::store::PortalStore;
use pgwire::api::{
    ClientInfo, ClientPortalStore, ConnectionManager, NoopHandler, PgWireServerHandlers,
};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use pgwire::messages::response::NoticeResponse;
use tracing::{info, warn};

/// Per-connection flag toggled with `SET pivot_stats = true`.
const STATS_FLAG: &str = "pivot_stats";

/// Per-connection flag toggled with `SET perf = 1`.
#[cfg(feature = "perf")]
const PERF_FLAG: &str = "perf";

/// Run SQL through the same engine as the PostgreSQL endpoint and return
/// heap-backed Arrow batches for the HTTP dashboard.
pub(crate) async fn execute_sql(
    engine: Arc<engine::Engine>,
    sql: String,
) -> Result<Vec<arrow_array::RecordBatch>, String> {
    let execution = engine
        .execute(sql, ExecuteOptions::default())
        .await
        .map_err(|error| error.to_string())?;
    match execution.output {
        StatementOutput::Rows { batches, .. } => Ok(batches),
        StatementOutput::Command(_) | StatementOutput::Set { .. } => Ok(Vec::new()),
    }
}

fn into_pgwire_error(error: engine::Error) -> PgWireError {
    let info = ErrorInfo::new("ERROR".to_string(), "XX000".to_string(), error.to_string());
    PgWireError::UserError(Box::new(info))
}

fn build_query_response(columns: Vec<engine::ResultColumn>, batches: Vec<PGRowBatch>) -> Response {
    let fields = batches.first().map_or_else(
        || {
            let schema = Arc::new(Schema::new(
                columns
                    .into_iter()
                    .map(|column| Field::new(column.name, column.data_type, true))
                    .collect::<Vec<_>>(),
            ));
            build_field_info(&schema)
        },
        |batch| batch.fields.clone(),
    );
    Response::Query(QueryResponse::new(
        fields,
        stream::iter(batches.into_iter().flat_map(|batch| batch.rows).map(Ok)),
    ))
}

fn build_command_response(command: Command) -> Response {
    match command {
        Command::Insert { rows } => {
            Response::Execution(Tag::new("INSERT").with_oid(0).with_rows(rows))
        }
        Command::CreateTable => Response::Execution(Tag::new("CREATE TABLE")),
        Command::CreateSchema => Response::Execution(Tag::new("CREATE SCHEMA")),
        Command::CreateUser => Response::Execution(Tag::new("CREATE USER")),
        Command::Compact => Response::Execution(Tag::new("COMPACT")),
    }
}

/// The pgwire simple-query frontend for one shared engine.
pub struct PivotQueryHandler {
    engine: Arc<engine::Engine>,
}

impl PivotQueryHandler {
    pub fn new(engine: Arc<engine::Engine>) -> Self {
        Self { engine }
    }

    async fn run_query(
        &self,
        query: &str,
        collect_stats: bool,
        with_perf: bool,
    ) -> engine::Result<Execution<PGRowBatch>> {
        #[cfg(feature = "perf")]
        let mut perf_guard = if with_perf {
            let sql = query.to_string();
            tokio::task::spawn_blocking(move || crate::perf::start(&sql))
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        #[cfg(not(feature = "perf"))]
        let _ = with_perf;
        #[cfg(feature = "perf")]
        let profile = perf_guard.is_some();
        #[cfg(not(feature = "perf"))]
        let profile = false;

        let result = self
            .engine
            .execute_with_output::<PGRowBatch>(
                query.to_string(),
                ExecuteOptions {
                    collect_stats,
                    profile,
                },
            )
            .await;

        #[cfg(feature = "perf")]
        if let Some(perf) = perf_guard.take() {
            let _ = tokio::task::spawn_blocking(move || drop(perf)).await;
        }
        result
    }
}

fn apply_set<C: ClientInfo>(client: &mut C, name: &str, value: Option<&str>) -> Response {
    if name.eq_ignore_ascii_case(STATS_FLAG) {
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(STATS_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(STATS_FLAG);
        }
    }
    #[cfg(feature = "perf")]
    if name.eq_ignore_ascii_case(PERF_FLAG) {
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(PERF_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(PERF_FLAG);
        }
    }
    Response::Execution(Tag::new(if value.is_none() { "RESET" } else { "SET" }))
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "true" | "t" | "1" | "on" | "yes"
    )
}

fn stats_on<C: ClientInfo>(client: &C) -> bool {
    client
        .metadata()
        .get(STATS_FLAG)
        .is_some_and(|value| value == "on")
}

#[cfg(feature = "perf")]
fn perf_on<C: ClientInfo>(client: &C) -> bool {
    client
        .metadata()
        .get(PERF_FLAG)
        .is_some_and(|value| value == "on")
}

#[async_trait]
impl SimpleQueryHandler for PivotQueryHandler {
    async fn do_query<C>(&self, client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let with_stats = stats_on(client);
        #[cfg(feature = "perf")]
        let with_perf = perf_on(client);
        #[cfg(not(feature = "perf"))]
        let with_perf = false;

        info!(sql = %query, "query received");
        let execution = self
            .run_query(query, with_stats, with_perf)
            .await
            .map_err(|error| {
                warn!(%error, sql = %query, "query failed");
                into_pgwire_error(error)
            })?;

        if with_stats {
            let notice = NoticeResponse::from(ErrorInfo::new(
                "INFO".to_string(),
                "00000".to_string(),
                execution.stats.summary(),
            ));
            client
                .send(PgWireBackendMessage::NoticeResponse(notice))
                .await?;
        }

        let response = match execution.output {
            StatementOutput::Rows { columns, batches } => build_query_response(columns, batches),
            StatementOutput::Command(command) => build_command_response(command),
            StatementOutput::Set { name, value } => apply_set(client, &name, value.as_deref()),
        };

        info!(sql = %query, "query succeeded");
        Ok(vec![response])
    }
}

/// The startup, query, cancel, and unsupported-protocol handlers handed to
/// pgwire for each connection.
pub struct PivotHandlers {
    query_handler: Arc<PivotQueryHandler>,
    cancel_handler: Arc<DefaultCancelHandler>,
    authenticator: Authenticator,
}

impl PivotHandlers {
    pub fn new(query_engine: Arc<engine::Engine>, metastore: Arc<dyn Metastore>) -> Self {
        let manager = Arc::new(ConnectionManager::new());
        Self {
            query_handler: Arc::new(PivotQueryHandler::new(query_engine)),
            cancel_handler: Arc::new(DefaultCancelHandler::new(manager.clone())),
            authenticator: Authenticator::new(metastore, manager),
        }
    }
}

impl PgWireServerHandlers for PivotHandlers {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        self.query_handler.clone()
    }

    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        self.authenticator.startup_handler()
    }

    fn cancel_handler(&self) -> Arc<impl CancelHandler> {
        self.cancel_handler.clone()
    }

    fn extended_query_handler(&self) -> Arc<impl pgwire::api::query::ExtendedQueryHandler> {
        Arc::new(NoopHandler)
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        Arc::new(NoopHandler)
    }
}

#[cfg(test)]
mod tests {
    use super::is_truthy;

    #[test]
    fn truthy_set_values_are_case_insensitive() {
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("yes"));
        assert!(!is_truthy("off"));
    }
}
