//! pgwire `ExtendedQueryHandler`: prepared statements over the
//! Parse/Bind/Describe/Execute protocol.
//!
//! Parse plans the statement once (without values) to resolve its metadata:
//! the binder-inferred type of each `$n` parameter and the result columns.
//! That is what Describe answers, and it means only statements whose
//! parameter types are fully inferable from the statement alone are accepted
//! (`INSERT INTO t VALUES ($1, $2)`, `SELECT ... WHERE x = $1`); a bare
//! `SELECT $1` is rejected at Parse.
//!
//! Execute decodes the Bind values against those types and re-plans the
//! statement with the values substituted in as constants, then runs it
//! through the same path as a simple query. The substituted plan is specific
//! to its values, so it bypasses the SQL-keyed plan cache.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{NaiveDate, NaiveDateTime};
use futures::Sink;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::ExtendedQueryHandler;
use pgwire::api::results::{FieldFormat, FieldInfo, Response};
use pgwire::api::stmt::QueryParser;
use pgwire::api::store::PortalStore;
use pgwire::api::{ClientInfo, ClientPortalStore, Type};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use tracing::{info, warn};

use crate::arrow_to_pgwire::pg_type_for_arrow;
use crate::query_handler::{PivotQueryHandler, build_outcome_response, stats_on, with_planner};
use planner::ParameterValue;
use planner::types::physical_arrow_type;

/// A statement prepared by Parse: the SQL plus the metadata the planner
/// resolved for it, kept so Describe and Bind decoding need no re-planning.
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    sql: String,
    /// The binder-inferred type of each `$n` parameter, ordered by n.
    parameter_types: Vec<planner::types::Type>,
    /// One `(name, type)` per result column, or `None` when the statement
    /// returns no result set (INSERT, DDL, SET) and Describe answers NoData.
    columns: Option<Vec<(String, planner::types::Type)>>,
}

/// The Postgres wire type a value of this pivot type is described as, the
/// same mapping the result encoder uses (via the type's arrow carrier), so
/// parameter and column OIDs can't drift from the row encoding.
fn pg_type_for(pivot_type: &planner::types::Type) -> Type {
    pg_type_for_arrow(&physical_arrow_type(pivot_type))
}

/// pgwire `QueryParser`: plans each Parse'd statement to resolve its
/// parameter types and result schema (see the module doc).
pub struct PivotQueryParser {
    catalog: Arc<catalog::PivotCatalog>,
}

impl PivotQueryParser {
    pub(crate) fn new(catalog: Arc<catalog::PivotCatalog>) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl QueryParser for PivotQueryParser {
    type Statement = PreparedQuery;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<PreparedQuery>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        // An empty Parse is answered with EmptyQueryResponse at Execute; there
        // is nothing to plan.
        if sql.trim().is_empty() {
            return Ok(PreparedQuery {
                sql: sql.to_string(),
                parameter_types: Vec::new(),
                columns: None,
            });
        }

        // Describe against a fresh catalog snapshot, on the blocking pool
        // (the thread-local planner wraps a non-`Send` DuckDB context). The
        // snapshot is read-only here, so it is rolled back either way; the
        // Execute re-plans in its own transaction.
        let catalog = self.catalog.clone();
        let query = sql.to_string();
        let transaction = catalog.begin_transaction();
        let planning_transaction = transaction.clone();
        let described = tokio::task::spawn_blocking(move || {
            with_planner(&catalog, |planner| {
                planner.describe(&query, planning_transaction)
            })?
        })
        .await;
        transaction.rollback();
        let description = described
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?
            .map_err(|e| {
                warn!(error = %e, sql = %sql, "parse failed");
                plan_error(e)
            })?;

        Ok(PreparedQuery {
            sql: sql.to_string(),
            parameter_types: description.parameter_types,
            columns: description.columns,
        })
    }

    fn get_parameter_types(&self, stmt: &PreparedQuery) -> PgWireResult<Vec<Type>> {
        Ok(stmt.parameter_types.iter().map(pg_type_for).collect())
    }

    fn get_result_schema(
        &self,
        stmt: &PreparedQuery,
        column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        let Some(columns) = &stmt.columns else {
            // An empty schema makes pgwire answer the Describe with NoData.
            return Ok(Vec::new());
        };
        Ok(columns
            .iter()
            .enumerate()
            .map(|(idx, (name, column_type))| {
                let format =
                    column_format.map_or(FieldFormat::Text, |format| format.format_for(idx));
                FieldInfo::new(name.clone(), None, None, pg_type_for(column_type), format)
            })
            .collect())
    }
}

#[async_trait]
impl ExtendedQueryHandler for PivotQueryHandler {
    type Statement = PreparedQuery;
    type QueryParser = PivotQueryParser;

    fn query_parser(&self) -> Arc<PivotQueryParser> {
        self.query_parser.clone()
    }

    async fn do_query<C>(
        &self,
        client: &mut C,
        portal: &Portal<PreparedQuery>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = PreparedQuery>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let statement = &portal.statement.statement;
        if statement.sql.trim().is_empty() {
            return Ok(Response::EmptyQuery);
        }
        let parameters = decode_parameter_values(portal)?;

        let with_stats = stats_on(client);
        #[cfg(feature = "perf")]
        let with_perf = crate::query_handler::perf_on(client);
        #[cfg(not(feature = "perf"))]
        let with_perf = false;

        info!(sql = %statement.sql, "prepared query received");
        let outcome = self
            .run_query(
                &statement.sql,
                parameters,
                portal.result_column_format.clone(),
                with_stats,
                with_perf,
            )
            .await
            .map_err(|e| {
                warn!(error = %e, sql = %statement.sql, "prepared query failed");
                e.into_pgwire()
            })?;

        let res = build_outcome_response(client, outcome, with_stats).await?;
        info!(sql = %statement.sql, "prepared query succeeded");
        Ok(res)
    }
}

/// Decode every Bind value into a [`ParameterValue`]. A text-format value is
/// carried as its SQL text (the planner casts it to the parameter's inferred
/// type); a binary-format value is decoded by the wire type the client
/// declared in Parse, falling back to the inferred type when it declared none
/// (the same precedence Describe reports).
fn decode_parameter_values(portal: &Portal<PreparedQuery>) -> PgWireResult<Vec<ParameterValue>> {
    let statement = &portal.statement.statement;
    if portal.parameters.len() != statement.parameter_types.len() {
        return Err(invalid_parameter(format!(
            "bind supplies {} parameters but the statement requires {}",
            portal.parameters.len(),
            statement.parameter_types.len()
        )));
    }
    (0..portal.parameters.len())
        .map(|idx| {
            let wire_type = portal
                .statement
                .parameter_types
                .get(idx)
                .cloned()
                .flatten()
                .unwrap_or_else(|| pg_type_for(&statement.parameter_types[idx]));
            decode_parameter_value(portal, idx, &wire_type)
        })
        .collect()
}

fn decode_parameter_value(
    portal: &Portal<PreparedQuery>,
    idx: usize,
    wire_type: &Type,
) -> PgWireResult<ParameterValue> {
    let Some(raw) = &portal.parameters[idx] else {
        return Ok(ParameterValue::Null);
    };
    if portal.parameter_format.is_text(idx) {
        let text = std::str::from_utf8(raw).map_err(|e| {
            invalid_parameter(format!("parameter ${} is not valid UTF-8: {e}", idx + 1))
        })?;
        return Ok(ParameterValue::Text(text.to_string()));
    }

    // Binary format: decode by the wire type, into the carrier the planner's
    // value cast expects. `portal.parameter` returned `Some` bytes above, so
    // a `None` decode cannot occur.
    let value = match wire_type {
        t if *t == Type::BOOL => ParameterValue::Boolean(required(portal, idx, t)?),
        t if *t == Type::INT2 => ParameterValue::Int(required::<i16>(portal, idx, t)? as i64),
        t if *t == Type::INT4 => ParameterValue::Int(required::<i32>(portal, idx, t)? as i64),
        t if *t == Type::INT8 => ParameterValue::Int(required::<i64>(portal, idx, t)?),
        t if *t == Type::FLOAT4 => ParameterValue::Float(required::<f32>(portal, idx, t)? as f64),
        t if *t == Type::FLOAT8 => ParameterValue::Float(required::<f64>(portal, idx, t)?),
        t if *t == Type::TEXT || *t == Type::VARCHAR || *t == Type::BPCHAR || *t == Type::NAME => {
            ParameterValue::Text(required::<String>(portal, idx, t)?)
        }
        // NUMERIC crosses as its text rendering; the planner casts it to the
        // inferred DECIMAL type without a lossy float detour.
        t if *t == Type::NUMERIC => {
            ParameterValue::Text(required::<rust_decimal::Decimal>(portal, idx, t)?.to_string())
        }
        t if *t == Type::DATE => {
            let date = required::<NaiveDate>(portal, idx, t)?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is a valid date");
            ParameterValue::Date(date.signed_duration_since(epoch).num_days() as i32)
        }
        t if *t == Type::TIMESTAMP => {
            let timestamp = required::<NaiveDateTime>(portal, idx, t)?;
            ParameterValue::Timestamp(timestamp.and_utc().timestamp_micros())
        }
        other => {
            return Err(invalid_parameter(format!(
                "binary format is not supported for parameter ${} of type {other}",
                idx + 1
            )));
        }
    };
    Ok(value)
}

/// Decode a present (non-NULL) parameter as `T`, treating an impossible
/// `None` as an error rather than silently binding NULL.
fn required<'a, T>(
    portal: &'a Portal<PreparedQuery>,
    idx: usize,
    wire_type: &Type,
) -> PgWireResult<T>
where
    T: postgres_types::FromSqlOwned + pgwire::types::FromSqlText<'a>,
{
    portal
        .parameter::<T>(idx, wire_type)?
        .ok_or_else(|| invalid_parameter(format!("parameter ${} decoded to no value", idx + 1)))
}

/// A Bind/parameter problem as the client sees it: SQLSTATE 22P02
/// (invalid text representation family), severity ERROR.
fn invalid_parameter(message: String) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        "22P02".to_string(),
        message,
    )))
}

/// A planning failure surfaced at Parse, as a normal error response
/// (severity ERROR, generic SQLSTATE).
fn plan_error(error: planner::Error) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        "XX000".to_string(),
        error.to_string(),
    )))
}
