//! pgwire extended query protocol: the Parse/Bind/Describe/Execute flow every
//! real driver (JDBC, tokio-postgres, psycopg) uses for prepared statements.
//!
//! Parse plans the SQL once, the expensive DuckDB parse/optimize round-trip,
//! into a [`planner::Plan`] whose `$n` placeholders are first-class typed
//! holes. Describe answers straight from that plan: parameter types and the
//! result schema, computed once here at Parse time.
//! Each Execute decodes the raw bound bytes the client sent into single-row
//! arrow arrays, drops them into the plan's holes
//! ([`planner::Plan::bind_parameters`]), and runs the bound plan through the
//! exact compile/execute path a simple query takes. The plan is the reuse
//! boundary; binding is the cheap per-execute step.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::types::Date32Type;
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, StringViewArray, TimestampSecondArray,
};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use chrono::{NaiveDate, NaiveDateTime};
use futures::Sink;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::ExtendedQueryHandler;
use pgwire::api::results::{FieldInfo, Response};
use pgwire::api::stmt::QueryParser;
use pgwire::api::store::PortalStore;
use pgwire::api::{ClientInfo, ClientPortalStore, Type};
use pgwire::error::{PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use tracing::{info, warn};

use crate::arrow_to_pgwire::{binary_encodable, build_field_info, pg_type_for_arrow};
use crate::query_handler::{
    PivotQueryHandler, perf_enabled, respond_with_outcome, stats_on, user_error,
};
use planner::types::{Type as PivotType, physical_arrow_type};

/// A statement prepared by a Parse message: planned once, executed per Bind.
/// The schemas Describe answers with are derived here, at Parse time, so
/// Describe and Execute never re-walk the plan.
#[derive(Clone)]
pub struct PreparedStatement {
    sql: String,
    /// `None` for an empty statement, which Postgres answers with
    /// `EmptyQueryResponse` instead of running anything.
    plan: Option<Arc<planner::Plan>>,
    /// The planned type of each `$n` parameter, by position.
    parameter_types: Vec<PivotType>,
    /// The result columns, or `None` for a statement that only reports a
    /// command tag (INSERT, DDL, SET).
    output_types: Option<Vec<PivotType>>,
}

/// pgwire [`QueryParser`]: the Parse-message half of the handler. Planning
/// happens here, so a malformed statement fails at Parse (as in Postgres) and
/// Describe can answer without ever executing.
pub struct PivotQueryParser {
    handler: Arc<PivotQueryHandler>,
}

#[async_trait]
impl QueryParser for PivotQueryParser {
    type Statement = PreparedStatement;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<PreparedStatement>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        info!(sql = %sql, "parse received");
        let trimmed = sql.trim();
        if trimmed.is_empty() || trimmed == ";" {
            return Ok(PreparedStatement {
                sql: sql.to_string(),
                plan: None,
                parameter_types: Vec::new(),
                output_types: None,
            });
        }

        // Plan against a snapshot the statement releases right after (planning
        // only reads): execution binds and compiles under its own, later
        // transaction.
        let catalog = self.handler.catalog().clone();
        let transaction = catalog.begin_transaction();
        let planned = crate::query_handler::plan_on_blocking_pool(
            &catalog,
            sql.to_string(),
            transaction.clone(),
        )
        .await;
        catalog.rollback_transaction(transaction);

        let plan = planned.map(Arc::new).map_err(|e| {
            warn!(error = %e, sql = %sql, "parse failed");
            e.into_pgwire()
        })?;
        let parameter_types = plan
            .parameter_types()
            .map_err(|e| user_error(e.to_string()))?;
        let output_types = plan
            .root
            .operator
            .returns_rows()
            .then(|| plan.root.output_types())
            .transpose()
            .map_err(|e| user_error(e.to_string()))?;
        Ok(PreparedStatement {
            sql: sql.to_string(),
            plan: Some(plan),
            parameter_types,
            output_types,
        })
    }

    fn get_parameter_types(&self, stmt: &PreparedStatement) -> PgWireResult<Vec<Type>> {
        Ok(stmt.parameter_types.iter().map(pg_parameter_type).collect())
    }

    fn get_result_schema(
        &self,
        stmt: &PreparedStatement,
        column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        let Some(output_types) = &stmt.output_types else {
            return Ok(Vec::new());
        };
        let format = column_format.unwrap_or(&Format::UnifiedText);
        check_result_format(format, output_types)?;
        Ok(result_field_info(stmt, format))
    }
}

/// The wire schema for a statement's result columns: the exact builder the row
/// encoder uses ([`build_field_info`]), fed an arrow schema assembled from the
/// plan's output names and types, so Describe's advertisement and the encoded
/// rows can't drift apart. Only meaningful for a row-returning statement.
fn result_field_info(stmt: &PreparedStatement, format: &Format) -> Vec<FieldInfo> {
    let output_types = stmt.output_types.as_deref().unwrap_or_default();
    let plan = stmt.plan.as_ref().expect("a row-returning plan exists");
    let fields: Vec<Field> = output_types
        .iter()
        .enumerate()
        .map(|(idx, column_type)| {
            // DuckDB's binder-resolved client names, falling back to
            // Postgres's placeholder for a name it didn't report.
            let name = plan
                .output_names
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "?column?".to_string());
            Field::new(name, physical_arrow_type(column_type), true)
        })
        .collect();
    build_field_info(&Arc::new(Schema::new(fields)), format)
        .as_ref()
        .clone()
}

/// The Postgres type a parameter of this planned type is declared as (and
/// decoded from when the client didn't declare its own): the same mapping the
/// result columns use.
fn pg_parameter_type(planned: &PivotType) -> Type {
    pg_type_for_arrow(&physical_arrow_type(planned))
}

/// Validate a portal's requested result formats against the statement's output
/// columns: a per-column format list must match the column count (Postgres's
/// own Bind-time rule), and binary may only be requested for columns pivot can
/// binary-encode, rather than corrupting them on the wire.
fn check_result_format(format: &Format, types: &[PivotType]) -> PgWireResult<()> {
    if let Format::Individual(codes) = format
        && codes.len() != types.len()
    {
        return Err(user_error(format!(
            "bind message has {} result format codes but query returns {} columns",
            codes.len(),
            types.len()
        )));
    }
    for (idx, column_type) in types.iter().enumerate() {
        if format.is_binary(idx) && !binary_encodable(&physical_arrow_type(column_type)) {
            return Err(user_error(format!(
                "binary result format is not supported for {column_type} columns"
            )));
        }
    }
    Ok(())
}

/// Decode every bound parameter into a single-row arrow array, by position.
/// Each value is decoded per the client's declared Parse type when there is
/// one (the raw bytes are in that type's text or binary wire form), else per
/// the plan's inferred type; binding casts the result to the hole's planned
/// type either way.
fn decode_parameters(
    portal: &Portal<PreparedStatement>,
    planned: &[PivotType],
) -> PgWireResult<Vec<ArrayRef>> {
    if portal.parameter_len() != planned.len() {
        return Err(user_error(
            planner::bind::Error::ParameterCount {
                expected: planned.len(),
                provided: portal.parameter_len(),
            }
            .to_string(),
        ));
    }
    // A per-parameter format-code list must match the parameter count, or the
    // per-index lookups below would index out of bounds (Postgres's own
    // Bind-time rule; the result-format twin lives in `check_result_format`).
    if let Format::Individual(codes) = &portal.parameter_format
        && codes.len() != planned.len()
    {
        return Err(user_error(format!(
            "bind message has {} parameter format codes but statement takes {} parameters",
            codes.len(),
            planned.len()
        )));
    }
    (0..planned.len())
        .map(|idx| {
            let declared = portal
                .statement
                .parameter_types
                .get(idx)
                .cloned()
                .flatten()
                .unwrap_or_else(|| pg_parameter_type(&planned[idx]));
            decode_parameter(portal, idx, &declared)
        })
        .collect()
}

/// Decode one bound parameter (text or binary wire form, chosen by the Bind
/// format codes) into a single-row array of the decoded type's natural arrow
/// shape. An SQL NULL becomes a one-row null array.
fn decode_parameter(
    portal: &Portal<PreparedStatement>,
    idx: usize,
    pg_type: &Type,
) -> PgWireResult<ArrayRef> {
    Ok(match *pg_type {
        Type::BOOL => Arc::new(BooleanArray::from(vec![
            portal.parameter::<bool>(idx, pg_type)?,
        ])),
        Type::INT2 => Arc::new(Int16Array::from(vec![
            portal.parameter::<i16>(idx, pg_type)?,
        ])),
        Type::INT4 => Arc::new(Int32Array::from(vec![
            portal.parameter::<i32>(idx, pg_type)?,
        ])),
        Type::INT8 => Arc::new(Int64Array::from(vec![
            portal.parameter::<i64>(idx, pg_type)?,
        ])),
        Type::FLOAT4 => Arc::new(Float32Array::from(vec![
            portal.parameter::<f32>(idx, pg_type)?,
        ])),
        Type::FLOAT8 => Arc::new(Float64Array::from(vec![
            portal.parameter::<f64>(idx, pg_type)?,
        ])),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME | Type::UNKNOWN => {
            Arc::new(StringViewArray::from(vec![
                portal.parameter::<String>(idx, pg_type)?,
            ]))
        }
        Type::DATE => {
            let date = portal.parameter::<NaiveDate>(idx, pg_type)?;
            Arc::new(Date32Array::from(vec![
                date.map(Date32Type::from_naive_date),
            ]))
        }
        Type::TIMESTAMP => {
            // The engine stores timestamps at second granularity everywhere
            // (see `build_scalar_value`, which drops a plan-time timestamp
            // constant to seconds the same way), so a bound value's sub-second
            // part is truncated rather than rejected.
            let timestamp = portal.parameter::<NaiveDateTime>(idx, pg_type)?;
            Arc::new(TimestampSecondArray::from(vec![
                timestamp.map(|t| t.and_utc().timestamp()),
            ]))
        }
        ref other => {
            return Err(user_error(format!(
                "unsupported parameter type {other} (bind it as text or add a cast)"
            )));
        }
    })
}

/// pgwire `ExtendedQueryHandler`: Execute decodes the portal's bound values,
/// binds them into the prepared plan, and runs it on the shared
/// [`PivotQueryHandler`]. Describe is served by the pgwire-provided defaults on
/// top of [`PivotQueryParser`]'s schema answers.
pub struct PivotExtendedQueryHandler {
    handler: Arc<PivotQueryHandler>,
    parser: Arc<PivotQueryParser>,
}

impl PivotExtendedQueryHandler {
    pub fn new(handler: Arc<PivotQueryHandler>) -> Self {
        Self {
            parser: Arc::new(PivotQueryParser {
                handler: handler.clone(),
            }),
            handler,
        }
    }
}

#[async_trait]
impl ExtendedQueryHandler for PivotExtendedQueryHandler {
    type Statement = PreparedStatement;
    type QueryParser = PivotQueryParser;

    fn query_parser(&self) -> Arc<PivotQueryParser> {
        self.parser.clone()
    }

    async fn do_query<C>(
        &self,
        client: &mut C,
        portal: &Portal<PreparedStatement>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = PreparedStatement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let statement = &portal.statement.statement;
        let Some(plan) = statement.plan.clone() else {
            return Ok(Response::EmptyQuery);
        };

        let with_stats = stats_on(client);
        let with_perf = perf_enabled(client);

        info!(sql = %statement.sql, "extended query received");
        let params = decode_parameters(portal, &statement.parameter_types)?;
        if let Some(output_types) = &statement.output_types {
            check_result_format(&portal.result_column_format, output_types)?;
        }

        let outcome = self
            .handler
            .run_prepared(
                plan,
                params,
                portal.result_column_format.clone(),
                with_stats,
                with_perf,
                statement.sql.clone(),
            )
            .await
            .map_err(|e| {
                warn!(error = %e, sql = %statement.sql, "extended query failed");
                e.into_pgwire()
            })?;

        let response = respond_with_outcome(client, outcome, with_stats).await?;
        info!(sql = %statement.sql, "extended query succeeded");
        Ok(response)
    }
}
