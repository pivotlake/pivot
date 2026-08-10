//! duckdb-planner: a thin FFI wrapper around DuckDB's C++ query planner.
//!
//! This crate sends SQL strings to an embedded DuckDB instance and exposes the
//! resolved logical plan as safe borrowed handles ([`LogicalOp`] / [`Expr`]).
//! It does *not* materialize the plan into owned Rust structs: walking the
//! handles and building whatever IR a caller needs is the caller's job.
//!
//! The main entry point is [`PlannerContext`], which owns the DuckDB state and
//! exposes [`plan`](PlannerContext::plan) (returns a [`Plan`] handle).
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use duckdb_planner::{
//!     DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, PlannerContext,
//! };
//! use duckdb_planner::duckdb_bridge::duckdb_types::LogicalOperatorType;
//!
//! struct UsersTable;
//!
//! impl DuckDBTable for UsersTable {
//!     fn clone_box(&self) -> Box<dyn DuckDBTable> { Box::new(UsersTable) }
//!     fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
//!         vec![DuckDBColumn {
//!             name: "name".to_string(),
//!             duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
//!             decimal_width: 0,
//!             decimal_scale: 0,
//!         }]
//!     }
//! }
//!
//! struct MyCatalog;
//!
//! impl DuckDBBind for MyCatalog {}
//!
//! // BoundTable names resolve through a per-plan transaction (a snapshot of the
//! // catalog), not through the provider itself.
//! struct MyTransaction;
//!
//! impl DuckDBTransaction for MyTransaction {
//!     fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
//!         schema == "main"
//!     }
//!     fn bind_table(&self, _datastore: &str, _schema: &str, name: &str) -> Option<Box<dyn DuckDBTable>> {
//!         match name {
//!             "users" => Some(Box::new(UsersTable)),
//!             _ => None,
//!         }
//!     }
//! }
//!
//! let mut ctx = PlannerContext::new(Arc::new(MyCatalog), vec!["db".to_string()], "db".to_string())
//!     .unwrap();
//!
//! // Plan a query inside a transaction; walk the root handle.
//! let plan = ctx.plan("SELECT name FROM users", Arc::new(MyTransaction)).unwrap();
//! let root = plan.root().unwrap();
//! assert_eq!(root.op_type().unwrap(), LogicalOperatorType::LOGICAL_PROJECTION);
//! ```

#![allow(clippy::upper_case_acronyms)]

pub mod catalog_provider;
pub mod duckdb_bridge;
pub mod handle;
mod types;

use std::sync::Arc;

use duckdb_bridge::ffi;
use thiserror::Error;

pub use catalog_provider::{DuckDBBind, DuckDBTable, DuckDBTransaction};
pub use duckdb_bridge::duckdb_types::LogicalTypeId;
pub use duckdb_bridge::ffi::DuckDBColumn;
pub use handle::{BridgeError, Expr, LogicalOp, Plan};
pub use types::{BoundLogicalType, ExtraTypeInfo, ScalarValue};

/// The outcome of planning a statement in prepare mode
/// ([`PlannerContext::plan_prepare`]): the parameter and result shapes a wire
/// protocol describes to the client, plus the plan itself when it is worth
/// reusing across executions.
///
/// A parameterized statement that reads a table carries no plan: a placeholder
/// blocks the scan pushdown a constant unlocks, so each execution replans with
/// its values through [`PlannerContext::plan_with_values`] instead.
pub struct PreparedPlanning {
    pub plan: Option<Plan>,
    /// The inferred types of the statement's parameters, ordered `$1..$n`.
    pub param_types: Vec<BoundLogicalType>,
    /// The binder-resolved result column names, in output order.
    pub output_names: Vec<String>,
    /// The binder-resolved result column types, parallel to `output_names`.
    pub output_types: Vec<BoundLogicalType>,
    /// Whether executing the statement produces a result set (a SELECT)
    /// rather than a command tag (an INSERT).
    pub returns_rows: bool,
}

/// Top-level error type for the planner.
#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    DuckDBPlanning(#[from] PlanningError),
    #[error("{0}")]
    UnsupportedPlan(String),
    #[error("Bridge error: {0}")]
    Bridge(String),
}

impl From<cxx::Exception> for Error {
    fn from(exception: cxx::Exception) -> Self {
        Error::Bridge(exception.what().to_string())
    }
}

impl From<BridgeError> for Error {
    fn from(error: BridgeError) -> Self {
        Error::Bridge(error.0)
    }
}

/// Error returned by DuckDB when it cannot produce a plan for a query.
#[derive(Debug, Error)]
pub struct PlanningError {
    pub exception_message: String,
    pub position: Option<String>,
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.exception_message)
    }
}

/// Translate the error fields the bridge reports into a typed [`enum@Error`].
fn bridge_error(result: &ffi::ExtractPlanResult) -> Error {
    let position = result
        .has_error_position
        .then(|| result.error_position.clone());
    match result.error_kind.as_str() {
        "duckdb_planning" => Error::DuckDBPlanning(PlanningError {
            exception_message: result.error_message.clone(),
            position,
        }),
        "unsupported_plan" => Error::UnsupportedPlan(result.error_message.clone()),
        "bridge_error" => Error::Bridge(result.error_message.clone()),
        other => Error::Bridge(format!(
            "Unknown bridge error kind `{other}`: {}",
            result.error_message
        )),
    }
}

/// Owns an in-process DuckDB instance and exposes SQL planning.
pub struct PlannerContext {
    cxx_context: cxx::UniquePtr<ffi::DuckPlannerContext>,
}

impl PlannerContext {
    /// Create a planner context backed by the given static provider, attaching
    /// one DuckDB database per name in `database_names` and making `default_name`
    /// the current database. The provider resolves only the transaction-
    /// independent names (scalar functions), which are generic across datastores;
    /// per-query table lookups are routed by database name through the
    /// transaction handed to [`plan`](Self::plan).
    /// Creating the context attaches every datastore as a DuckDB database, and
    /// any of those steps can fail inside DuckDB; the C++ exception comes back
    /// as [`Error::Bridge`].
    pub fn new(
        provider: Arc<dyn DuckDBBind>,
        database_names: Vec<String>,
        default_name: String,
    ) -> Result<Self, Error> {
        let ctx = Box::new(catalog_provider::CatalogContext::new(
            provider,
            database_names,
            default_name,
        ));
        Ok(Self {
            cxx_context: ffi::new_context(ctx)?,
        })
    }

    /// Plan a SQL query inside `transaction`: sends the query to DuckDB and
    /// returns a [`Plan`] handle owning the resolved logical plan. The caller
    /// walks it through [`Plan::root`] and the [`LogicalOp`]/[`Expr`] accessors.
    ///
    /// Every table the query references is bound through `transaction`, so one
    /// plan sees one consistent snapshot of the catalog. The transaction is
    /// published to the bridge only for the duration of this call.
    pub fn plan(
        &mut self,
        query: &str,
        transaction: Arc<dyn DuckDBTransaction>,
    ) -> Result<Plan, Error> {
        let transaction_ctx = catalog_provider::TransactionContext::new(transaction);
        let result = ffi::extract_plan(self.cxx_context.pin_mut(), query, &transaction_ctx)?;
        if !result.error_kind.is_empty() {
            return Err(bridge_error(&result));
        }

        let ffi::ExtractPlanResult {
            plan, output_names, ..
        } = result;
        Ok(Plan::new(plan, output_names))
    }

    /// Plan a SQL statement with prepared-statement parameters allowed: each
    /// `$n` binds as a typed placeholder that stays in the plan. See
    /// [`PreparedPlanning`] for what comes back and when the plan is withheld.
    pub fn plan_prepare(
        &mut self,
        query: &str,
        transaction: Arc<dyn DuckDBTransaction>,
    ) -> Result<PreparedPlanning, Error> {
        let transaction_ctx = catalog_provider::TransactionContext::new(transaction);
        let result =
            ffi::extract_plan_prepare(self.cxx_context.pin_mut(), query, &transaction_ctx)?;
        if !result.error_kind.is_empty() {
            return Err(bridge_error(&result));
        }

        let ffi::ExtractPlanResult {
            plan,
            output_names,
            output_types,
            param_types,
            returns_rows,
            ..
        } = result;
        Ok(PreparedPlanning {
            plan: (!plan.is_null()).then(|| Plan::new(plan, output_names.clone())),
            param_types: param_types
                .into_iter()
                .map(BoundLogicalType::from_bridge)
                .collect(),
            output_names,
            output_types: output_types
                .into_iter()
                .map(BoundLogicalType::from_bridge)
                .collect(),
            returns_rows,
        })
    }

    /// Plan a SQL statement with the given parameter values bound as
    /// constants, ordered `$1..$n`, so the plan is fully optimized for exactly
    /// these values.
    pub fn plan_with_values(
        &mut self,
        query: &str,
        transaction: Arc<dyn DuckDBTransaction>,
        values: &[ScalarValue],
    ) -> Result<Plan, Error> {
        let mut params = ffi::param_list_new()?;
        for (i, value) in values.iter().enumerate() {
            let index = u32::try_from(i + 1).map_err(|_| {
                Error::Bridge(format!("too many prepared-statement parameters: {}", i + 1))
            })?;
            push_param_value(params.pin_mut(), index, value)?;
        }

        let transaction_ctx = catalog_provider::TransactionContext::new(transaction);
        let result = ffi::extract_plan_with_values(
            self.cxx_context.pin_mut(),
            query,
            &transaction_ctx,
            &params,
        )?;
        if !result.error_kind.is_empty() {
            return Err(bridge_error(&result));
        }

        let ffi::ExtractPlanResult {
            plan, output_names, ..
        } = result;
        Ok(Plan::new(plan, output_names))
    }
}

/// Push one parameter value into the FFI list under its 1-based position,
/// dispatching to the typed push function for the value's variant.
fn push_param_value(
    list: std::pin::Pin<&mut ffi::ParamValueList>,
    index: u32,
    value: &ScalarValue,
) -> Result<(), Error> {
    match value {
        ScalarValue::Boolean(v) => ffi::param_list_push_bool(list, index, *v)?,
        ScalarValue::Int8(v) => ffi::param_list_push_i8(list, index, *v)?,
        ScalarValue::Int16(v) => ffi::param_list_push_i16(list, index, *v)?,
        ScalarValue::Int32(v) => ffi::param_list_push_i32(list, index, *v)?,
        ScalarValue::Int64(v) => ffi::param_list_push_i64(list, index, *v)?,
        ScalarValue::UInt8(v) => ffi::param_list_push_u8(list, index, *v)?,
        ScalarValue::UInt16(v) => ffi::param_list_push_u16(list, index, *v)?,
        ScalarValue::UInt32(v) => ffi::param_list_push_u32(list, index, *v)?,
        ScalarValue::UInt64(v) => ffi::param_list_push_u64(list, index, *v)?,
        ScalarValue::Int128(v) => {
            ffi::param_list_push_hugeint(list, index, (v >> 64) as i64, *v as u64)?
        }
        ScalarValue::Float32(v) => ffi::param_list_push_f32(list, index, *v)?,
        ScalarValue::Float64(v) => ffi::param_list_push_f64(list, index, *v)?,
        ScalarValue::Decimal {
            value,
            width,
            scale,
        } => ffi::param_list_push_decimal(
            list,
            index,
            (value >> 64) as i64,
            *value as u64,
            *width,
            *scale,
        )?,
        ScalarValue::Utf8(v) => ffi::param_list_push_string(list, index, v)?,
        ScalarValue::Date(days) => ffi::param_list_push_date(list, index, *days)?,
        ScalarValue::Timestamp(micros) => ffi::param_list_push_timestamp(list, index, *micros)?,
        ScalarValue::Interval {
            months,
            days,
            micros,
        } => ffi::param_list_push_interval(list, index, *months, *days, *micros)?,
        ScalarValue::Null(ty) => ffi::param_list_push_null(list, index, ty.to_bridge())?,
        ScalarValue::Variant(_) | ScalarValue::Other(_) => {
            return Err(Error::Bridge(format!(
                "unsupported prepared-statement parameter value for ${index}: {value}"
            )));
        }
    }
    Ok(())
}
