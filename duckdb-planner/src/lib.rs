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
//!     DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId, PlannerContext,
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
//!             not_null: false,
//!         }]
//!     }
//! }
//!
//! struct MyCatalog;
//!
//! impl DuckDBBind for MyCatalog {
//!     fn try_bind(&self, name: &str) -> Option<Box<dyn DuckDBTable>> {
//!         match name {
//!             "users" => Some(Box::new(UsersTable)),
//!             _ => None,
//!         }
//!     }
//! }
//!
//! let mut ctx = PlannerContext::new(Arc::new(MyCatalog));
//!
//! // Plan a query; walk the root handle.
//! let plan = ctx.plan("SELECT name FROM users").unwrap();
//! let root = plan.root();
//! assert_eq!(root.op_type(), LogicalOperatorType::LOGICAL_PROJECTION);
//! ```

#![allow(clippy::upper_case_acronyms)]

pub mod catalog_provider;
pub mod duckdb_bridge;
pub mod handle;
mod types;

use std::sync::Arc;

use duckdb_bridge::ffi;
use thiserror::Error;

pub use catalog_provider::{DuckDBBind, DuckDBTable};
pub use duckdb_bridge::duckdb_types::LogicalTypeId;
pub use duckdb_bridge::ffi::DuckDBColumn;
pub use handle::{Expr, LogicalOp, Plan};
pub use types::ScalarValue;

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
    /// Create a planner context backed by the given catalog provider.
    pub fn new(catalog: Arc<dyn DuckDBBind>) -> Self {
        let ctx = Box::new(catalog_provider::CatalogContext::new(catalog));
        Self {
            cxx_context: ffi::new_context(ctx),
        }
    }

    /// Plan a SQL query: sends the query to DuckDB and returns a [`Plan`] handle
    /// owning the resolved logical plan. The caller walks it through
    /// [`Plan::root`] and the [`LogicalOp`]/[`Expr`] accessors.
    pub fn plan(&mut self, query: &str) -> Result<Plan, Error> {
        let result = ffi::extract_plan(self.cxx_context.pin_mut(), query);
        if !result.error_kind.is_empty() {
            return Err(bridge_error(&result));
        }

        let ffi::ExtractPlanResult {
            plan, output_names, ..
        } = result;
        Ok(Plan::new(plan, output_names))
    }
}
