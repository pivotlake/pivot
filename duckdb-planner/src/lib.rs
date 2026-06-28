//! duckdb-planner: a Rust wrapper around DuckDB's C++ query planner.
//!
//! This crate sends SQL strings to an embedded DuckDB instance and builds a Rust [`PlanNode`] tree from the resulting plan.
//!
//! The main entry point is [`PlannerContext`], which owns the DuckDB state and
//! exposes [`plan`](PlannerContext::plan) (returns a [`PlannedQuery`]).
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use duckdb_planner::{
//!     DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId, Operator, PlannerContext,
//! };
//!
//! struct UsersTable;
//!
//! impl DuckDBTable for UsersTable {
//!     fn clone_box(&self) -> Box<dyn DuckDBTable> { Box::new(UsersTable) }
//!     fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
//!         vec![DuckDBColumn {
//!             name: "name".to_string(),
//!             duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
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
//! // Plan a query; the returned PlannedQuery's `root` is the PlanNode tree.
//! let plan = ctx.plan("SELECT name FROM users").unwrap();
//!
//! // Pattern-match on the operator to extract details.
//! match &plan.root.operator {
//!     Operator::Projection(p) => assert_eq!(p.projections.len(), 1),
//!     other => println!("unexpected root: {other}"),
//! }
//! ```

#![allow(clippy::upper_case_acronyms)]

pub mod catalog_provider;
pub mod duckdb_bridge;
pub mod dynamic_filter;
pub mod expression;
pub mod operator;
pub mod plan;
mod plan_build;
mod types;

use std::sync::Arc;

use duckdb_bridge::ffi;
use thiserror::Error;

pub use catalog_provider::{DuckDBBind, DuckDBTable};
pub use duckdb_bridge::duckdb_types::LogicalTypeId;
pub use duckdb_bridge::ffi::DuckDBColumn;
pub use operator::Operator;
pub use plan::PlanNode;
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

impl From<plan_build::BuildError> for Error {
    fn from(err: plan_build::BuildError) -> Self {
        // The builder only fails on plan shapes pivot doesn't support yet (an
        // unmapped operator/expression, a non-constant LIMIT, ...), which the
        // old C++ path surfaced as an `unsupported_plan` error.
        Error::UnsupportedPlan(err.0)
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

/// A successfully planned query: the operator tree plus the result column names
/// DuckDB resolved for the client, in select order (e.g. `["hour",
/// "count_star()"]`). `output_names` may be empty when the bridge could not
/// recover them, in which case the caller keeps the operators' own names.
pub struct PlannedQuery {
    pub root: PlanNode,
    pub output_names: Vec<String>,
}

impl std::fmt::Display for PlannedQuery {
    /// Renders the plan tree (the `root`); the resolved `output_names` are
    /// omitted, matching how the higher-level `Plan` displays as its root.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.root)
    }
}

/// Translate the error fields the bridge reports into a typed [`Error`].
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

    /// Plan a SQL query: sends the query to DuckDB, builds a [`PlanNode`] tree
    /// directly from the C++ plan the bridge exposes, and attaches the
    /// `DuckDBTable` trait objects to each `Input` node. Returns the resolved
    /// tree alongside the client-facing result column names.
    pub fn plan(&mut self, query: &str) -> Result<PlannedQuery, Error> {
        let result = ffi::extract_plan(self.cxx_context.pin_mut(), query);
        if !result.error_kind.is_empty() {
            return Err(bridge_error(&result));
        }

        let root = ffi::plan_root(result.plan.as_ref().expect("plan present on success"));
        // Walking the live plan takes the bound table handles out of the catalog
        // entries (in scan order); `resolve_inputs` then binds them by index.
        let mut tables = Vec::new();
        let plan = plan_build::build_plan(root, &mut tables)?;

        let tables: Vec<Box<dyn catalog_provider::DuckDBTable>> = tables
            .into_iter()
            .map(|ot| ot.table.expect("planner returned an unbound table"))
            .collect();
        Ok(PlannedQuery {
            root: plan.resolve_inputs(tables),
            output_names: result.output_names,
        })
    }
}
