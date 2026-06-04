//! duckdb-planner: a Rust wrapper around DuckDB's C++ query planner.
//!
//! This crate sends SQL strings to an embedded DuckDB instance, deserializes it into a Rust [`PlanNode`] tree.
//!
//! The main entry point is [`PlannerContext`], which owns the DuckDB state and
//! exposes [`plan`](PlannerContext::plan) (returns a [`PlanNode`]).
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
//! // Plan a query — returns a PlanNode tree.
//! let plan = ctx.plan("SELECT name FROM users").unwrap();
//!
//! // Pattern-match on the operator to extract details.
//! match &plan.operator {
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
mod types;

use std::sync::Arc;

use custom_deserializer::CustomDeserializer;
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
    #[error("Deserialization error: {0}")]
    SerdeDeserialize(#[from] serde_json::Error),
}

/// Error returned by DuckDB when it cannot produce a plan for a query.
#[derive(CustomDeserializer, Debug, Error)]
pub struct PlanningError {
    pub exception_message: String,
    pub position: Option<String>,
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.exception_message)
    }
}

#[derive(CustomDeserializer)]
struct BridgeErrorPayload {
    kind: String,
    exception_message: String,
    position: Option<String>,
}

impl BridgeErrorPayload {
    fn into_error(self) -> Error {
        match self.kind.as_str() {
            "duckdb_planning" => Error::DuckDBPlanning(PlanningError {
                exception_message: self.exception_message,
                position: self.position,
            }),
            "unsupported_plan" => Error::UnsupportedPlan(self.exception_message),
            "bridge_error" => Error::Bridge(self.exception_message),
            other => Error::Bridge(format!(
                "Unknown bridge error kind `{other}`: {}",
                self.exception_message
            )),
        }
    }
}

/// Internal wrapper for the JSON response from the C++ bridge, which is
/// either a successfully planned [`PlanNode`] or a structured bridge error.
#[derive(CustomDeserializer)]
enum PlanResult {
    Success(PlanNode),
    Error(BridgeErrorPayload),
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

    /// Plan a SQL query: sends the query to DuckDB, deserializes the JSON
    /// logical plan into a [`PlanNode`] tree, and attaches the `DuckDBTable`
    /// trait objects to each `Input` node.
    pub fn plan(&mut self, query: &str) -> Result<PlanNode, Error> {
        let result = ffi::extract_plan(self.cxx_context.pin_mut(), query);
        let plan: PlanResult = serde_json::from_str(&result.json)?;
        match plan {
            PlanResult::Success(plan) => {
                let tables: Vec<Box<dyn catalog_provider::DuckDBTable>> = result
                    .tables
                    .into_iter()
                    .map(|ot| ot.table.expect("planner returned an unbound table"))
                    .collect();
                Ok(plan.resolve_inputs(tables))
            }
            PlanResult::Error(err) => Err(err.into_error()),
        }
    }
}
