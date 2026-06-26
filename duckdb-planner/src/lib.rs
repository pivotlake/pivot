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
mod from_fb;
pub mod operator;
pub mod plan;
mod types;

use std::sync::Arc;

use duckdb_bridge::ffi;
use duckdb_bridge::plan_fb as fb;
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
    #[error("Plan decode error: {0}")]
    Decode(String),
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

/// Translate the bridge's structured error message into an [`Error`].
fn bridge_error(err: fb::BridgeError) -> Error {
    let message = err.exception_message().unwrap_or_default().to_string();
    let position = err.position().map(str::to_string);
    match err.kind().unwrap_or_default() {
        "duckdb_planning" => Error::DuckDBPlanning(PlanningError {
            exception_message: message,
            position,
        }),
        "unsupported_plan" => Error::UnsupportedPlan(message),
        "bridge_error" => Error::Bridge(message),
        other => Error::Bridge(format!("Unknown bridge error kind `{other}`: {message}")),
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

    /// Plan a SQL query: sends the query to DuckDB, reads the FlatBuffers logical
    /// plan into a [`PlanNode`] tree, and attaches the `DuckDBTable` trait objects
    /// to each `Input` node.
    pub fn plan(&mut self, query: &str) -> Result<PlannedQuery, Error> {
        let result = ffi::extract_plan(self.cxx_context.pin_mut(), query);
        // SAFETY: the buffer was produced by our own in-process C++ bridge, so
        // skip the FlatBuffers verifier: verification is redundant for trusted
        // data and its default depth cap (64) would reject deeply nested
        // expression trees the writer imposes no limit on.
        let plan = unsafe { flatbuffers::root_unchecked::<fb::PlanResult>(&result.plan) };
        match plan.result_type() {
            fb::PlanResultKind::SuccessPayload => {
                let payload = plan
                    .result_as_success_payload()
                    .ok_or_else(|| Error::Decode("missing success payload".to_string()))?;
                let tables: Vec<Box<dyn catalog_provider::DuckDBTable>> = result
                    .tables
                    .into_iter()
                    .map(|ot| ot.table.expect("planner returned an unbound table"))
                    .collect();
                let root = from_fb::decode_plan_node(payload.plan())?;
                let output_names = payload
                    .output_names()
                    .map(|names| names.iter().map(str::to_string).collect())
                    .unwrap_or_default();
                Ok(PlannedQuery {
                    root: root.resolve_inputs(tables),
                    output_names,
                })
            }
            fb::PlanResultKind::BridgeError => {
                let err = plan
                    .result_as_bridge_error()
                    .ok_or_else(|| Error::Decode("missing bridge error".to_string()))?;
                Err(bridge_error(err))
            }
            other => Err(Error::Decode(format!("unknown plan result kind {}", other.0))),
        }
    }
}
