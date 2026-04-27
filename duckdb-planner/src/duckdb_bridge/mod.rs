//! FFI layer to the DuckDB C++ planner.
//!
//! * [`ffi`] — `cxx`-generated bridge exposing `DuckPlannerContext`,
//!   `new_context` and `extract_plan`.
//! * [`duckdb_types`] — `autocxx`-generated Rust mirrors of DuckDB enums
//!   (`LogicalOperatorType`, `ExpressionType`, etc.).

pub mod duckdb_types;

use crate::catalog_provider::{
    CatalogContext, OptionalTableWrapper, catalog_get_table, pushdown_filter,
};

/// CXX bridge to the hand-written C++ glue in `bridge.cpp` / `bridge.h`.
#[cxx::bridge]
pub mod ffi {
    /// Column definition for FFI — carries the type as a `u8` discriminant
    /// because CXX cannot pass Rust enums across the bridge.
    struct DuckDBColumn {
        pub name: String,
        pub duckdb_logical_type_id: u8,
    }

    /// Result of a catalog table lookup, allowing C++ to inspect the outcome.
    struct CatalogGetTableResult {
        pub found: bool,
        pub columns: Vec<DuckDBColumn>,
        /// Opaque handle to the Rust `DuckDBTable` object that was looked up.
        pub table: Box<OptionalTableWrapper>,
    }

    /// Result of `extract_plan`: the JSON plan string plus the `OptionalTableWrapper`
    /// boxes collected from each LogicalGet node during serialization.
    struct ExtractPlanResult {
        pub json: String,
        pub tables: Vec<Box<OptionalTableWrapper>>,
    }

    extern "Rust" {
        type CatalogContext;
        type OptionalTableWrapper;
        fn catalog_get_table(ctx: &CatalogContext, name: &str) -> CatalogGetTableResult;
        fn pushdown_filter(table: &OptionalTableWrapper, filters_json: &str) -> bool;
    }

    unsafe extern "C++" {
        include!("duckdb-planner/src/duckdb_bridge/cpp/bridge.h");
        type DuckPlannerContext;

        fn new_context(catalog: Box<CatalogContext>) -> UniquePtr<DuckPlannerContext>;
        fn extract_plan(ctx: Pin<&mut DuckPlannerContext>, query: &str) -> ExtractPlanResult;
    }
}
