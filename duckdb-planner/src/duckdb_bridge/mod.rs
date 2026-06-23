//! FFI layer to the DuckDB C++ planner.
//!
//! * [`ffi`] — `cxx`-generated bridge exposing `DuckPlannerContext`,
//!   `new_context` and `extract_plan`.
//! * [`duckdb_types`] — `autocxx`-generated Rust mirrors of DuckDB enums
//!   (`LogicalOperatorType`, `ExpressionType`, etc.).

pub mod duckdb_types;

use crate::catalog_provider::{
    CatalogContext, OptionalTableWrapper, catalog_get_scalar_function, catalog_get_table,
    catalog_get_table_function, pushdown_filter,
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

    /// Result of a catalog table-function lookup: the function's argument types
    /// and full output columns (DuckDB logical type id discriminants), or
    /// `found = false` if the provider has no such function.
    struct CatalogGetTableFunctionResult {
        pub found: bool,
        pub arg_type_ids: Vec<u8>,
        pub columns: Vec<DuckDBColumn>,
    }

    /// Result of a catalog scalar-function lookup: the function's argument and
    /// return types (DuckDB logical type id discriminants) and whether it must be
    /// registered `VOLATILE`, or `found = false` if the provider has no such
    /// function.
    struct CatalogGetScalarFunctionResult {
        pub found: bool,
        pub arg_type_ids: Vec<u8>,
        pub return_type_id: u8,
        pub is_volatile: bool,
    }

    extern "Rust" {
        type CatalogContext;
        type OptionalTableWrapper;
        fn catalog_get_table(ctx: &CatalogContext, name: &str) -> CatalogGetTableResult;
        fn catalog_get_table_function(
            ctx: &CatalogContext,
            name: &str,
        ) -> CatalogGetTableFunctionResult;
        fn catalog_get_scalar_function(
            ctx: &CatalogContext,
            name: &str,
        ) -> CatalogGetScalarFunctionResult;
        fn pushdown_filter(table: &mut OptionalTableWrapper, filters_json: &str) -> Result<bool>;
    }

    unsafe extern "C++" {
        include!("duckdb-planner/src/duckdb_bridge/cpp/bridge.h");
        type DuckPlannerContext;

        fn new_context(catalog: Box<CatalogContext>) -> UniquePtr<DuckPlannerContext>;
        fn extract_plan(ctx: Pin<&mut DuckPlannerContext>, query: &str) -> ExtractPlanResult;
    }
}
