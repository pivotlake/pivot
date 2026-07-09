//! FFI layer to the DuckDB C++ planner.
//!
//! * [`ffi`] — `cxx`-generated bridge exposing `DuckPlannerContext`,
//!   `new_context`, `extract_plan`, and the accessor functions the Rust plan
//!   builder uses to read DuckDB's own plan objects.
//! * [`duckdb_types`] — `autocxx`-generated Rust mirrors of DuckDB enums
//!   (`LogicalOperatorType`, `ExpressionType`, etc.).

pub mod duckdb_types;

use crate::catalog_provider::{
    CatalogContext, OptionalTableWrapper, catalog_get_scalar_function, catalog_get_table,
    catalog_get_table_function, pushdown_filter,
};

/// CXX bridge to the hand-written C++ glue in `bridge.cpp` / `bridge.h`.
///
/// `extract_plan` hands back an opaque [`PlanHandle`] that owns DuckDB's
/// resolved `LogicalOperator` tree. The Rust plan builder walks that live tree
/// directly through the accessor functions below (`lo_*` for operators, `expr_*`
/// for expressions): there is no intermediate representation, the Rust side just
/// reads fields straight off the C++ objects.
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

    /// Outcome of `extract_plan`.
    ///
    /// On success `error_kind` is empty and `plan` owns DuckDB's resolved plan
    /// tree (kept alive for the duration of the Rust walk). On failure `plan` is
    /// null and the `error_*` fields describe what went wrong (`error_kind` is
    /// one of `duckdb_planning`, `unsupported_plan`, `bridge_error`).
    struct ExtractPlanResult {
        pub error_kind: String,
        pub error_message: String,
        pub has_error_position: bool,
        pub error_position: String,
        pub plan: UniquePtr<PlanHandle>,
        /// The binder-resolved result column names DuckDB would hand a client,
        /// in select order (e.g. `["hour", "count_star()"]`). Empty when the
        /// bridge could not recover them.
        pub output_names: Vec<String>,
    }

    extern "Rust" {
        type CatalogContext;
        type OptionalTableWrapper;
        /// `statement_handle` is the opaque per-statement context the planner
        /// threaded into [`extract_plan`], carried here by the DuckDB transaction
        /// (see `PivotTransaction`). The provider reinterprets it to resolve the
        /// table against that statement's catalog snapshot; `0` means none bound.
        fn catalog_get_table(
            ctx: &CatalogContext,
            name: &str,
            statement_handle: usize,
        ) -> CatalogGetTableResult;
        fn catalog_get_table_function(
            ctx: &CatalogContext,
            name: &str,
        ) -> CatalogGetTableFunctionResult;
        fn catalog_get_scalar_function(
            ctx: &CatalogContext,
            name: &str,
        ) -> CatalogGetScalarFunctionResult;
        fn pushdown_filter(table: &mut OptionalTableWrapper, expr: &Expression) -> Result<bool>;
    }

    unsafe extern "C++" {
        include!("duckdb-planner/src/duckdb_bridge/cpp/bridge.h");
        type DuckPlannerContext;

        /// Owns the resolved DuckDB plan tree returned by `extract_plan` (and
        /// frees the per-plan catalog entries when dropped); must stay alive
        /// while Rust walks it.
        type PlanHandle;
        /// A DuckDB `LogicalOperator`. Opaque; read via the `lo_*` accessors.
        type LogicalOperator;
        /// A DuckDB bound `Expression`. Opaque; read via the `expr_*` accessors.
        type Expression;
        /// An owning list of synthesized expressions (a `LogicalGet`'s
        /// pushed-down filter conditions). Read via `expr_list_*`.
        type ExpressionList;
        /// A DuckDB `Value` (a query constant or a table-function argument).
        /// Opaque; read via the `value_*` accessors after `value_type`.
        type Value;

        fn new_context(catalog: Box<CatalogContext>) -> UniquePtr<DuckPlannerContext>;
        /// `statement_handle` is an opaque per-statement context pointer; the
        /// bridge stores it on the statement's DuckDB transaction so the catalog's
        /// `LookupEntry` can hand it back to `catalog_get_table` during binding.
        fn extract_plan(
            ctx: Pin<&mut DuckPlannerContext>,
            query: &str,
            statement_handle: usize,
        ) -> ExtractPlanResult;

        fn plan_root(plan: &PlanHandle) -> &LogicalOperator;

        /// DuckDB's virtual row-id column identifier, used by the late-materialization
        /// row-id stripping to recognise the threaded-up row-id column.
        fn rowid_column_id() -> usize;

        // ---- LogicalOperator: shared structure ----
        /// DuckDB `LogicalOperatorType` discriminant.
        fn lo_type(op: &LogicalOperator) -> u8;
        /// The operator's display name (`LogicalOperator::GetName`).
        fn lo_name(op: &LogicalOperator) -> String;
        fn lo_child_count(op: &LogicalOperator) -> usize;
        fn lo_child(op: &LogicalOperator, index: usize) -> &LogicalOperator;

        // ---- Projection ----
        fn lo_projection_expr_count(op: &LogicalOperator) -> usize;
        fn lo_projection_expr(op: &LogicalOperator, index: usize) -> &Expression;

        // ---- Filter ----
        fn lo_filter_expr_count(op: &LogicalOperator) -> usize;
        fn lo_filter_expr(op: &LogicalOperator, index: usize) -> &Expression;
        /// A `LogicalFilter`'s `projection_map` (empty when it passes all child
        /// columns through). Each entry is a child-output column index to keep.
        fn lo_filter_projection_map_count(op: &LogicalOperator) -> usize;
        fn lo_filter_projection_map_index(op: &LogicalOperator, index: usize) -> usize;
        /// The `LogicalTypeId` of the filter's `index`th output column, paired
        /// with `lo_filter_projection_map_index` for the replay projection.
        fn lo_filter_type_id(op: &LogicalOperator, index: usize) -> u8;

        // ---- OrderBy ----
        fn lo_orderby_count(op: &LogicalOperator) -> usize;
        fn lo_orderby_direction(op: &LogicalOperator, index: usize) -> u8;
        fn lo_orderby_expr(op: &LogicalOperator, index: usize) -> &Expression;

        // ---- Aggregate ----
        fn lo_aggregate_group_count(op: &LogicalOperator) -> usize;
        fn lo_aggregate_group(op: &LogicalOperator, index: usize) -> &Expression;
        fn lo_aggregate_expr_count(op: &LogicalOperator) -> usize;
        fn lo_aggregate_expr(op: &LogicalOperator, index: usize) -> &Expression;

        // ---- TopN ----
        fn lo_topn_order_count(op: &LogicalOperator) -> usize;
        fn lo_topn_order_direction(op: &LogicalOperator, index: usize) -> u8;
        fn lo_topn_order_expr(op: &LogicalOperator, index: usize) -> &Expression;
        fn lo_topn_limit(op: &LogicalOperator) -> usize;
        fn lo_topn_offset(op: &LogicalOperator) -> usize;
        fn lo_topn_has_dynamic_filter(op: &LogicalOperator) -> bool;
        /// Pointer identity of the producer's shared `DynamicFilterData` cell,
        /// used to correlate it with consumer scans (the Rust side assigns slots).
        fn lo_topn_dynamic_filter_data_id(op: &LogicalOperator) -> usize;
        /// The child-output column index whose boundary the TopN publishes.
        fn lo_topn_dynamic_filter_column(op: &LogicalOperator) -> usize;
        fn lo_topn_dynamic_filter_comparison(op: &LogicalOperator) -> u8;

        // ---- Limit ----
        /// DuckDB `LimitNodeType` discriminant of a `LIMIT`/`OFFSET` bound
        /// (`UNSET` / `CONSTANT_VALUE` / a percentage or expression form). The
        /// constant value, when applicable, is read via `lo_limit_value` /
        /// `lo_limit_offset`.
        fn lo_limit_value_kind(op: &LogicalOperator) -> u8;
        fn lo_limit_value(op: &LogicalOperator) -> usize;
        fn lo_limit_offset_kind(op: &LogicalOperator) -> u8;
        fn lo_limit_offset(op: &LogicalOperator) -> usize;

        // ---- Get: base table ----
        /// Whether this `LogicalGet` reads a base table (vs a table function).
        fn lo_get_has_table(op: &LogicalOperator) -> bool;
        /// Move the pivot `DuckDBTable` handle out of the scan's catalog entry.
        /// Called once per base-table scan during the walk.
        fn lo_get_take_table(op: &LogicalOperator) -> Box<OptionalTableWrapper>;
        /// The scan's projected output columns (respecting `projection_ids`),
        /// each a storage column index + its `LogicalTypeId`.
        fn lo_get_output_count(op: &LogicalOperator) -> usize;
        fn lo_get_output_column(op: &LogicalOperator, index: usize) -> usize;
        fn lo_get_output_type(op: &LogicalOperator, index: usize) -> u8;
        /// The static `col op const` predicates DuckDB pushed into `table_filters`,
        /// rebuilt as expressions (owned by the returned list). Empty list when
        /// there are none.
        fn lo_get_pushed_conditions(op: &LogicalOperator) -> Result<UniquePtr<ExpressionList>>;
        /// Dynamic-filter consumers attached to the scan's `table_filters`.
        fn lo_get_dynamic_filter_count(op: &LogicalOperator) -> usize;
        fn lo_get_dynamic_filter_data_id(op: &LogicalOperator, index: usize) -> usize;
        fn lo_get_dynamic_filter_column(op: &LogicalOperator, index: usize) -> usize;
        fn lo_get_dynamic_filter_comparison(op: &LogicalOperator, index: usize) -> u8;

        // ---- Get: table function ----
        fn lo_get_function_name(op: &LogicalOperator) -> String;
        fn lo_get_has_named_params(op: &LogicalOperator) -> bool;
        fn lo_get_param_count(op: &LogicalOperator) -> usize;
        fn lo_get_param(op: &LogicalOperator, index: usize) -> &Value;

        // ---- CreateTable ----
        fn lo_create_table_name(op: &LogicalOperator) -> String;
        fn lo_create_column_count(op: &LogicalOperator) -> usize;
        fn lo_create_column_name(op: &LogicalOperator, index: usize) -> String;
        fn lo_create_column_type(op: &LogicalOperator, index: usize) -> u8;
        fn lo_create_option_count(op: &LogicalOperator) -> usize;
        fn lo_create_option_key(op: &LogicalOperator, index: usize) -> String;
        fn lo_create_option_value(op: &LogicalOperator, index: usize) -> String;
        fn lo_create_if_not_exists(op: &LogicalOperator) -> bool;
        fn lo_create_or_replace(op: &LogicalOperator) -> bool;
        fn lo_create_temporary(op: &LogicalOperator) -> bool;
        fn lo_create_has_query(op: &LogicalOperator) -> bool;
        fn lo_create_constraint_count(op: &LogicalOperator) -> usize;

        // ---- Set / Reset ----
        fn lo_set_name(op: &LogicalOperator) -> String;
        fn lo_set_value(op: &LogicalOperator) -> String;
        fn lo_reset_name(op: &LogicalOperator) -> String;

        // ---- ComparisonJoin: late materialization ----
        /// Whether this is the SEMI join DuckDB's late_materialization optimizer
        /// produces (vs a user IN/EXISTS), which the bridge collapses into a
        /// Materialize.
        fn lo_is_late_materialization_join(op: &LogicalOperator) -> bool;
        /// The full-column (LHS) side's output storage columns (row-id excluded),
        /// which the Materialize re-reads for the surviving rows.
        fn lo_late_materialization_column_count(op: &LogicalOperator) -> usize;
        fn lo_late_materialization_column(op: &LogicalOperator, index: usize) -> usize;

        // ---- ExpressionList (owned synthesized expressions) ----
        fn expr_list_count(list: &ExpressionList) -> usize;
        fn expr_list_get(list: &ExpressionList, index: usize) -> &Expression;

        // ---- Expression: shared ----
        /// DuckDB `ExpressionType` discriminant.
        fn expr_type(expr: &Expression) -> u8;
        /// `LogicalTypeId` of the expression's result.
        fn expr_return_type(expr: &Expression) -> u8;
        fn expr_has_alias(expr: &Expression) -> bool;
        fn expr_alias(expr: &Expression) -> String;

        // BoundReferenceExpression / BoundColumnRefExpression
        fn expr_ref_index(expr: &Expression) -> usize;
        fn expr_columnref_index(expr: &Expression) -> usize;

        // BoundComparisonExpression
        fn expr_comparison_left(expr: &Expression) -> &Expression;
        fn expr_comparison_right(expr: &Expression) -> &Expression;

        // BoundBetweenExpression
        fn expr_between_input(expr: &Expression) -> &Expression;
        fn expr_between_lower(expr: &Expression) -> &Expression;
        fn expr_between_upper(expr: &Expression) -> &Expression;
        fn expr_between_lower_inclusive(expr: &Expression) -> bool;
        fn expr_between_upper_inclusive(expr: &Expression) -> bool;

        // BoundConstantExpression
        fn expr_constant(expr: &Expression) -> &Value;

        // ---- Value: typed accessors (shared by constants and table-function
        // arguments). Read `value_type` first, then the matching accessor.
        fn value_type(v: &Value) -> u8;
        fn value_bool(v: &Value) -> bool;
        fn value_i8(v: &Value) -> i8;
        fn value_i16(v: &Value) -> i16;
        fn value_i32(v: &Value) -> i32;
        fn value_i64(v: &Value) -> i64;
        fn value_u8(v: &Value) -> u8;
        fn value_u16(v: &Value) -> u16;
        fn value_u32(v: &Value) -> u32;
        fn value_u64(v: &Value) -> u64;
        fn value_f32(v: &Value) -> f32;
        fn value_f64(v: &Value) -> f64;
        fn value_string(v: &Value) -> String;
        fn value_date(v: &Value) -> i32;
        fn value_timestamp(v: &Value) -> i64;
        fn value_interval_months(v: &Value) -> i32;
        fn value_interval_days(v: &Value) -> i32;
        fn value_interval_micros(v: &Value) -> i64;

        // BoundAggregateExpression
        fn expr_aggregate_name(expr: &Expression) -> String;
        fn expr_aggregate_distinct(expr: &Expression) -> bool;
        fn expr_aggregate_child_count(expr: &Expression) -> usize;
        fn expr_aggregate_child(expr: &Expression, index: usize) -> &Expression;

        // BoundFunctionExpression
        fn expr_function_name(expr: &Expression) -> String;
        fn expr_function_child_count(expr: &Expression) -> usize;
        fn expr_function_child(expr: &Expression, index: usize) -> &Expression;

        // BoundOperatorExpression (COMPARE_IN, OPERATOR_NOT)
        fn expr_operator_child_count(expr: &Expression) -> usize;
        fn expr_operator_child(expr: &Expression, index: usize) -> &Expression;

        // BoundConjunctionExpression
        fn expr_conjunction_child_count(expr: &Expression) -> usize;
        fn expr_conjunction_child(expr: &Expression, index: usize) -> &Expression;

        // BoundCaseExpression
        fn expr_case_check_count(expr: &Expression) -> usize;
        fn expr_case_when(expr: &Expression, index: usize) -> &Expression;
        fn expr_case_then(expr: &Expression, index: usize) -> &Expression;
        fn expr_case_else(expr: &Expression) -> &Expression;

        // BoundCastExpression
        fn expr_cast_child(expr: &Expression) -> &Expression;
    }
}
