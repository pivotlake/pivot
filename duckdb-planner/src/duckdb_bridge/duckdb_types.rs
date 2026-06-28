//! Auto-generated Rust mirrors of DuckDB C++ enums via `autocxx`.
//!
//! The [`include_cpp!`] macro pulls in the DuckDB enum headers and generates
//! Rust types that can be used for discriminant matching when building the plan.

use autocxx::prelude::*;

include_cpp! {
    #include "duckdb/common/enums/logical_operator_type.hpp"
    #include "duckdb/common/types.hpp"
    #include "duckdb/common/enums/expression_type.hpp"
    #include "duckdb/common/enums/order_type.hpp"
    #include "duckdb/planner/table_filter.hpp"
    #include "duckdb/planner/bound_result_modifier.hpp"
    safety!(unsafe)
    generate!("duckdb::LogicalOperatorType")
    generate!("duckdb::LogicalTypeId")
    generate!("duckdb::ExpressionType")
    generate!("duckdb::OrderType")
    generate!("duckdb::LogicalTypeId")
    generate!("duckdb::TableFilterType")
    generate!("duckdb::LimitNodeType")
}

pub use ffi::duckdb::ExpressionType;
pub use ffi::duckdb::LimitNodeType;
pub use ffi::duckdb::LogicalOperatorType;
pub use ffi::duckdb::LogicalTypeId;
pub use ffi::duckdb::OrderType;
pub use ffi::duckdb::TableFilterType;
