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
    #include "duckdb/common/enums/join_type.hpp"
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
    generate!("duckdb::JoinType")
}

pub use ffi::duckdb::ExpressionType;
pub use ffi::duckdb::JoinType;
pub use ffi::duckdb::LimitNodeType;
pub use ffi::duckdb::LogicalOperatorType;
pub use ffi::duckdb::LogicalTypeId;
pub use ffi::duckdb::OrderType;
pub use ffi::duckdb::TableFilterType;

use std::fmt;

/// Give each DuckDB enum a `from_u8` decoder plus numeric `Debug`/`Display`.
/// DuckDB defines them as `enum class : uint8_t`, so the `u8` discriminant the
/// bridge reports round-trips exactly through a transmute, and the formatting
/// impls just render that discriminant. `from_u8` is `pub(crate)` on purpose:
/// only the handle layer decodes raw bytes, and callers get typed accessors
/// rather than a way to forge an invalid enum.
macro_rules! impl_duckdb_enum {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $ty {
                pub(crate) fn from_u8(value: u8) -> Self {
                    unsafe { std::mem::transmute::<u8, $ty>(value) }
                }
            }

            impl fmt::Debug for $ty {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.clone() as u8)
                }
            }

            impl fmt::Display for $ty {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.clone() as u8)
                }
            }
        )+
    };
}

impl_duckdb_enum!(
    LogicalOperatorType,
    ExpressionType,
    JoinType,
    LimitNodeType,
    LogicalTypeId,
    OrderType,
);
