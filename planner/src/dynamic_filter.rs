//! Planner-side view of a dynamic-filter reference.
//!
//! Mirrors [`duckdb_planner::dynamic_filter::DynamicFilter`] but with the
//! comparison translated into the planner's [`CompareType`]. This is pure plan
//! data: it carries the `slot_id` that correlates a producer (`TopN`) with its
//! consumer scans, but *not* the shared slot itself. The
//! [`DynamicFilterSlot`](dispatch::DynamicFilterSlot) is allocated fresh on each
//! [`Plan::compile`](crate::Plan::compile) and handed to the operators by
//! `slot_id`, so a cached `Plan` holds no runtime state.

use std::fmt;

use duckdb_planner::dynamic_filter as duckdb_dynamic_filter;

use crate::expression::{self, CompareType};

/// A dynamic-filter reference on an operator: which column it tests, the
/// comparison, and the `slot_id` correlating it with the other end of the
/// producer↔consumer pair. The shared slot is resolved at compile time (see
/// [`Plan::compile`](crate::Plan::compile)).
#[derive(Debug)]
pub struct DynamicFilter {
    pub slot_id: usize,
    pub column_idx: usize,
    pub compare_type: CompareType,
}

impl TryFrom<duckdb_dynamic_filter::DynamicFilter> for DynamicFilter {
    type Error = expression::Error;

    fn try_from(df: duckdb_dynamic_filter::DynamicFilter) -> Result<Self, Self::Error> {
        Ok(DynamicFilter {
            slot_id: df.slot_id,
            column_idx: df.column_idx,
            compare_type: df.compare_type.try_into()?,
        })
    }
}

impl fmt::Display for DynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynamicFilter(#{} {} ?)",
            self.column_idx, self.compare_type
        )
    }
}
