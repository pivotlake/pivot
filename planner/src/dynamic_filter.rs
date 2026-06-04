//! Planner-side view of a dynamic-filter cell.
//!
//! Mirrors [`duckdb_planner::dynamic_filter::DynamicFilter`] but with the
//! comparison translated into the planner's [`CompareType`]. Once resolved, the
//! `Arc<DynamicFilterSlot>` is the cell's identity — `slot_id` was only needed
//! to rebuild the shared-slot graph during deserialization and isn't carried
//! forward.

use std::fmt;
use std::sync::Arc;

use duckdb_planner::dynamic_filter as duckdb_dynamic_filter;
use duckdb_planner::dynamic_filter::DynamicFilterSlot;

use crate::expression::{self, CompareType};

/// A resolved dynamic filter: which column it tests, the comparison, and the
/// shared slot the producer publishes the boundary value into.
#[derive(Debug)]
pub struct DynamicFilter {
    pub column_idx: usize,
    pub compare_type: CompareType,
    pub slot: Arc<DynamicFilterSlot>,
}

impl TryFrom<duckdb_dynamic_filter::DynamicFilter> for DynamicFilter {
    type Error = expression::Error;

    fn try_from(df: duckdb_dynamic_filter::DynamicFilter) -> Result<Self, Self::Error> {
        let slot = df
            .slot
            .expect("DynamicFilter must be slot-resolved before reaching the planner");
        Ok(DynamicFilter {
            column_idx: df.column_idx,
            compare_type: df.compare_type.try_into()?,
            slot,
        })
    }
}

impl fmt::Display for DynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DynamicFilter(#{} {} ?)", self.column_idx, self.compare_type)
    }
}
