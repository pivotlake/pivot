//! Concrete representations of per-group aggregation state.
//!
//! - [`Compiled`] is a fixed, statically typed tuple of numeric operations.
//! - [`RuntimeAggregation`] is a query-defined cell array that supports
//!   arbitrary arity and string extrema.

mod compiled;
mod runtime;

pub use compiled::{Compiled, CountSlot, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use runtime::{RuntimeAggregation, RuntimeAggregationContext};
