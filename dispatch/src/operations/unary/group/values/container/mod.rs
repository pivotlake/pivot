//! The two [`AggregationValue`](super::AggregationValue) containers built on the
//! [`Aggregation`](super::aggregation::Aggregation) ops:
//!
//! - [`Compiled`] — a fixed tuple of ops; branch-free, any mix (numeric and/or
//!   string), each slot reading its own typed array.
//! - [`Dynamic`] — a runtime numeric signature folded per-slot by kind, generic
//!   over the accumulator width.

mod compiled;
mod dynamic;

pub use compiled::{Compiled, OpTuple};
pub use dynamic::Dynamic;
