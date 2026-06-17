//! The two [`AggregationValue`](super::AggregationValue) containers built on
//! [`Slot`]s (each a [`Read`](super::read::Read) + [`Fold`](super::aggregation::Fold)):
//!
//! - [`Compiled`] — a fixed tuple of slots; branch-free, any mix (numeric and/or
//!   string), each slot reading its own typed array.
//! - [`Dynamic`] — a runtime signature folded per-slot by kind, generic over the
//!   accumulator width.

mod compiled;
mod dynamic;
mod slot;

pub use compiled::{Compiled, OpTuple};
pub use dynamic::Dynamic;
pub use slot::{
    CountSlot, MaxSlot, MinSlot, Slot, StrMaxSlot, StrMinSlot, SumSlot,
};
