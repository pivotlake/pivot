//! The two [`AggregationValue`](super::AggregationValue) containers built on
//! [`Slot`]s (each a [`Read`](super::read::Read) + [`Fold`](super::fold::Fold)):
//!
//! - [`Compiled`] — a fixed tuple of slots; branch-free, any mix (numeric and/or
//!   string), each slot reading its own typed array.
//! - [`Dynamic`] — a runtime signature folded per-slot by kind, generic over the
//!   accumulator width.

mod compiled;
mod dynamic;
mod mono;
mod patched;

pub use compiled::{
    Compiled, CountSlot, MaxSlot, MinSlot, OpTuple, StrMaxSlot, StrMinSlot, SumSlot,
};
pub use dynamic::Dynamic;
pub use mono::Mono;
pub use patched::Patched;
