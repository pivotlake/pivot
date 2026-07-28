//! GROUP BY aggregation containers.
//!
//! - [`Compiled`] stores a fixed tuple of numeric slots.
//! - [`Dynamic`] stores any runtime number and mix of supported slots.

mod compiled;
mod dynamic;

pub use compiled::{Compiled, CountSlot, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use dynamic::Dynamic;
