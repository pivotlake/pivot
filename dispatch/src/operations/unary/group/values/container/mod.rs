//! The [`AggregationValue`](super::AggregationValue) containers, each built on
//! slots that read a column ([`Read`](super::read::Read)) and fold it:
//!
//! - [`Compiled`] — a fixed *numeric* tuple of slots; branch-free, each slot
//!   reading its own typed array via its [`Fold`](super::fold::Fold).
//! - [`Dynamic`] — a runtime signature (numeric and/or string) folded per-slot by
//!   kind, generic over the accumulator width and an `ONLY_ADDITIVE` fast-path
//!   flag. A string extreme persists its winner into the value arena via the
//!   inherent [`StrMin`](super::fold::StrMin)/[`StrMax`](super::fold::StrMax)
//!   methods; numeric arms drive the contextless [`Fold`](super::fold::Fold).
//! - [`Variable`] — a runtime signature of runtime *arity* (a slot count no
//!   [`Dynamic`] arity is monomorphised for). Folds per-slot exactly as
//!   [`Dynamic`] does; its cells sit inline in the hash entry just the same,
//!   with the count fixed at query build instead of in the type.

mod compiled;
mod dynamic;
mod variable;

pub use compiled::{Compiled, CountSlot, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use dynamic::Dynamic;
pub use variable::Variable;
