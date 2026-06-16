//! The three [`AggregationValue`](super::AggregationValue) containers, each a way
//! of composing the [`read`](super::read) × [`fold`](super::fold) axes:
//!
//! - [`Mono`] — one fold, per-slot read; the homogeneous fast path (and string
//!   `MIN`/`MAX`).
//! - [`Compiled`] — a tuple of whole [`Op`](super::op::Op) atoms; branch-free, any
//!   mix (including string + numeric).
//! - [`Dynamic`] — runtime per-slot kind dispatch; the numeric fallback.

mod compiled;
mod dynamic;
mod mono;

pub use compiled::{Compiled, OpTuple};
pub use dynamic::Dynamic;
pub use mono::Mono;
