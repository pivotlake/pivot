//! The **[`Fold`]** trait — one per aggregate *op*, over the value a
//! [`Read`](super::read::Read) already produced. A fold knows nothing about the
//! column it came from: an integer `Sum` folds an `i64` whatever width it was
//! read at, so there is one `Sum`, not one per width. The read/fold split is what
//! collapses the `ops × widths` signature space to `ops + widths`.
//!
//! The ops:
//! - [`Count`] — folds `()` (reads no column).
//! - [`Sum<A>`](Sum) / [`WideSum`] — folds `i64`, accumulating in `A` (`i64` /
//!   `i128`).
//! - [`Min<A>`](Min) / [`Max<A>`](Max) — fold `i64`.
//! - [`StrMin`] / [`StrMax`] — fold `&str`, comparing against the current winner
//!   and persisting (lazily) only when the new string wins.
//!
//! A fixed signature is a tuple of (read, fold) pairs
//! ([`Compiled`](super::container::Compiled)); a runtime signature folds each slot
//! by its kind ([`Dynamic`](super::container::Dynamic)). Both drive the *same*
//! `F::update(cell, R::read(input, idx), arena, cfg)` — strings and integers
//! alike, no branch.

mod count;
mod extreme;
mod string;
mod sum;

pub use count::Count;
pub use extreme::{Max, Min};
pub use string::{StrMax, StrMin};
pub use sum::{Sum, WideSum};

use super::cell::Cell;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::sync::Arc;

/// One aggregate op, folding a value `V` (what its [`Read`](super::read::Read)
/// yields) into an accumulator cell. The seed/update/merge split mirrors the
/// GROUP BY consume phases — `seed` a new group from a row's value, `update`
/// folds the next value in (a string extreme persists only a winner here),
/// `merge` combines two finished partials.
pub trait Fold<V>: Send + Sync + 'static {
    /// This op's accumulator cell (`i64` / `i128` / `ArenaKey`).
    type Acc: Cell;
    /// Runtime data [`merge`](Self::merge) / [`update`](Self::update) need that
    /// the type can't carry: the value arena for a string extreme; `()` otherwise.
    type Cfg: Clone + Send + Sync + 'static;

    /// Build the fold config from the value arena.
    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;

    /// Materialise a new group's cell from a row's value.
    fn seed(v: V, arena: &mut WorkerArena) -> Self::Acc;
    /// Fold a value into an existing cell (a string extreme compares the raw
    /// `&str` and persists only when it wins).
    fn update(acc: Self::Acc, v: V, arena: &mut WorkerArena, cfg: &Self::Cfg) -> Self::Acc;
    /// Combine two finished partials — the partition merge and radix fold.
    fn merge(a: Self::Acc, b: Self::Acc, cfg: &Self::Cfg) -> Self::Acc;

    /// This cell as an `ORDER BY <agg>` sort key (widened to `i128`).
    fn sort_key(acc: Self::Acc) -> i128;

    /// Render a finished column of cells into the Arrow array + field.
    fn finish(
        name: &str,
        col: SlabColumn<Self::Acc>,
        arena: &Arc<SharedArena>,
    ) -> (Field, ArrayRef);
}
