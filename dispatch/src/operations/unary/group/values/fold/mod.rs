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
use arrow_array::ArrayRef;
use arrow_schema::Field;

/// An op's *accumulator* behaviour — everything that doesn't depend on the read
/// value `V`: the cell type, how two partials combine, the sort key, the render.
///
/// Kept separate from [`Fold`] so a container can name `Acc`/`SharedContext` **without**
/// going through the `for<'b> Fold<…>` bound a borrowed read value (`&'b str`)
/// forces — that lifetime-carrying projection sends the monomorphisation
/// collector into a loop when the value flows through the top-k heap.
pub trait FoldAcc: Send + Sync + 'static {
    /// This op's accumulator cell (`i64` / `i128` / `ArenaKey`-in-`i128`).
    type Acc: Cell;
    /// The shared, read-side context [`merge`](Self::merge)/[`finish`](Self::finish)
    /// resolve through: the value arena (`Arc<SharedArena>`) for a string extreme,
    /// `()` for a numeric op.
    type SharedContext: Clone + Send + Sync + 'static;
    /// The per-worker write state [`seed`](Fold::seed)/[`update`](Fold::update)
    /// fold into during consume: `()` for a numeric op (it persists nothing), the
    /// concrete `WorkerArena` for a string extreme (it stores winners). A
    /// numeric op's `()` lets consume thread `&mut ()` — free, since a `()`
    /// reference can't alias the table the probe loop mutates.
    type WorkerContext;

    /// Combine two finished partials — the partition merge and radix fold.
    fn merge(a: Self::Acc, b: Self::Acc, ctx: &Self::SharedContext) -> Self::Acc;
    /// Render a finished column of cells into the Arrow array + field.
    fn finish(
        name: &str,
        col: SlabColumn<Self::Acc>,
        ctx: &Self::SharedContext,
    ) -> (Field, ArrayRef);
}

/// One aggregate op, folding a value `V` (what its [`Read`](super::read::Read)
/// yields) into the cell. Only `seed`/`update` depend on `V`; the rest of the op
/// lives on [`FoldAcc`]. The split mirrors the GROUP BY consume phases — `seed` a
/// new group from a row's value, `update` folds the next value in (a string
/// extreme persists only a winner here).
pub trait Fold<V>: FoldAcc {
    /// Materialise a new group's cell from a row's value.
    fn seed(v: V, wc: &mut Self::WorkerContext) -> Self::Acc;
    /// Fold a value into an existing cell (a string extreme compares the raw
    /// `&str` and persists only when it wins).
    fn update(
        acc: Self::Acc,
        v: V,
        wc: &mut Self::WorkerContext,
        ctx: &Self::SharedContext,
    ) -> Self::Acc;
}
