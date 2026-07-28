//! The **[`Fold`]** trait — one per *numeric* aggregate op, over the value a
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
//!
//! `Fold` is numeric-only: every op folds an owned, `'static` value (`()` / `i64`)
//! with no per-worker or shared state, so it carries no context. String extremes
//! ([`StrMin`]/[`StrMax`]) need a value arena and a borrowed `&str`, so they are
//! *not* folds — they expose inherent methods used only by the runtime
//! [`Dynamic`](super::container::Dynamic) container's string arms.
//!
//! A fixed numeric signature is a tuple of (read, fold) pairs
//! ([`Compiled`](super::container::Compiled)); a runtime signature folds each slot
//! by its kind ([`Dynamic`](super::container::Dynamic)).

mod count;
mod extreme;
mod f64;
mod string;
mod sum;
mod u128;

pub use count::Count;
pub use extreme::{Max, Min};
pub use f64::{F64Max, F64Min, F64Sum};
pub use string::{StrMax, StrMin};
pub use sum::{Sum, WideSum};
pub use u128::{U128Max, U128Min, U128Sum};

use super::cell::Cell;
use crate::arrays::SlabColumn;
use arrow_array::ArrayRef;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;

/// One numeric aggregate op, folding the [`Val`](Self::Val) a
/// [`Read`](super::read::Read) yields into a cell of type [`Acc`](Self::Acc). The
/// methods mirror the GROUP BY phases — [`seed`](Self::seed) a new group from a
/// row's value, [`update`](Self::update) folds the next row in,
/// [`merge`](Self::merge) combines two finished partials, [`finish`](Self::finish)
/// renders the column. No context: a numeric op persists nothing and resolves
/// through nothing.
///
/// Both `Val` and `Acc` are lifetime-free — a numeric op folds an owned `()`/`i64`
/// — so a container names them with no `for<'b>` projection (the borrowed `&str`
/// that would force one belongs to a string extreme, which is not a `Fold`).
pub trait Fold: Send + Sync + 'static {
    /// The value this op folds: `()` for [`Count`] (reads no column), `i64` for an
    /// integer `Sum`/`Min`/`Max`. A slot pairs this op with a
    /// [`Read`](super::read::Read) whose `Val<'b>` is exactly this type.
    type Val;
    /// This op's accumulator cell (`i64` / `i128`).
    type Acc: Cell;

    /// Whether this op's output can never be SQL NULL: `true` only for
    /// [`Count`], which renders `0` (not NULL) for a group with no non-NULL
    /// rows. The containers keep such a slot's seen bit always set, so its
    /// output column skips the null-buffer pass entirely.
    const ALWAYS_SEEN: bool = false;

    /// The cell of a group that has seen no non-NULL value yet: the fold's
    /// identity, absorbed by any [`update`](Self::update)/[`merge`](Self::merge)
    /// (`0` for a sum/count, the width's extreme for `MIN`/`MAX`). Keeps the
    /// fold loops branch-free: a NULL row simply contributes the identity, and
    /// whether the group's output is NULL is tracked by the container's seen
    /// bits, not the cell.
    fn empty() -> Self::Acc;
    /// Materialise a new group's cell from a row's value.
    fn seed(v: Self::Val) -> Self::Acc;
    /// Fold a value into an existing cell.
    fn update(acc: Self::Acc, v: Self::Val) -> Self::Acc;
    /// Combine two finished partials — the partition merge and radix fold.
    fn merge(a: Self::Acc, b: Self::Acc) -> Self::Acc;
    /// Render a finished column of cells into the Arrow array + field. `nulls`
    /// marks the groups that saw no non-NULL value (never set for a count).
    fn finish(
        name: &str,
        col: SlabColumn<Self::Acc>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef);
}
