//! The **[`Aggregation`]** trait — one per aggregate op, fully typed to its own
//! input array and accumulator. There is no `Read`/`Fold` split and no shared
//! reader: a [`StrMin`] is typed to a `StringViewArray` and an integer `Sum` to a
//! `PrimitiveArray<T>`, so an op only ever sees the one kind of input it works on
//! — no `unreachable!`, no bit-punning, and a string extreme stays lazy (its
//! [`update`](Aggregation::update) holds the real `&str` and persists only a
//! winner).
//!
//! The ops:
//! - [`Count`] — `COUNT`, no input.
//! - [`Sum<T>`](Sum) / [`WideSum<T>`](WideSum) — `SUM`, narrow / wide accumulator.
//! - [`Min<T>`](Min) / [`Max<T>`](Max) — integer extremes.
//! - [`StrMin`] / [`StrMax`] — string extremes (an `ArenaKey` cell).
//!
//! A fixed signature is a tuple of these ([`Compiled`](super::container::Compiled));
//! a runtime numeric signature folds each slot by its kind
//! ([`Dynamic`](super::container::Dynamic)).

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
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// One aggregate op: how it reads its input, folds, merges and renders. The
/// seed/update/merge split mirrors the GROUP BY consume phases — `seed` a new
/// group from a row, `update` folds the next row in (a string extreme persists
/// only a winner here), `merge` combines two finished partials.
pub trait Aggregation: Send + Sync + 'static {
    /// This op's accumulator cell (`i64` / `i128` / `ArenaKey`).
    type Acc: Cell;
    /// This op's downcast input array for a batch — `&PrimitiveArray<T>`,
    /// `&StringViewArray`, or `()` for [`Count`]. This *is* the arrow array
    /// [`update`](Self::update) reads from.
    type Input<'b>;
    /// Runtime data [`merge`](Self::merge) needs that the type can't carry: the
    /// value arena for a string extreme; `()` otherwise. Built once at `Group`
    /// creation.
    type Cfg: Clone + Send + Sync + 'static;

    /// Downcast this op's input column, once per batch.
    fn bind(batch: &RecordBatch, column: usize) -> Self::Input<'_>;
    /// Build the merge config from the value arena.
    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;

    /// Materialise a new group's cell from row `idx`.
    fn seed(input: &Self::Input<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Acc;
    /// Fold row `idx` into an existing cell (a string extreme compares the raw
    /// `&str` and persists only when it wins).
    fn update(
        acc: Self::Acc,
        input: &Self::Input<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::Cfg,
    ) -> Self::Acc;
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
