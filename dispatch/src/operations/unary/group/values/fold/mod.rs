//! Axis 2 — **FOLD**: how a slot's cells combine, and how the finished cell
//! renders to Arrow.
//!
//! A fold is the family an aggregate belongs to:
//! - [`Add`] — additive (`COUNT`/`SUM`), output `Int64`/`Decimal128`;
//! - [`Min`]/[`Max`] — integer extremes, same numeric output;
//! - [`StrMin`]/[`StrMax`] — string extremes over an `ArenaKey` cell, output
//!   `Utf8View`.
//!
//! [`seed`](Fold::seed) starts a new group from a row; [`update`](Fold::update)
//! folds another row in (defaulting to `combine(acc, seed)`, but the string folds
//! override it to compare the raw `&str` and persist only a winner — so a losing
//! row never touches the arena); [`combine`](Fold::combine) merges two finished
//! partials. [`Op<R, F>`](super::op::Op) pairs a [`Read`](super::read) with a fold;
//! [`Mono`](super::container::Mono) carries a fold directly with a per-slot
//! [`SlotReader`].

mod add;
mod extreme;
mod string;

pub use add::Add;
pub use extreme::{Max, Min};
pub use string::{StrMax, StrMin};

use super::cell::Cell;
use super::read::SlotReader;
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::sync::Arc;

/// How a family of slots folds cells of width `A` and renders its output column.
pub trait Fold<A: Cell> {
    /// Runtime data [`combine`](Self::combine)/[`update`](Self::update) need that
    /// the type can't carry: the value arena for the string folds (to resolve and
    /// compare `ArenaKey`s), `()` for the numeric ones. Built once per `Group`.
    type Cfg: Clone + Send + Sync + 'static;
    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;

    /// Start a new group's cell from row `idx` (the string folds persist here).
    fn seed(slot: &SlotReader<'_>, idx: usize, arena: &mut WorkerArena) -> A;

    /// Fold row `idx` into an existing cell. Defaults to merging the row's seed in;
    /// a string fold overrides this to compare the raw bytes first and persist
    /// only when the row wins.
    #[inline(always)]
    fn update(
        acc: A,
        slot: &SlotReader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::Cfg,
    ) -> A {
        Self::combine(acc, Self::seed(slot, idx, arena), cfg)
    }

    /// Merge two finished partials (the partition merge and radix fold).
    fn combine(acc: A, incoming: A, cfg: &Self::Cfg) -> A;

    /// This cell as an `ORDER BY <agg>` sort key (widened to `i128`). String cells
    /// return their raw view bits — a string extreme never feeds a top-k.
    fn sort_key(cell: A) -> i128;

    /// Render a finished column of cells into the Arrow array + field.
    fn finish(name: &str, col: SlabColumn<A>, arena: &Arc<SharedArena>) -> (Field, ArrayRef);
}
