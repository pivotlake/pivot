//! Numeric aggregate operations.
//!
//! [`Read`](super::read::Read) handles Arrow's physical column width, while
//! [`Fold`] handles the aggregate operation. A fold therefore sees a normalized
//! value such as `i64`, independent of whether the source was `Int16`, `Int32`,
//! or `Int64`.
//!
//! `Fold` itself is context-free and numeric. Float, wide-input, and string
//! helpers live beside it but use inherent methods because their cells require
//! representation-specific conversion or arena access.

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

    /// Materialise a new group's cell from a row's value.
    fn seed(v: Self::Val) -> Self::Acc;
    /// Fold a value into an existing cell.
    fn update(acc: Self::Acc, v: Self::Val) -> Self::Acc;
    /// Combine two finished partials — the partition merge and radix fold.
    fn merge(a: Self::Acc, b: Self::Acc) -> Self::Acc;
    /// Render a finished column of cells into the Arrow array + field.
    fn finish(name: &str, col: SlabColumn<Self::Acc>) -> (Field, ArrayRef);
}
