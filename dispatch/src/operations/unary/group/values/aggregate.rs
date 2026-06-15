//! The aggregate ops — the single source of truth for what each aggregate
//! contributes per row.
//!
//! Each op is a zero-sized type implementing [`Aggregate`]. They're used two
//! ways:
//! - [`Compiled`](super::compiled::Compiled) monomorphises over a *tuple* of
//!   them for a fixed signature (straight-line, no per-row branch);
//! - [`DynamicValueExtractor`](super::DynamicValueExtractor) (the runtime
//!   fallback) wraps them in a small enum and dispatches per row.
//!
//! Either way the per-row logic lives here once; accumulating the contributions
//! across rows is uniform `i64` addition, handled by `AggregationRow`.

use std::marker::PhantomData;

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch};

/// A single aggregate's per-row contribution: [`Count`] yields `1`, [`Sum`]
/// yields the row's (widened) column value. Implementors are zero-sized — the
/// input column, if any, lives in the per-batch [`Reader`](Aggregate::Reader).
///
/// The op's *type* fixes which aggregate it is, so it's only ever told its input
/// `column` — never a kind. (Selecting which op to use from a query's
/// [`AggregationKind`](super::AggregationKind) happens earlier: at plan time for the
/// compiled path, in the runtime `match` for the fallback.)
///
/// This describes only the per-row *input*; accumulating it across rows is
/// uniform `i64` addition handled by `AggregationRow`, not here.
pub trait Aggregate {
    /// Per-batch reader — the downcast input column, or `()` for `COUNT(*)`.
    type Reader<'b>;
    fn make_reader(batch: &RecordBatch, column: usize) -> Self::Reader<'_>;
    /// This aggregate's contribution for row `idx`.
    fn contribution(reader: &Self::Reader<'_>, idx: usize) -> i64;
}

/// `COUNT(*)` / `COUNT(non-null col)`: contributes `1` per row, reads no column.
pub struct Count;

impl Aggregate for Count {
    type Reader<'b> = ();
    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn contribution(_reader: &(), _idx: usize) -> i64 {
        1
    }
}

/// `SUM(col)` over an integer column, widened to `i64`.
pub struct Sum<T>(PhantomData<T>);

impl<T: ArrowPrimitiveType> Aggregate for Sum<T>
where
    T::Native: Into<i64>,
{
    type Reader<'b> = &'b PrimitiveArray<T>;
    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn contribution(reader: &&PrimitiveArray<T>, idx: usize) -> i64 {
        unsafe { reader.value_unchecked(idx) }.into()
    }
}
