//! The aggregate ops — the single source of truth for what each aggregate
//! reads per row and how it combines.
//!
//! Each op is a zero-sized type implementing [`Aggregate`]. They're used two
//! ways:
//! - [`Compiled`](super::compiled::Compiled) monomorphises over a *tuple* of
//!   them for a fixed signature (straight-line, no per-row branch);
//! - [`DynamicValueExtractor`](super::DynamicValueExtractor) (the runtime
//!   fallback) wraps them in a small enum and dispatches per row.
//!
//! An op contributes a per-row `i64` ([`contribution`](Aggregate::contribution))
//! and declares its [`KIND`](Aggregate::KIND); the actual combining (add for
//! `SUM`/`COUNT`, min/max for the extremes) lives once on
//! [`AggregationKind::combine`]. `Compiled` calls that with a `const` kind so it
//! folds to straight-line code; the fallback passes a runtime kind.

use std::marker::PhantomData;

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch};

use super::AggregationKind;

/// A single aggregate's per-row input. Implementors are zero-sized — the input
/// column, if any, lives in the per-batch [`Reader`](Aggregate::Reader).
///
/// The op's *type* fixes which aggregate it is, so it's only ever told its input
/// `column` — never a kind. (Selecting which op to use from a query's
/// [`AggregationKind`] happens earlier: at plan time for the compiled path, in
/// the runtime `match` for the fallback.) Combining contributions across rows is
/// [`KIND`](Self::KIND)`.`[`combine`](AggregationKind::combine), not here.
pub trait Aggregate {
    /// How this op combines — add (`SUM`/`COUNT`) or take an extreme (`MIN`/`MAX`).
    const KIND: AggregationKind;
    /// Per-batch reader — the downcast input column, or `()` for `COUNT(*)`.
    type Reader<'b>;
    fn make_reader(batch: &RecordBatch, column: usize) -> Self::Reader<'_>;
    /// This aggregate's contribution for row `idx` (widened to `i64`).
    fn contribution(reader: &Self::Reader<'_>, idx: usize) -> i64;
}

/// `COUNT(*)` / `COUNT(non-null col)`: contributes `1` per row, reads no column.
pub struct Count;

impl Aggregate for Count {
    const KIND: AggregationKind = AggregationKind::Count;
    type Reader<'b> = ();
    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn contribution(_reader: &(), _idx: usize) -> i64 {
        1
    }
}

/// The column-reading ops — currently just `SUM` over an integer column, widened
/// to `i64`. (`MIN`/`MAX` need no op of their own on the runtime-dispatched path:
/// they read the column identically to `SUM` and differ only in how they combine,
/// which is [`AggregationKind::combine`] keyed on the slot's kind. Dedicated
/// `Min`/`Max`/`MinStr`/`MaxStr` ops arrive with the compiled string-extreme
/// path.) The macro shape stays so adding such ops is one line each.
macro_rules! column_op {
    ($(#[$doc:meta])* $Op:ident => $kind:ident) => {
        $(#[$doc])*
        pub struct $Op<T>(PhantomData<T>);

        impl<T: ArrowPrimitiveType> Aggregate for $Op<T>
        where
            T::Native: Into<i64>,
        {
            const KIND: AggregationKind = AggregationKind::$kind;
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
    };
}

column_op!(
    /// `SUM(col)` over an integer column, widened to `i64`.
    Sum => Sum
);
