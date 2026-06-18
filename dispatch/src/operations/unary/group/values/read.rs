//! The **[`Read`]** trait — how a column type yields one row's value to a
//! [`Fold`](super::fold::Fold). It owns the *width* (the only thing that
//! varies with the column type); the fold owns the *op*. Splitting them is what
//! keeps the signature space **additive** (`reads + folds`) instead of
//! multiplicative (`ops × widths`): `Sum`/`Min`/`Max` are one fold each, shared
//! across every integer width, and a new width is one `Read` impl — not one new
//! op per existing op.
//!
//! A `Read`'s [`Val`](Read::Val) is what the fold consumes: `i64` for an integer
//! column (widened once here), `&str` for a string column (borrowed, *not*
//! persisted — the string extreme persists only a winner, in its fold), and `()`
//! for [`Count`](super::fold::Count), which reads no column.

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch, StringViewArray};
use std::marker::PhantomData;

/// How one column type is read for a batch. One impl per *input shape* (integer
/// width / string / none), reused by every [`Fold`](super::fold::Fold)
/// over that shape.
pub trait Read: Send + Sync + 'static {
    /// The downcast input column for a batch (`&PrimitiveArray<T>`,
    /// `&StringViewArray`, or `()`), bound once per batch by [`bind`](Self::bind).
    type Input<'b>;
    /// One row's value, handed to a fold's
    /// [`update`](super::fold::Fold::update). `i64` / `&str` / `()`.
    type Val<'b>;

    /// Downcast this column once per batch.
    fn bind(batch: &RecordBatch, column: usize) -> Self::Input<'_>;
    /// Read row `idx`. An integer widens to `i64` here; a string is *borrowed*
    /// (no arena write — laziness lives in the fold).
    fn read<'b>(input: &Self::Input<'b>, idx: usize) -> Self::Val<'b>;
}

/// Reads an integer column of width `T`, widened to `i64`. The single place a
/// column width appears — every numeric fold (`Sum`/`Min`/`Max`) consumes the
/// `i64` this yields, so none of them is monomorphised per width.
pub struct IntRead<T>(PhantomData<T>);

impl<T: ArrowPrimitiveType + Send + Sync> Read for IntRead<T>
where
    T::Native: Into<i64>,
{
    type Input<'b> = &'b PrimitiveArray<T>;
    type Val<'b> = i64;

    #[inline(always)]
    fn bind(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn read<'b>(input: &Self::Input<'b>, idx: usize) -> Self::Val<'b> {
        unsafe { input.value_unchecked(idx) }.into()
    }
}

/// Reads a string (`Utf8View`) column, *borrowing* the `&str` — the string
/// extreme's fold compares it and persists only when it wins, so nothing is
/// written to the value arena on a read.
pub struct StrRead;

impl Read for StrRead {
    type Input<'b> = &'b StringViewArray;
    type Val<'b> = &'b str;

    #[inline(always)]
    fn bind(batch: &RecordBatch, column: usize) -> &StringViewArray {
        batch.column(column).as_string_view()
    }
    #[inline(always)]
    fn read<'b>(input: &Self::Input<'b>, idx: usize) -> Self::Val<'b> {
        unsafe { input.value_unchecked(idx) }
    }
}

/// Reads nothing — for [`Count`](super::fold::Count), whose fold ignores
/// the input.
pub struct NoRead;

impl Read for NoRead {
    type Input<'b> = ();
    type Val<'b> = ();

    #[inline(always)]
    fn bind(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn read<'b>(_input: &Self::Input<'b>, _idx: usize) -> Self::Val<'b> {}
}
