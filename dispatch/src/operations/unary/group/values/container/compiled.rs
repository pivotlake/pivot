//! Fixed-arity, statically typed aggregation.
//!
//! Each slot is a [`Pair`] of a typed [`Read`] implementation and a [`Fold`].
//! The slot tuple determines the accumulator tuple, readers, and output builders
//! at compile time. The generated row operations are therefore unrolled and
//! contain no per-slot dispatch.
//!
//! This representation is numeric-only and uses `()` for its query context and
//! worker state. Signatures containing string extrema use
//! [`RuntimeAggregation`](super::RuntimeAggregation), which can carry an arena.
//!
//! [`Pair`] is a nominal type instead of a nested `(R, F)` tuple because keeping
//! the generated type shallow avoids a compiler monomorphization cycle when the
//! value flows through top-k output.

use super::super::cell::Cell;
use super::super::fold::{Count, Fold, Max, Min, Sum};
use super::super::read::{IntRead, NoRead, Read};
use super::super::{AggregationColumnBuilders, AggregationSlot, ByValueAggregation};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;

/// A type-level association between one input reader and one aggregate fold.
///
/// `Pair` is never instantiated; [`Compiled`] stores only accumulator cells.
pub struct Pair<R, F>(PhantomData<(R, F)>);

/// `COUNT(*)` with accumulator type `A`.
pub type CountSlot<A = i64> = Pair<NoRead, Count<A>>;
/// `SUM(col: T)` accumulating in `A` — `SumSlot<T, i128>` is the wide sum.
pub type SumSlot<T, A = i64> = Pair<IntRead<T>, Sum<A>>;
/// `MIN(col: T)` over an integer column, accumulating in `A`.
pub type MinSlot<T, A = i64> = Pair<IntRead<T>, Min<A>>;
/// `MAX(col: T)` over an integer column, accumulating in `A`.
pub type MaxSlot<T, A = i64> = Pair<IntRead<T>, Max<A>>;
// String extrema are intentionally absent: they need an arena and therefore use
// RuntimeAggregation.

/// Types supplied by a tuple of compiled slots.
///
/// The macro writes the operational code directly; this trait only names tuple
/// types that the generic container structs need in their fields.
pub trait OpTuple: Send + Sync + 'static {
    /// The per-slot accumulator cells.
    type Accs: Cell;
    /// The per-slot output-column builders: one [`SlabColumn`] per slot, in the
    /// same tuple shape as [`Accs`](Self::Accs). Held by [`CompiledColumns`], which
    /// a struct declared once over a generic `Ops` can only name through this
    /// associated type (the same reason [`Accs`](Self::Accs) exists).
    type Cols;
}

/// A fixed aggregate signature: the slots named by the op tuple `Ops`, each its
/// own cell. The running cells are the only state — their tuple type is `Ops::Accs`.
pub struct Compiled<Ops: OpTuple> {
    accs: Ops::Accs,
}

/// The output-column builders for a [`Compiled`] signature: one [`SlabColumn`] per
/// slot, in the tuple shape [`OpTuple::Cols`]. The value-side counterpart to a key
/// extractor's [`KeyColumns`](crate::operations::unary::group::keys::KeyColumns).
pub struct CompiledColumns<Ops: OpTuple> {
    cols: Ops::Cols,
}

impl<Ops: OpTuple> Copy for Compiled<Ops> {}
impl<Ops: OpTuple> Clone for Compiled<Ops> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Ops: OpTuple> Default for Compiled<Ops> {
    fn default() -> Self {
        Self {
            accs: Ops::Accs::default(),
        }
    }
}

/// Generate the tuple plumbing and unrolled aggregation implementation for one
/// supported arity.
macro_rules! impl_compiled {
    ($($R:ident $F:ident $idx:tt),+) => {
        impl<$($R, $F),+> OpTuple for ($(Pair<$R, $F>,)+)
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>,)+
        {
            type Accs = ($($F::Acc,)+);
            type Cols = ($(SlabColumn<$F::Acc>,)+);
        }

        // SAFETY: Every accumulator is a numeric Cell: it is Copy, needs no
        // destructor, and accepts an all-zero bit pattern.
        unsafe impl<$($R, $F),+> ByValueAggregation for Compiled<($(Pair<$R, $F>,)+)>
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>, $F::Acc: Into<i128>,)+
        {
            type Reader<'b> = ($($R::Input<'b>,)+);
            type Context = ();
            type Columns = CompiledColumns<($(Pair<$R, $F>,)+)>;
            type SortKey = i128;
            type WorkerState = ();

            fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
                debug_assert_eq!(slots.len(), [$($idx),+].len(), "slot count must match the tuple arity");
                ($($R::bind(batch, slots[$idx].column),)+)
            }

            #[inline(always)]
            fn value(reader: &Self::Reader<'_>, idx: usize, _wc: &mut ()) -> Self {
                Self { accs: ($($F::seed($R::read(&reader.$idx, idx)),)+) }
            }

            #[inline(always)]
            fn update_from_reader(
                self,
                reader: &Self::Reader<'_>,
                idx: usize,
                _wc: &mut (),
                _ctx: &(),
            ) -> Self {
                Self {
                    accs: ($($F::update(self.accs.$idx, $R::read(&reader.$idx, idx)),)+),
                }
            }

            #[inline(always)]
            fn merge(self, other: Self, _ctx: &()) -> Self {
                Self { accs: ($($F::merge(self.accs.$idx, other.accs.$idx),)+) }
            }

            #[inline(always)]
            fn sort_key(&self, slot: usize) -> i128 {
                // Widen the cell directly — a numeric extreme/sum/count to its
                // `ORDER BY` key.
                match slot {
                    $($idx => self.accs.$idx.into(),)+
                    _ => unreachable!("sort_key slot {slot} out of range"),
                }
            }
        }

        impl<$($R, $F),+> AggregationColumnBuilders for CompiledColumns<($(Pair<$R, $F>,)+)>
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>, $F::Acc: Into<i128>,)+
        {
            type Value = Compiled<($(Pair<$R, $F>,)+)>;
            type Context = ();

            fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _context: &()) -> Self {
                Self { cols: ($(SlabColumn::<$F::Acc>::with_capacity(allocator, rows),)+) }
            }

            #[inline(always)]
            fn push_owned(&mut self, value: &Self::Value) {
                $(self.cols.$idx.push(value.accs.$idx);)+
            }

            #[inline(always)]
            fn push_entry(&mut self, state: &Self::Value) {
                self.push_owned(state);
            }

            fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
                let mut fields = Vec::new();
                let mut arrays = Vec::new();
                $(
                    let (f, a) = $F::finish(&format!("v{}", $idx), self.cols.$idx);
                    fields.push(f);
                    arrays.push(a);
                )+
                (fields, arrays)
            }
        }
    };
}

impl_compiled!(R0 F0 0);
impl_compiled!(R0 F0 0, R1 F1 1);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3, R4 F4 4);
impl_compiled!(R0 F0 0, R1 F1 1, R2 F2 2, R3 F3 3, R4 F4 4, R5 F5 5);
