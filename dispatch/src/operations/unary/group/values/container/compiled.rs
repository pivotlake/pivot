//! Compile-time aggregation container.
//!
//! [`Compiled`] stores a fixed tuple of numeric aggregation slots. Each slot
//! combines a typed [`Read`] implementation with a [`Fold`], so row processing
//! has no runtime slot dispatch. String aggregates and signatures not selected
//! for specialization use [`Dynamic`](super::Dynamic).
//!
//! [`Pair`] is a named marker instead of a nested `(Read, Fold)` tuple. Keeping
//! the type shallow avoids excessive recursive monomorphization when a value
//! passes through top-k.

use super::super::cell::Cell;
use super::super::fold::{Count, Fold, Max, Min, Sum};
use super::super::read::{IntRead, NoRead, Read};
use super::super::{AggregationSlot, AggregationValue, ArityBody, ValueColumnBuilder};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;

/// Type marker pairing a slot reader with its fold operation.
pub struct Pair<R, F>(PhantomData<(R, F)>);

/// Slot aliases keep signatures readable, for example
/// `Compiled<(SumSlot<Int32Type>, CountSlot)>` instead of the raw pairs.
pub type CountSlot<A = i64> = Pair<NoRead, Count<A>>;
/// `SUM(col: T)` accumulating in `A`. Use `i128` for a wide sum.
pub type SumSlot<T, A = i64> = Pair<IntRead<T>, Sum<A>>;
/// `MIN(col: T)` over an integer column, accumulating in `A`.
pub type MinSlot<T, A = i64> = Pair<IntRead<T>, Min<A>>;
/// `MAX(col: T)` over an integer column, accumulating in `A`.
pub type MaxSlot<T, A = i64> = Pair<IntRead<T>, Max<A>>;

/// Associates a tuple of slot markers with its accumulator and output-column
/// tuple types. Folding is generated directly by `impl_compiled!`.
pub trait OpTuple: Send + Sync + 'static {
    /// One accumulator cell per slot.
    type Accs: Cell;
    /// One output-column builder per slot.
    type ColumnBuilders;
}

/// Accumulator cells for a fixed aggregation signature.
pub struct Compiled<Ops: OpTuple> {
    accs: Ops::Accs,
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

/// Output-column builders for a [`Compiled`] signature.
pub struct CompiledColumnBuilder<Ops: OpTuple> {
    builders: Ops::ColumnBuilders,
}

/// Implements each supported tuple arity as straight-line reads and folds.
///
/// The equality constraint on `R::Val` and `F::Val` ensures each reader
/// produces exactly the value its fold accepts.
macro_rules! impl_compiled {
    ($($R:ident $F:ident $idx:tt),+) => {
        impl<$($R, $F),+> OpTuple for ($(Pair<$R, $F>,)+)
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>,)+
        {
            type Accs = ($($F::Acc,)+);
            type ColumnBuilders = ($(SlabColumn<$F::Acc>,)+);
        }

        // Compiled folds are numeric and need no arena context.
        impl<$($R, $F),+> AggregationValue for Compiled<($(Pair<$R, $F>,)+)>
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>, $F::Acc: Into<i128>,)+
        {
            type Owned = Self;
            type StorageMetadata = ();
            type Reader<'b> = ($($R::Input<'b>,)+);
            type SharedContext = ();
            type ColumnBuilder = CompiledColumnBuilder<($(Pair<$R, $F>,)+)>;
            type SortKey = i128;
            type WorkerContext = ();

            fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
                debug_assert_eq!(slots.len(), [$($idx),+].len(), "slot count must match the tuple arity");
                ($($R::bind(batch, slots[$idx].column),)+)
            }

            fn storage_metadata(_ctx: &()) {}

            fn metadata_for_arity<const N: usize>() {}

            #[inline(always)]
            fn dispatch_arity<Ret>(_metadata: (), body: impl ArityBody<Ret>) -> Ret {
                // The tuple arity is already in the type.
                body.run::<0>()
            }

            fn stored_size(_metadata: ()) -> usize {
                size_of::<Self>()
            }

            fn stored_align() -> usize {
                align_of::<Self>()
            }

            #[inline(always)]
            unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a Self {
                unsafe { &*(ptr as *const Self) }
            }

            #[inline(always)]
            unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut Self {
                unsafe { &mut *(ptr as *mut Self) }
            }

            #[inline(always)]
            fn seed(&mut self, reader: &Self::Reader<'_>, idx: usize, _wc: &mut ()) {
                self.accs = ($($F::seed($R::read(&reader.$idx, idx)),)+);
            }

            #[inline(always)]
            fn update(&mut self, reader: &Self::Reader<'_>, idx: usize, _wc: &mut (), _ctx: &()) {
                self.accs = ($($F::update(self.accs.$idx, $R::read(&reader.$idx, idx)),)+);
            }

            #[inline(always)]
            fn merge_from(&mut self, source: &Self, _ctx: &()) {
                self.accs = ($($F::merge(self.accs.$idx, source.accs.$idx),)+);
            }

            #[inline(always)]
            fn copy_from(&mut self, source: &Self) {
                *self = *source;
            }

            #[inline(always)]
            fn sort_key(&self, slot: usize) -> i128 {
                // All accumulator types convert to the common ORDER BY key type.
                match slot {
                    $($idx => self.accs.$idx.into(),)+
                    _ => unreachable!("sort_key slot {slot} out of range"),
                }
            }

            #[inline(always)]
            fn to_owned(&self, _ctx: &(), _wc: &mut Option<()>) -> Self {
                *self
            }
        }

        impl<$($R, $F),+> ValueColumnBuilder for CompiledColumnBuilder<($(Pair<$R, $F>,)+)>
        where
            $($R: Read, $F: Fold, for<'b> $R: Read<Val<'b> = $F::Val>, $F::Acc: Into<i128>,)+
        {
            type Value = Compiled<($(Pair<$R, $F>,)+)>;
            type Context = ();

            fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _context: &()) -> Self {
                Self { builders: ($(SlabColumn::<$F::Acc>::with_capacity(allocator, rows),)+) }
            }

            #[inline(always)]
            fn push(&mut self, value: &Self::Value) {
                $(self.builders.$idx.push(value.accs.$idx);)+
            }

            #[inline(always)]
            fn push_stored(&mut self, stored: &Self::Value) {
                self.push(stored);
            }

            fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
                let mut fields = Vec::new();
                let mut arrays = Vec::new();
                $(
                    let (f, a) = $F::finish(&format!("v{}", $idx), self.builders.$idx);
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
