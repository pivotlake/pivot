//! Compiled (monomorphised) mixed-slot GROUP BY aggregations.
//!
//! [`MixedRowValueExtractor`](super::MixedRowValueExtractor) handles slot
//! signatures with `MIN`/`MAX` by matching a per-slot enum reader on every row.
//! This module specialises a *fixed* mixed signature into straight-line code,
//! the extremes counterpart of [`Compiled`](super::Compiled) for additive
//! slots. The additive [`Aggregate`](super::Aggregate) ops can't express an
//! extreme: their accumulation is uniform integer addition, while an extreme
//! combines by order comparison and a *string* extreme accumulates an
//! [`ArenaKey`] whose improvement persists bytes into the worker arena. So the
//! ops here are richer:
//!
//! - each op carries its own accumulator type (`i64` for counts / sums /
//!   numeric extremes, [`ArenaKey`] for string extremes), so a tuple of ops
//!   yields a heterogeneous accumulator row with no `SlotAcc`-style enum;
//! - each op implements the arena-aware `init` / `fold` / `combine` hooks the
//!   [`ValueExtractor`] contract exposes, plus its own output-column builder.
//!
//! Adding a new compiled mixed shape is one tuple type (e.g.
//! `CompiledMixed<(MinStrOp, CountOp)>`), selected in the planner; the enum
//! extractor stays as the fallback for any signature we haven't compiled.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Int64Type};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};

use crate::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::keys::ArenaKey;
use crate::operations::unary::group::values::{AggregationSlot, ValueColumns, ValueExtractor};

/// One aggregate slot of a compiled mixed signature.
///
/// Unlike the additive [`Aggregate`](super::Aggregate) op (a per-row `i64`
/// contribution folded by uniform addition), a mixed op owns its whole slot:
/// its accumulator type, how a row initialises / improves it (with arena
/// access, so a string extreme can persist candidates), how two workers'
/// accumulators combine in the partition merge, and how the finished
/// accumulators become an output column.
pub trait MixedOp: Send + 'static {
    /// The per-group accumulator. `Default` is never observed as a value —
    /// hash-table entries are built via `init` — it only zero-initialises
    /// storage.
    type Acc: Copy + Default + Send;
    /// Per-batch reader — the downcast input column, or `()` for a count.
    type Reader<'b>;
    /// Builder for this slot's output column.
    type Builder;

    fn make_reader(batch: &RecordBatch, column: usize) -> Self::Reader<'_>;
    /// A fresh entry's accumulator from row `idx`.
    fn init(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Acc;
    /// Fold row `idx` into an existing accumulator.
    fn fold(
        acc: Self::Acc,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> Self::Acc;
    /// Combine two already-built accumulators during the partition merge.
    fn combine(a: Self::Acc, b: Self::Acc, arena: &SharedArena) -> Self::Acc;
    /// The accumulator as an `ORDER BY <slot> LIMIT k` sort key. Only integer
    /// slots are sortable; a string extreme panics (the planner never sorts on
    /// one).
    fn sort_key(acc: Self::Acc) -> i64;

    fn make_builder(allocator: &mut SlabAllocator, rows: usize) -> Self::Builder;
    fn push(builder: &mut Self::Builder, acc: Self::Acc);
    /// Materialise the column for output slot `slot` (its field is named
    /// `v{slot}`, matching the enum extractor's layout).
    fn finish(builder: Self::Builder, slot: usize, arena: &Arc<SharedArena>) -> (Field, ArrayRef);
}

/// Finish an integer slot: an `Int64` column named `v{slot}`.
fn finish_int(builder: PrimitiveBuilder<Int64Type>, slot: usize) -> (Field, ArrayRef) {
    (
        Field::new(format!("v{slot}"), DataType::Int64, false),
        builder.into_array(None),
    )
}

/// Finish a string-extreme slot: re-point each persisted [`ArenaKey`] at its
/// arena bytes as a zero-copy `Utf8View` column (≤ 12-byte strings inline into
/// the view itself).
fn finish_str(
    raw_keys: SlabColumn<u128>,
    slot: usize,
    arena: &Arc<SharedArena>,
) -> (Field, ArrayRef) {
    let len = raw_keys.len();
    let raw = ScalarBuffer::<u128>::new(raw_keys.into_buffer(), 0, len);
    let mut views = Vec::with_capacity(len);
    for &r in raw.iter() {
        let key = ArenaKey::from_raw(r);
        if key.is_inline() {
            views.push(key.as_u128());
        } else {
            views.push(make_view(
                key.resolve(arena),
                key.buffer_index(),
                key.offset(),
            ));
        }
    }
    let buffers = arena.to_arrow_buffers();
    // Safety: views point at live arena bytes (or inline); the Arc'd arena
    // keeps the ring memory alive with the array.
    let arr: ArrayRef = Arc::new(unsafe {
        StringViewArray::new_unchecked(ScalarBuffer::from(views), buffers, None)
    });
    (
        Field::new(format!("v{slot}"), DataType::Utf8View, false),
        arr,
    )
}

/// `COUNT(*)` / `COUNT(non-null col)`: +1 per row, reads no column.
pub struct CountOp;

impl MixedOp for CountOp {
    type Acc = i64;
    type Reader<'b> = ();
    type Builder = PrimitiveBuilder<Int64Type>;

    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _column: usize) {}
    #[inline(always)]
    fn init(_reader: &(), _idx: usize, _arena: &mut WorkerArena) -> i64 {
        1
    }
    #[inline(always)]
    fn fold(acc: i64, _reader: &(), _idx: usize, _arena: &mut WorkerArena) -> i64 {
        acc + 1
    }
    #[inline(always)]
    fn combine(a: i64, b: i64, _arena: &SharedArena) -> i64 {
        a + b
    }
    #[inline(always)]
    fn sort_key(acc: i64) -> i64 {
        acc
    }
    fn make_builder(allocator: &mut SlabAllocator, rows: usize) -> Self::Builder {
        PrimitiveBuilder::with_capacity(allocator, rows)
    }
    #[inline(always)]
    fn push(builder: &mut Self::Builder, acc: i64) {
        builder.push(&acc, 1);
    }
    fn finish(builder: Self::Builder, slot: usize, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        finish_int(builder, slot)
    }
}

/// `SUM(col)` over an integer column, widened to `i64`.
pub struct SumOp<T>(PhantomData<T>);

impl<T: ArrowPrimitiveType + Send> MixedOp for SumOp<T>
where
    T::Native: Into<i64>,
{
    type Acc = i64;
    type Reader<'b> = &'b PrimitiveArray<T>;
    type Builder = PrimitiveBuilder<Int64Type>;

    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn init(reader: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> i64 {
        unsafe { reader.value_unchecked(idx) }.into()
    }
    #[inline(always)]
    fn fold(acc: i64, reader: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> i64 {
        acc + unsafe { reader.value_unchecked(idx) }.into()
    }
    #[inline(always)]
    fn combine(a: i64, b: i64, _arena: &SharedArena) -> i64 {
        a + b
    }
    #[inline(always)]
    fn sort_key(acc: i64) -> i64 {
        acc
    }
    fn make_builder(allocator: &mut SlabAllocator, rows: usize) -> Self::Builder {
        PrimitiveBuilder::with_capacity(allocator, rows)
    }
    #[inline(always)]
    fn push(builder: &mut Self::Builder, acc: i64) {
        builder.push(&acc, 1);
    }
    fn finish(builder: Self::Builder, slot: usize, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        finish_int(builder, slot)
    }
}

/// `MIN`/`MAX` over an integer column (widened to `i64`); `MIN` iff `MIN`.
pub struct ExtremeIntOp<T, const MIN: bool>(PhantomData<T>);

/// `MIN(col)` over an integer column.
pub type MinIntOp<T> = ExtremeIntOp<T, true>;
/// `MAX(col)` over an integer column.
pub type MaxIntOp<T> = ExtremeIntOp<T, false>;

impl<T: ArrowPrimitiveType + Send, const MIN: bool> MixedOp for ExtremeIntOp<T, MIN>
where
    T::Native: Into<i64>,
{
    type Acc = i64;
    type Reader<'b> = &'b PrimitiveArray<T>;
    type Builder = PrimitiveBuilder<Int64Type>;

    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &PrimitiveArray<T> {
        batch.column(column).as_primitive::<T>()
    }
    #[inline(always)]
    fn init(reader: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> i64 {
        unsafe { reader.value_unchecked(idx) }.into()
    }
    #[inline(always)]
    fn fold(acc: i64, reader: &&PrimitiveArray<T>, idx: usize, _arena: &mut WorkerArena) -> i64 {
        let v = unsafe { reader.value_unchecked(idx) }.into();
        if (v < acc) == MIN { v } else { acc }
    }
    #[inline(always)]
    fn combine(a: i64, b: i64, _arena: &SharedArena) -> i64 {
        if MIN { a.min(b) } else { a.max(b) }
    }
    #[inline(always)]
    fn sort_key(acc: i64) -> i64 {
        acc
    }
    fn make_builder(allocator: &mut SlabAllocator, rows: usize) -> Self::Builder {
        PrimitiveBuilder::with_capacity(allocator, rows)
    }
    #[inline(always)]
    fn push(builder: &mut Self::Builder, acc: i64) {
        builder.push(&acc, 1);
    }
    fn finish(builder: Self::Builder, slot: usize, _arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        finish_int(builder, slot)
    }
}

/// `MIN`/`MAX` over a string column: the accumulator is the [`ArenaKey`] of the
/// best candidate persisted so far. `init` persists the first candidate, `fold`
/// persists only improvements, `combine` resolves two persisted extremes through
/// the shared arena (comparisons only, no writes).
pub struct ExtremeStrOp<const MIN: bool>;

/// `MIN(col)` over a string column.
pub type MinStrOp = ExtremeStrOp<true>;
/// `MAX(col)` over a string column.
pub type MaxStrOp = ExtremeStrOp<false>;

impl<const MIN: bool> MixedOp for ExtremeStrOp<MIN> {
    type Acc = ArenaKey;
    type Reader<'b> = &'b StringViewArray;
    type Builder = SlabColumn<u128>;

    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &StringViewArray {
        batch.column(column).as_string_view()
    }
    #[inline(always)]
    fn init(reader: &&StringViewArray, idx: usize, arena: &mut WorkerArena) -> ArenaKey {
        let s = unsafe { reader.value_unchecked(idx) };
        arena.push_bytes(s.as_bytes())
    }
    #[inline(always)]
    fn fold(
        acc: ArenaKey,
        reader: &&StringViewArray,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> ArenaKey {
        let candidate = unsafe { reader.value_unchecked(idx) }.as_bytes();
        let better = match candidate.cmp(acc.resolve(arena.shared())) {
            Ordering::Less => MIN,
            Ordering::Greater => !MIN,
            Ordering::Equal => false,
        };
        if better {
            arena.push_bytes(candidate)
        } else {
            acc
        }
    }
    #[inline(always)]
    fn combine(a: ArenaKey, b: ArenaKey, arena: &SharedArena) -> ArenaKey {
        let keep_b = match b.resolve(arena).cmp(a.resolve(arena)) {
            Ordering::Less => MIN,
            Ordering::Greater => !MIN,
            Ordering::Equal => false,
        };
        if keep_b { b } else { a }
    }
    fn sort_key(_acc: ArenaKey) -> i64 {
        unreachable!("a string extreme is never a top-k sort slot")
    }
    fn make_builder(allocator: &mut SlabAllocator, rows: usize) -> Self::Builder {
        SlabColumn::with_capacity(allocator, rows)
    }
    #[inline(always)]
    fn push(builder: &mut Self::Builder, acc: ArenaKey) {
        builder.push(acc.as_u128());
    }
    fn finish(builder: Self::Builder, slot: usize, arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        finish_str(builder, slot, arena)
    }
}

/// A tuple of [`MixedOp`]s — one per output slot — lifted to row-at-a-time
/// operations over the tuple of their accumulators. Implemented per arity by
/// `impl_mixed_ops!`; [`CompiledMixed`] is generic over it.
pub trait MixedOps: Send + 'static {
    /// The heterogeneous accumulator row (one `Acc` per op).
    type Accs: Copy + Default + Send;
    type Readers<'b>;
    type Builders;

    fn make_readers<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Readers<'b>;
    fn init(readers: &Self::Readers<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Accs;
    fn fold(
        accs: Self::Accs,
        readers: &Self::Readers<'_>,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> Self::Accs;
    fn combine(a: Self::Accs, b: Self::Accs, arena: &SharedArena) -> Self::Accs;
    fn sort_key(accs: &Self::Accs, slot: usize) -> i64;
    fn make_builders(allocator: &mut SlabAllocator, rows: usize) -> Self::Builders;
    fn push(builders: &mut Self::Builders, accs: &Self::Accs);
    fn finish(builders: Self::Builders, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>);
}

/// Implements [`MixedOps`] for an op tuple of a given arity. `$idx` are the
/// tuple field indices, which also index `value_slots` and name the output
/// columns.
macro_rules! impl_mixed_ops {
    ($n:literal; $($Op:ident $idx:tt),+) => {
        impl<$($Op: MixedOp),+> MixedOps for ($($Op,)+) {
            type Accs = ($($Op::Acc,)+);
            type Readers<'b> = ($($Op::Reader<'b>,)+);
            type Builders = ($($Op::Builder,)+);

            #[inline(always)]
            fn make_readers<'b>(
                batch: &'b RecordBatch,
                slots: &[AggregationSlot],
            ) -> Self::Readers<'b> {
                assert_eq!(slots.len(), $n, "slot count must match compiled arity");
                ($( $Op::make_reader(batch, slots[$idx].column), )+)
            }

            #[inline(always)]
            fn init(
                readers: &Self::Readers<'_>,
                idx: usize,
                arena: &mut WorkerArena,
            ) -> Self::Accs {
                ($( $Op::init(&readers.$idx, idx, arena), )+)
            }

            #[inline(always)]
            fn fold(
                accs: Self::Accs,
                readers: &Self::Readers<'_>,
                idx: usize,
                arena: &mut WorkerArena,
            ) -> Self::Accs {
                ($( $Op::fold(accs.$idx, &readers.$idx, idx, arena), )+)
            }

            #[inline(always)]
            fn combine(a: Self::Accs, b: Self::Accs, arena: &SharedArena) -> Self::Accs {
                ($( $Op::combine(a.$idx, b.$idx, arena), )+)
            }

            #[inline(always)]
            fn sort_key(accs: &Self::Accs, slot: usize) -> i64 {
                match slot {
                    $( $idx => $Op::sort_key(accs.$idx), )+
                    _ => unreachable!("sort slot out of range"),
                }
            }

            fn make_builders(allocator: &mut SlabAllocator, rows: usize) -> Self::Builders {
                ($( $Op::make_builder(allocator, rows), )+)
            }

            #[inline(always)]
            fn push(builders: &mut Self::Builders, accs: &Self::Accs) {
                $( $Op::push(&mut builders.$idx, accs.$idx); )+
            }

            fn finish(
                builders: Self::Builders,
                arena: &Arc<SharedArena>,
            ) -> (Vec<Field>, Vec<ArrayRef>) {
                let mut fields = Vec::with_capacity($n);
                let mut columns = Vec::with_capacity($n);
                $(
                    let (f, a) = $Op::finish(builders.$idx, $idx, arena);
                    fields.push(f);
                    columns.push(a);
                )+
                (fields, columns)
            }
        }
    };
}

impl_mixed_ops!(1; A 0);
impl_mixed_ops!(2; A 0, B 1);
impl_mixed_ops!(3; A 0, B 1, C 2);
impl_mixed_ops!(4; A 0, B 1, C 2, D 3);
impl_mixed_ops!(5; A 0, B 1, C 2, D 3, E 4);
impl_mixed_ops!(6; A 0, B 1, C 2, D 3, E 4, F 5);

/// A group's value for a compiled mixed shape: the ops' accumulator tuple.
pub struct MixedAccs<T>(pub T);

impl<T: Copy> Copy for MixedAccs<T> {}
impl<T: Copy> Clone for MixedAccs<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Default> Default for MixedAccs<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T: Copy + Default> Value for MixedAccs<T> {
    fn merge(self, _v: Self) -> Self {
        // Compiled mixed rows combine via the ValueExtractor hooks
        // (fold / combine), which carry the arena a string extreme needs;
        // nothing routes them through the context-free merge.
        unreachable!("MixedAccs combines via ValueExtractor::{{fold,combine}}")
    }
}

/// A [`ValueExtractor`] monomorphised over a tuple of [`MixedOp`]s — one per
/// output slot. Per-row init/fold and the partition-merge combine are
/// straight-line typed code with no per-row slot dispatch.
pub struct CompiledMixed<Ops>(PhantomData<Ops>);

impl<Ops: MixedOps> ValueExtractor for CompiledMixed<Ops> {
    // Same reasoning as the enum mixed extractor: a string extreme persists per
    // value built; scattering builds one per row, which would copy every
    // candidate into the arena. Stay in-place so folding persists only
    // improvements.
    const SUPPORTS_RADIX: bool = false;

    type Value = MixedAccs<Ops::Accs>;
    type Reader<'b> = Ops::Readers<'b>;
    type Columns = CompiledMixedColumns<Ops>;
    type SortKey = i64;

    #[inline(always)]
    fn make_reader<'b>(
        batch: &'b RecordBatch,
        value_slots: &[AggregationSlot],
    ) -> Self::Reader<'b> {
        Ops::make_readers(batch, value_slots)
    }

    fn value(_reader: &Self::Reader<'_>, _idx: usize) -> Self::Value {
        unreachable!("CompiledMixed builds values via init (it needs the arena)")
    }

    #[inline(always)]
    fn init(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Value {
        MixedAccs(Ops::init(reader, idx, arena))
    }

    #[inline(always)]
    fn fold(
        current: Self::Value,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> Self::Value {
        MixedAccs(Ops::fold(current.0, reader, idx, arena))
    }

    #[inline(always)]
    fn combine(
        current: Self::Value,
        incoming: Self::Value,
        arena: &SharedArena,
        _slots: &[AggregationSlot],
    ) -> Self::Value {
        MixedAccs(Ops::combine(current.0, incoming.0, arena))
    }

    #[inline(always)]
    fn sort_key(value: &Self::Value, slot: usize) -> i64 {
        Ops::sort_key(&value.0, slot)
    }
}

/// Emits one column per op: `Int64` for integer slots, zero-copy `Utf8View`
/// into the arena for string-extreme slots — the same layout as the enum
/// extractor's columns. The slot kinds are fixed by the op tuple, so the
/// runtime `value_slots` are ignored.
pub struct CompiledMixedColumns<Ops: MixedOps> {
    builders: Ops::Builders,
}

impl<Ops: MixedOps> ValueColumns for CompiledMixedColumns<Ops> {
    type Value = MixedAccs<Ops::Accs>;

    fn with_capacity(
        allocator: &mut SlabAllocator,
        rows: usize,
        _value_slots: &[AggregationSlot],
    ) -> Self {
        Self {
            builders: Ops::make_builders(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Self::Value) {
        Ops::push(&mut self.builders, &value.0);
    }

    fn finish(self, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        Ops::finish(self.builders, arena)
    }
}
