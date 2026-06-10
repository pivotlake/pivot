//! The mixed-slot value extractor — grouped aggregates whose slots aren't all
//! additive integers (`MIN`/`MAX`, including over strings) alongside the usual
//! `COUNT`/`SUM` slots.
//!
//! The additive extractors ([`AggregationRowValueExtractor`], [`Compiled`])
//! keep every slot as a running integer merged by `+=`. A `MIN`/`MAX` slot
//! breaks that uniformity twice over: its combine is an order comparison, and
//! a *string* extreme isn't an integer at all — it's an [`ArenaKey`] pointing
//! at the best candidate seen so far, and improving it persists the new
//! candidate into the worker's arena. This extractor therefore:
//!
//! - stores each slot as a [`SlotAcc`] (an `i64` or an `ArenaKey`),
//! - dispatches each slot's per-row behaviour at runtime from a small enum
//!   reader (like the additive fallback extractor),
//! - implements the context-aware [`ValueExtractor`] hooks: `init` persists a
//!   string slot's first candidate, `fold` persists only when the candidate
//!   improves on the entry's current extreme, and `combine` resolves two
//!   persisted extremes through the shared arena (comparisons only, no
//!   writes),
//! - opts out of radix scatter ([`ValueExtractor::SUPPORTS_RADIX`] = false):
//!   scattering builds a value per *row*, which would write every string
//!   candidate to the arena instead of only the improvements.
//!
//! [`AggregationRowValueExtractor`]: super::AggregationRowValueExtractor
//! [`Compiled`]: super::Compiled

use crate::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::keys::ArenaKey;
use crate::operations::unary::group::values::{
    AggregationKind, AggregationSlot, ValueColumns, ValueExtractor,
};
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt16Type, UInt32Type};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::cmp::Ordering;
use std::sync::Arc;

/// One slot's accumulator: a running integer (count/sum/numeric extreme) or
/// the persisted bytes of the best string seen (string extreme). Which arm a
/// slot uses is fixed by its [`AggregationKind`]; the enum only exists because
/// the row stores heterogeneous slots side by side.
#[derive(Copy, Clone)]
pub enum SlotAcc {
    Int(i64),
    Str(ArenaKey),
}

impl Default for SlotAcc {
    fn default() -> Self {
        SlotAcc::Int(0)
    }
}

impl SlotAcc {
    #[inline(always)]
    fn int(self) -> i64 {
        match self {
            SlotAcc::Int(v) => v,
            SlotAcc::Str(_) => unreachable!("integer slot holds a string accumulator"),
        }
    }

    #[inline(always)]
    fn str_key(self) -> ArenaKey {
        match self {
            SlotAcc::Str(k) => k,
            SlotAcc::Int(_) => unreachable!("string slot holds an integer accumulator"),
        }
    }
}

/// A group's value: `N` mixed accumulators.
pub struct MixedRow<const N: usize>(pub [SlotAcc; N]);

impl<const N: usize> Copy for MixedRow<N> {}
impl<const N: usize> Clone for MixedRow<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize> Default for MixedRow<N> {
    fn default() -> Self {
        Self([SlotAcc::default(); N])
    }
}

impl<const N: usize> Value for MixedRow<N> {
    fn merge(self, _v: Self) -> Self {
        // Mixed rows combine via the ValueExtractor hooks (fold / combine),
        // which carry the slot kinds and the arena a string extreme needs;
        // nothing routes them through the context-free merge.
        unreachable!("MixedRow combines via ValueExtractor::{{fold,combine}}")
    }
}

/// An integer column widened to `i64` reads, over any physical width — MIN/MAX
/// inputs keep their storage type (e.g. a date as `UInt16` day counts).
enum IntReader<'b> {
    I8(&'b PrimitiveArray<Int8Type>),
    I16(&'b PrimitiveArray<Int16Type>),
    I32(&'b PrimitiveArray<Int32Type>),
    I64(&'b PrimitiveArray<Int64Type>),
    U16(&'b PrimitiveArray<UInt16Type>),
    U32(&'b PrimitiveArray<UInt32Type>),
}

impl IntReader<'_> {
    fn new<'b>(batch: &'b RecordBatch, column: usize) -> IntReader<'b> {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Int8 => IntReader::I8(col.as_primitive()),
            DataType::Int16 => IntReader::I16(col.as_primitive()),
            DataType::Int32 => IntReader::I32(col.as_primitive()),
            DataType::Int64 => IntReader::I64(col.as_primitive()),
            DataType::UInt16 => IntReader::U16(col.as_primitive()),
            DataType::UInt32 => IntReader::U32(col.as_primitive()),
            other => panic!("grouped MIN/MAX: unsupported column type {other:?}"),
        }
    }

    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        // Safety: idx < batch row count.
        unsafe {
            match self {
                IntReader::I8(a) => a.value_unchecked(idx) as i64,
                IntReader::I16(a) => a.value_unchecked(idx) as i64,
                IntReader::I32(a) => a.value_unchecked(idx) as i64,
                IntReader::I64(a) => a.value_unchecked(idx),
                IntReader::U16(a) => a.value_unchecked(idx) as i64,
                IntReader::U32(a) => a.value_unchecked(idx) as i64,
            }
        }
    }
}

/// Per-row input for one slot, resolved at runtime from the slot's kind and
/// column type. The extreme arms carry `min: true` / `false` for direction.
enum SlotReader<'b> {
    Count,
    Sum(IntReader<'b>),
    ExtremeInt(IntReader<'b>, bool),
    ExtremeStr(&'b StringViewArray, bool),
}

/// Per-batch reader: one [`SlotReader`] per output slot.
pub struct MixedRowReader<'b, const N: usize> {
    slots: [SlotReader<'b>; N],
}

/// A [`ValueExtractor`] over `N` mixed count/sum/min/max slots.
pub struct MixedRowValueExtractor<const N: usize>;

impl<const N: usize> MixedRowValueExtractor<N> {
    /// Slot `s`'s accumulator for a fresh entry at row `idx`.
    #[inline(always)]
    fn slot_init(slot: &SlotReader<'_>, idx: usize, arena: &mut WorkerArena) -> SlotAcc {
        match slot {
            SlotReader::Count => SlotAcc::Int(1),
            SlotReader::Sum(r) | SlotReader::ExtremeInt(r, _) => SlotAcc::Int(r.at(idx)),
            SlotReader::ExtremeStr(a, _) => {
                let s = unsafe { a.value_unchecked(idx) };
                SlotAcc::Str(arena.push_bytes(s.as_bytes()))
            }
        }
    }

    /// Fold row `idx` into slot `s`'s current accumulator.
    #[inline(always)]
    fn slot_fold(
        slot: &SlotReader<'_>,
        current: SlotAcc,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> SlotAcc {
        match slot {
            SlotReader::Count => SlotAcc::Int(current.int() + 1),
            SlotReader::Sum(r) => SlotAcc::Int(current.int() + r.at(idx)),
            SlotReader::ExtremeInt(r, min) => {
                let v = r.at(idx);
                let cur = current.int();
                SlotAcc::Int(if (v < cur) == *min { v } else { cur })
            }
            SlotReader::ExtremeStr(a, min) => {
                let candidate = unsafe { a.value_unchecked(idx) }.as_bytes();
                let cur = current.str_key();
                let better = match candidate.cmp(cur.resolve(arena.shared())) {
                    Ordering::Less => *min,
                    Ordering::Greater => !*min,
                    Ordering::Equal => false,
                };
                if better {
                    SlotAcc::Str(arena.push_bytes(candidate))
                } else {
                    current
                }
            }
        }
    }
}

impl<const N: usize> ValueExtractor for MixedRowValueExtractor<N> {
    // A string extreme persists per value built; scattering builds one per
    // row, which would copy every candidate into the arena. Stay in-place so
    // folding persists only improvements.
    const SUPPORTS_RADIX: bool = false;

    type Value = MixedRow<N>;
    type Reader<'b> = MixedRowReader<'b, N>;
    type Columns = MixedRowColumns<N>;
    type SortKey = i64;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        value_slots: &[AggregationSlot],
    ) -> MixedRowReader<'b, N> {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let slots = std::array::from_fn(|s| {
            let slot = value_slots[s];
            match slot.kind {
                AggregationKind::CountStar | AggregationKind::Count => SlotReader::Count,
                AggregationKind::Sum => SlotReader::Sum(IntReader::new(batch, slot.column)),
                AggregationKind::Min => {
                    SlotReader::ExtremeInt(IntReader::new(batch, slot.column), true)
                }
                AggregationKind::Max => {
                    SlotReader::ExtremeInt(IntReader::new(batch, slot.column), false)
                }
                AggregationKind::MinStr => {
                    SlotReader::ExtremeStr(batch.column(slot.column).as_string_view(), true)
                }
                AggregationKind::MaxStr => {
                    SlotReader::ExtremeStr(batch.column(slot.column).as_string_view(), false)
                }
            }
        });
        MixedRowReader { slots }
    }

    fn value(_reader: &MixedRowReader<'_, N>, _idx: usize) -> MixedRow<N> {
        unreachable!("MixedRowValueExtractor builds values via init (it needs the arena)")
    }

    #[inline(always)]
    fn init(reader: &MixedRowReader<'_, N>, idx: usize, arena: &mut WorkerArena) -> MixedRow<N> {
        MixedRow(std::array::from_fn(|s| {
            Self::slot_init(&reader.slots[s], idx, arena)
        }))
    }

    #[inline(always)]
    fn fold(
        current: MixedRow<N>,
        reader: &MixedRowReader<'_, N>,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> MixedRow<N> {
        MixedRow(std::array::from_fn(|s| {
            Self::slot_fold(&reader.slots[s], current.0[s], idx, arena)
        }))
    }

    fn combine(
        current: MixedRow<N>,
        incoming: MixedRow<N>,
        arena: &SharedArena,
        slots: &[AggregationSlot],
    ) -> MixedRow<N> {
        MixedRow(std::array::from_fn(|s| {
            let (a, b) = (current.0[s], incoming.0[s]);
            match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum => {
                    SlotAcc::Int(a.int() + b.int())
                }
                AggregationKind::Min => SlotAcc::Int(a.int().min(b.int())),
                AggregationKind::Max => SlotAcc::Int(a.int().max(b.int())),
                AggregationKind::MinStr | AggregationKind::MaxStr => {
                    let min = slots[s].kind == AggregationKind::MinStr;
                    let (ka, kb) = (a.str_key(), b.str_key());
                    let keep_b = match kb.resolve(arena).cmp(ka.resolve(arena)) {
                        Ordering::Less => min,
                        Ordering::Greater => !min,
                        Ordering::Equal => false,
                    };
                    SlotAcc::Str(if keep_b { kb } else { ka })
                }
            }
        }))
    }

    #[inline(always)]
    fn sort_key(value: &MixedRow<N>, slot: usize) -> i64 {
        value.0[slot].int()
    }
}

/// One output column's builder: integers on a slab, or raw [`ArenaKey`] bit
/// patterns whose decoding into views happens at finish (resolving needs the
/// arena).
enum MixedColBuilder {
    Int(PrimitiveBuilder<Int64Type>),
    Str(SlabColumn<u128>),
}

/// Emits one column per slot: `Int64` for integer slots, zero-copy `Utf8View`
/// into the arena for string-extreme slots.
pub struct MixedRowColumns<const N: usize> {
    cols: [MixedColBuilder; N],
}

impl<const N: usize> ValueColumns for MixedRowColumns<N> {
    type Value = MixedRow<N>;

    fn with_capacity(
        allocator: &mut SlabAllocator,
        rows: usize,
        value_slots: &[AggregationSlot],
    ) -> Self {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let mut s = 0;
        Self {
            cols: std::array::from_fn(|_| {
                let col = match value_slots[s].kind {
                    AggregationKind::MinStr | AggregationKind::MaxStr => {
                        MixedColBuilder::Str(SlabColumn::with_capacity(allocator, rows))
                    }
                    _ => MixedColBuilder::Int(PrimitiveBuilder::with_capacity(allocator, rows)),
                };
                s += 1;
                col
            }),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &MixedRow<N>) {
        for (s, col) in self.cols.iter_mut().enumerate() {
            match col {
                MixedColBuilder::Int(b) => b.push(&value.0[s].int(), 1),
                MixedColBuilder::Str(views) => views.push(value.0[s].str_key().as_u128()),
            }
        }
    }

    fn finish(self, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, c) in self.cols.into_iter().enumerate() {
            match c {
                MixedColBuilder::Int(b) => {
                    fields.push(Field::new(format!("v{s}"), DataType::Int64, false));
                    columns.push(b.into_array(None));
                }
                MixedColBuilder::Str(raw_keys) => {
                    // Re-point each persisted extreme at its arena bytes
                    // (≤ 12-byte strings inline into the view itself).
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
                    // Safety: views point at live arena bytes (or inline); the
                    // Arc'd arena keeps the ring memory alive with the array.
                    let arr: ArrayRef = Arc::new(unsafe {
                        StringViewArray::new_unchecked(ScalarBuffer::from(views), buffers, None)
                    });
                    fields.push(Field::new(format!("v{s}"), DataType::Utf8View, false));
                    columns.push(arr);
                }
            }
        }
        (fields, columns)
    }
}
