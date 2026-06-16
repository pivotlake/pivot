//! [`Compiled`] — a fixed signature monomorphised over a tuple of [`Aggregate`]
//! atoms, straight-line with no per-row dispatch.
//!
//! Each slot is a whole atom with its own cell type, so a `Compiled` shape stores
//! a heterogeneous tuple of cells and mixes families freely — including a string
//! extreme beside an integer one (`Compiled<(StrMin, Max<Int32Type>)>`), which is
//! exactly the `MIN(str), MAX(int)` case. The planner instantiates the tuple it
//! needs; everything else routes to [`Mono`](super::Mono)/[`Dynamic`](super::Dynamic).
//!
//! The per-arity [`OpTuple`] impls below carry all the tuple plumbing, so the
//! [`AggregationValue`] impl for [`Compiled`] is a single thin delegation.

use super::super::op::SlotOp;
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// A tuple of [`Aggregate`] atoms, with the per-slot plumbing the container needs.
/// Implemented for tuples of arity 1–6 by the macro below.
pub trait OpTuple: Send + Sync + 'static {
    /// The heterogeneous cell tuple — one cell per op, each its own width.
    type Accs: Copy + Default + Send + Sync + 'static;
    type Reader<'b>;
    type Cfg: Clone + Send + Sync + 'static;
    type Columns;

    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;
    fn seed(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Accs;
    fn update(
        accs: Self::Accs,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::Cfg,
    ) -> Self::Accs;
    fn merge(a: Self::Accs, b: Self::Accs, cfg: &Self::Cfg) -> Self::Accs;
    fn sort_key(accs: &Self::Accs, slot: usize) -> i128;
    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns;
    fn push(accs: &Self::Accs, cols: &mut Self::Columns);
    fn finish(cols: Self::Columns, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>);
}

macro_rules! impl_optuple {
    ($($O:ident $idx:tt),+) => {
        impl<$($O: SlotOp),+> OpTuple for ($($O,)+) {
            type Accs = ($($O::Acc,)+);
            type Reader<'b> = ($($O::Reader<'b>,)+);
            type Cfg = ($($O::Cfg,)+);
            type Columns = ($(SlabColumn<$O::Acc>,)+);

            #[inline(always)]
            fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg {
                ($($O::cfg(arena),)+)
            }
            #[inline(always)]
            fn make_reader<'b>(
                batch: &'b RecordBatch,
                slots: &[AggregationSlot],
            ) -> Self::Reader<'b> {
                ($($O::make_reader(batch, slots[$idx].column),)+)
            }
            #[inline(always)]
            fn seed(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Accs {
                ($($O::seed(&reader.$idx, idx, arena),)+)
            }
            #[inline(always)]
            fn update(
                accs: Self::Accs,
                reader: &Self::Reader<'_>,
                idx: usize,
                arena: &mut WorkerArena,
                cfg: &Self::Cfg,
            ) -> Self::Accs {
                ($($O::update(accs.$idx, &reader.$idx, idx, arena, &cfg.$idx),)+)
            }
            #[inline(always)]
            fn merge(a: Self::Accs, b: Self::Accs, cfg: &Self::Cfg) -> Self::Accs {
                ($($O::merge(a.$idx, b.$idx, &cfg.$idx),)+)
            }
            #[inline(always)]
            fn sort_key(accs: &Self::Accs, slot: usize) -> i128 {
                match slot {
                    $($idx => $O::sort_key(accs.$idx),)+
                    _ => unreachable!("slot index out of range for compiled arity"),
                }
            }
            fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns {
                ($(SlabColumn::<$O::Acc>::with_capacity(allocator, rows),)+)
            }
            #[inline(always)]
            fn push(accs: &Self::Accs, cols: &mut Self::Columns) {
                $(cols.$idx.push(accs.$idx);)+
            }
            fn finish(cols: Self::Columns, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
                let mut fields = Vec::new();
                let mut arrays = Vec::new();
                $({
                    let (f, a) = $O::finish(&format!("v{}", $idx), cols.$idx, arena);
                    fields.push(f);
                    arrays.push(a);
                })+
                (fields, arrays)
            }
        }
    };
}

impl_optuple!(O0 0);
impl_optuple!(O0 0, O1 1);
impl_optuple!(O0 0, O1 1, O2 2);
impl_optuple!(O0 0, O1 1, O2 2, O3 3);
impl_optuple!(O0 0, O1 1, O2 2, O3 3, O4 4);
impl_optuple!(O0 0, O1 1, O2 2, O3 3, O4 4, O5 5);

/// A fixed aggregate signature: `N` slots given by the op tuple `Ops`, each its
/// own cell. The cells live in `Ops::Accs`; everything else delegates to
/// [`OpTuple`].
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

impl<Ops: OpTuple> AggregationValue for Compiled<Ops> {
    type Reader<'b> = Ops::Reader<'b>;
    type MergeConfig = Ops::Cfg;
    type Columns = Ops::Columns;
    type SortKey = i128;

    #[inline(always)]
    fn merge_config(_slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Ops::Cfg {
        Ops::cfg(arena)
    }
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Ops::Reader<'b> {
        Ops::make_reader(batch, slots)
    }
    #[inline(always)]
    fn value(reader: &Ops::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self {
        Self {
            accs: Ops::seed(reader, idx, arena),
        }
    }
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Ops::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Ops::Cfg,
    ) -> Self {
        Self {
            accs: Ops::update(self.accs, reader, idx, arena, cfg),
        }
    }
    #[inline(always)]
    fn merge(self, other: Self, cfg: &Ops::Cfg) -> Self {
        Self {
            accs: Ops::merge(self.accs, other.accs, cfg),
        }
    }
    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        Ops::sort_key(&self.accs, slot)
    }
    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Ops::Columns {
        Ops::new_columns(allocator, rows)
    }
    #[inline(always)]
    fn push_to(&self, cols: &mut Ops::Columns) {
        Ops::push(&self.accs, cols)
    }
    fn finish_columns(cols: Ops::Columns, arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        Ops::finish(cols, arena)
    }
}
