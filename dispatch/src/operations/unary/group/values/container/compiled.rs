//! [`Compiled`] — a fixed signature monomorphised over a tuple of
//! [`Aggregation`] ops, straight-line with no per-row dispatch.
//!
//! Each slot is a whole op with its own input array and cell, so a `Compiled`
//! shape mixes families freely — a string extreme beside an integer one
//! (`Compiled<(StrMin, Max<Int32Type>)>`) — each reading its *own typed array*.
//! That's what keeps string extremes lazy and removes every int/str special case.
//! The planner instantiates the tuple it needs; numeric runtime signatures fall
//! back to [`Dynamic`](super::Dynamic).
//!
//! The per-arity [`OpTuple`] impls carry the tuple plumbing, so the
//! [`AggregationValue`] impl for [`Compiled`] is one thin delegation.

use super::slot::Slot;
use super::super::{AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// A tuple of [`Slot`]s (each a `(Read, Fold)` pair), with the per-slot plumbing
/// the container needs. Implemented for tuples of arity 1–6 by the macro below.
pub trait OpTuple: Send + Sync + 'static {
    /// The heterogeneous cell tuple — one cell per op, each its own width.
    type Accs: Copy + Default + Send + Sync + 'static;
    /// The bound input arrays, one per op.
    type Inputs<'b>;
    type Cfg: Clone + Send + Sync + 'static;
    type Columns;

    fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg;
    fn bind<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Inputs<'b>;
    fn seed(inputs: &Self::Inputs<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Accs;
    fn update(
        accs: Self::Accs,
        inputs: &Self::Inputs<'_>,
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
        impl<$($O: Slot),+> OpTuple for ($($O,)+) {
            type Accs = ($($O::Acc,)+);
            type Inputs<'b> = ($($O::Input<'b>,)+);
            type Cfg = ($($O::Cfg,)+);
            type Columns = ($(SlabColumn<$O::Acc>,)+);

            #[inline(always)]
            fn cfg(arena: &Arc<SharedArena>) -> Self::Cfg {
                ($($O::cfg(arena),)+)
            }
            #[inline(always)]
            fn bind<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Inputs<'b> {
                ($($O::bind(batch, slots[$idx].column),)+)
            }
            #[inline(always)]
            fn seed(inputs: &Self::Inputs<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Accs {
                ($($O::seed(&inputs.$idx, idx, arena),)+)
            }
            #[inline(always)]
            fn update(
                accs: Self::Accs,
                inputs: &Self::Inputs<'_>,
                idx: usize,
                arena: &mut WorkerArena,
                cfg: &Self::Cfg,
            ) -> Self::Accs {
                ($($O::update(accs.$idx, &inputs.$idx, idx, arena, &cfg.$idx),)+)
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

/// A fixed aggregate signature: the slots given by the op tuple `Ops`, each its
/// own cell. The cells live in `Ops::Accs`; everything else delegates to [`OpTuple`].
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
    type Reader<'b> = Ops::Inputs<'b>;
    type MergeConfig = Ops::Cfg;
    type Columns = Ops::Columns;
    type SortKey = i128;

    #[inline(always)]
    fn merge_config(_slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Ops::Cfg {
        Ops::cfg(arena)
    }
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Ops::Inputs<'b> {
        Ops::bind(batch, slots)
    }
    #[inline(always)]
    fn value(reader: &Ops::Inputs<'_>, idx: usize, arena: &mut WorkerArena) -> Self {
        Self {
            accs: Ops::seed(reader, idx, arena),
        }
    }
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Ops::Inputs<'_>,
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
    fn finish_columns(
        cols: Ops::Columns,
        arena: &Arc<SharedArena>,
        _cfg: &Ops::Cfg,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        // A fixed signature renders each slot from its static op — the per-slot
        // descriptor a runtime value needs is already in the tuple type.
        Ops::finish(cols, arena)
    }
}
