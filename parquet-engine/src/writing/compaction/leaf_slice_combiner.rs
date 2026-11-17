//! Puts the slices of a column back together before it is encoded.
//!
//! A variant column of an output row group is shredded in slices of rows,
//! each on whichever worker takes it, so the slices' leaves come out in any
//! order. [`LeafSliceCombiner`] keeps a leaf's slices until all of them are
//! there, and then hands the encoder the whole leaf, its slices in row
//! order.

use std::collections::{BTreeMap, HashMap};

use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use super::super::types::{LeafChunkJob, RowGroupId};

/// Joins the slices of each leaf, in row order, into one leaf to encode. A
/// leaf of a column that was not cut passes through as it is. Every leaf is
/// routed to one worker, the compaction worker; the other workers'
/// combiners receive nothing.
#[derive(Default)]
pub(super) struct LeafSliceCombiner {
    /// The slices that have arrived of each leaf still missing some, by the
    /// leaf's row group and its number there, and then by slice number.
    arrived_slices: HashMap<(RowGroupId, usize), BTreeMap<usize, LeafChunkJob>>,
}

impl LeafSliceCombiner {
    /// One combiner per worker.
    pub(super) fn factories(worker_count: usize) -> Vec<DefaultUnaryFactory<Self>> {
        DefaultUnaryFactory::create_for_workers(worker_count)
    }
}

impl Unary<LeafChunkJob, LeafChunkJob> for LeafSliceCombiner {
    fn consume(
        &mut self,
        slice: LeafChunkJob,
        sender: &mut dyn Sender<LeafChunkJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if slice.slice.count == 1 {
            sender.send(slice)?;
            return Ok(());
        }
        let leaf = (slice.context.row_group_id, slice.leaf_index);
        let slice_count = slice.slice.count;
        let arrived = self.arrived_slices.entry(leaf).or_default();
        arrived.insert(slice.slice.index, slice);
        if arrived.len() < slice_count {
            return Ok(());
        }
        // The map goes through the slices in row order.
        let mut slices = self.arrived_slices.remove(&leaf).unwrap().into_values();
        let mut whole = slices.next().expect("a cut column has slices");
        for slice in slices {
            whole.value_chunks.extend(slice.value_chunks);
            if let Some(levels) = &mut whole.def_levels {
                levels.extend_from_slice(
                    &slice
                        .def_levels
                        .expect("a leaf's level depth is fixed by its field"),
                );
            }
        }
        sender.send(whole)?;
        Ok(())
    }

    /// Slices kept here are of leaves that are still to go on.
    fn has_pending_work(&self) -> bool {
        !self.arrived_slices.is_empty()
    }
}
