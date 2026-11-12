//! Keeping spilled buffers until their ring slots are actually needed.
//!
//! Spilling writes a buffer's rows to disk; it does not free the buffer. The
//! buffer stays resident, and its rows readable, in the [`PartitionEvictor`],
//! laid out in the same [`PartitionTree`] its slabs were allocated from. The
//! memory context reaches the evictor through the [`Evict`] handle an operator
//! registers at build time, and takes a slot only when it has none to give:
//! the first buffer, in spill order, whose writes have all landed and that no
//! merge job is reading.
//!
//! A worker whose own evictor has no such buffer takes one from the evictor of
//! another worker on its node, and when every buffer on the node is still
//! being written it is woken once one of them can be taken (see
//! [`SharedPartitionEvictor`]).
//!
//! The evictor is the one holder of its buffers. The merge gets weak handles
//! to the spilled slabs ([`spilled_slabs_by_partition`]), upgrades the ones it
//! reads for as long as it reads them, and falls back to the partition file
//! for a slab whose buffer is gone, so eviction goes on through the merge
//! without the two ever having to agree on anything but a reference count.
//!
//! [`spilled_slabs_by_partition`]: PartitionEvictor::spilled_slabs_by_partition

use crate::memory::evict::{Evict, Evicted};
use crate::memory::spill::partition_tree::PartitionTree;
use crate::memory::spill::waiting_workers::WaitingWorkers;
use crate::memory::spill::{SpilledBuffer, SpilledSlab};
use crate::memory::{WriteBuffer, memory_ctx};
use crate::numa::Topology;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};

/// Spilled buffers awaiting eviction, each under the partition group slot it
/// was allocated from, and the slabs of the buffers already evicted, whose
/// rows exist only in the partition files now.
pub struct PartitionEvictor {
    tree: PartitionTree<Vec<SpilledBuffer>>,
    evicted: Vec<SpilledSlab>,
    /// The workers of this evictor's node that are waiting for a buffer, to
    /// wake when one of this evictor's can be taken or all of them go away.
    waiting_workers: Arc<WaitingWorkers>,
}

/// One worker's evictor as its operator and its memory context share it,
/// together with the evictors of the other workers on its node.
///
/// A spilled buffer whose writes have landed is as good a slot to any worker
/// of the node as to the one that spilled it, and that one may not need a slot
/// for a while. So a worker whose own evictor has no buffer to give takes one
/// from a sibling's.
///
/// Each evictor is behind a lock that is held only for a push, an eviction or
/// the listing of its slabs. A worker that finds one held spins until it is
/// let go instead of sleeping, so the time workers lose to each other is spent
/// on the CPU, in [`lock_spinning`](Self::lock_spinning), and not hidden in a
/// wait. The operator and the memory context of one worker never hold its
/// evictor at the same time: the operator locks it between allocations, and
/// the memory context evicts from it inside an allocation.
///
/// A worker that finds no buffer to take, while some buffer on the node still
/// has writes in flight, has nothing to do until one of those lands. It
/// records itself with the node's [`WaitingWorkers`] and says it will be woken
/// (see [`Evicted::NothingUntilWoken`]). It is woken when a buffer's last
/// write lands, and when an evictor goes away with its buffers, as a failed
/// dataflow's does, since the writes it was waiting for may never be counted
/// off. A buffer whose rows are on disk but which a merge job is reading does
/// not count: nothing wakes a worker when the job lets go of it.
#[derive(Clone)]
pub struct SharedPartitionEvictor {
    /// The worker this evictor belongs to, as a global worker index.
    worker: usize,
    own: Arc<Mutex<PartitionEvictor>>,
    /// The evictors of the other workers on this worker's node, starting with
    /// the worker after this one, so that workers short of slots at the same
    /// time start with different siblings. Each is held weakly, so that it
    /// goes away, and frees its buffers, with its own worker's operator.
    siblings: Arc<[Weak<Mutex<PartitionEvictor>>]>,
    waiting_workers: Arc<WaitingWorkers>,
}

impl SharedPartitionEvictor {
    /// An evictor for each worker of `topology`, in worker order, each knowing
    /// the evictors of the other workers on its node.
    pub fn create_for_workers(topology: Topology) -> Vec<Self> {
        let nodes_waiting_workers: Vec<Arc<WaitingWorkers>> = (0..topology.node_count)
            .map(|node| {
                let first_worker = node * topology.workers_per_node;
                Arc::new(WaitingWorkers::new(topology.node_siblings(first_worker)))
            })
            .collect();
        let evictors: Vec<Arc<Mutex<PartitionEvictor>>> = (0..topology.total_workers())
            .map(|worker| {
                let waiting_workers =
                    nodes_waiting_workers[topology.node_of_worker(worker)].clone();
                Arc::new(Mutex::new(PartitionEvictor::new(waiting_workers)))
            })
            .collect();
        (0..topology.total_workers())
            .map(|worker| {
                let node_workers = topology.node_siblings(worker);
                let siblings = (worker + 1..node_workers.end)
                    .chain(node_workers.start..worker)
                    .map(|sibling| Arc::downgrade(&evictors[sibling]))
                    .collect();
                Self {
                    worker,
                    own: evictors[worker].clone(),
                    siblings,
                    waiting_workers: nodes_waiting_workers[topology.node_of_worker(worker)].clone(),
                }
            })
            .collect()
    }

    /// The evictor of a worker that has no siblings.
    pub fn create_without_siblings() -> Self {
        Self::create_for_workers(Topology::single_node(1))
            .pop()
            .expect("one worker has one evictor")
    }

    /// Lock this worker's own evictor, to push into it, evict from it or list
    /// its slabs.
    pub fn lock_own(&self) -> MutexGuard<'_, PartitionEvictor> {
        Self::lock_spinning(&self.own)
    }

    /// Lock `evictor`, spinning for as long as another worker holds it.
    fn lock_spinning(evictor: &Mutex<PartitionEvictor>) -> MutexGuard<'_, PartitionEvictor> {
        loop {
            match evictor.try_lock() {
                Ok(guard) => return guard,
                Err(TryLockError::WouldBlock) => std::hint::spin_loop(),
                Err(TryLockError::Poisoned(_)) => {
                    panic!("a worker panicked while it held a partition evictor")
                }
            }
        }
    }

    /// Take the first buffer this worker's evictor can give up or, when it has
    /// none, the first one a sibling's can. When there is none to take, says
    /// whether any buffer on the node still has writes in flight. This
    /// worker's evictor is let go of before a sibling's is locked, so two
    /// workers taking from each other never wait on one another while holding
    /// their own.
    fn take_from_node(&self) -> ControlFlow<WriteBuffer, bool> {
        let mut has_writes_in_flight = {
            let mut own = self.lock_own();
            if let Some(buffer) = own.evict() {
                return ControlFlow::Break(buffer);
            }
            own.has_writes_in_flight()
        };
        for sibling in self.siblings.iter() {
            // A sibling whose operator is gone took its buffers with it.
            let Some(sibling) = sibling.upgrade() else {
                continue;
            };
            let mut evictor = Self::lock_spinning(&sibling);
            if let Some(buffer) = evictor.evict() {
                return ControlFlow::Break(buffer);
            }
            has_writes_in_flight |= evictor.has_writes_in_flight();
        }
        ControlFlow::Continue(has_writes_in_flight)
    }
}

impl PartitionEvictor {
    pub fn new(waiting_workers: Arc<WaitingWorkers>) -> Self {
        Self {
            tree: PartitionTree::new(Vec::new),
            evicted: Vec::new(),
            waiting_workers,
        }
    }

    /// Keep `spilled` until it is evicted, and have its last write wake the
    /// workers waiting for a buffer when it lands.
    pub fn push(&mut self, spilled: SpilledBuffer) {
        spilled
            .writes_in_flight
            .wake_when_landed(self.waiting_workers.clone());
        self.tree
            .node_mut(spilled.partition_group_slot)
            .push(spilled);
    }

    /// Give up the first buffer, in spill order, whose rows are all on disk
    /// and that nothing else holds, or `None` when there is no such buffer
    /// right now: every spilled buffer still has writes in flight, or a merge
    /// job is reading it.
    pub fn evict(&mut self) -> Option<WriteBuffer> {
        let candidate = self.tree.walk_spill_order(|_, spilled_buffers| {
            match spilled_buffers
                .iter()
                .position(|spilled| spilled.is_on_disk() && Arc::strong_count(&spilled.buffer) == 1)
            {
                Some(idx) => ControlFlow::Break(spilled_buffers.remove(idx)),
                None => ControlFlow::Continue(()),
            }
        });
        let ControlFlow::Break(SpilledBuffer {
            writes_in_flight,
            buffer,
            slabs,
            partition_group_slot,
        }) = candidate
        else {
            return None;
        };
        let mut write_buffer = match Arc::try_unwrap(buffer) {
            Ok(write_buffer) => write_buffer,
            // A merge job upgraded its handle between the count above and
            // here; the buffer stays, and the caller asks again.
            Err(buffer) => {
                self.tree
                    .node_mut(partition_group_slot)
                    .push(SpilledBuffer {
                        writes_in_flight,
                        buffer,
                        slabs,
                        partition_group_slot,
                    });
                return None;
            }
        };
        // From here on these rows can only be read from the partition files.
        self.evicted.extend(slabs);
        // The handle still says whether the slot was zeroed when the buffer
        // was taken for rows; it has been filled with them since. The next
        // taker gets a dirty buffer, so one that wants zeros zeroes it.
        write_buffer.zeroed = false;
        memory_ctx()
            .ring()
            .set_slot_zeroed(write_buffer.slot_idx, false);
        memory_ctx().record_spilled_buffers_released(1);
        Some(write_buffer)
    }

    /// Whether any buffer kept here still has a write in flight, and so will
    /// become evictable without anything else having to happen.
    pub fn has_writes_in_flight(&self) -> bool {
        self.tree
            .nodes()
            .iter()
            .flatten()
            .any(|spilled| !spilled.is_on_disk())
    }

    /// Every spilled slab, by partition, for the merge: the slab's handle
    /// upgrades for as long as its buffer is resident, and the slab's rows are
    /// at the given offset of the partition's file for when it is not. Slabs
    /// of buffers already evicted come with a handle that no longer upgrades.
    /// An empty list when nothing spilled, which is the usual case, so a
    /// flush then builds nothing for it.
    pub fn spilled_slabs_by_partition(&mut self, partitions: usize) -> Vec<Vec<SpilledSlab>> {
        if self.evicted.is_empty() && self.tree.nodes().iter().all(Vec::is_empty) {
            return Vec::new();
        }
        let mut by_partition: Vec<Vec<SpilledSlab>> = (0..partitions).map(|_| Vec::new()).collect();
        let _: ControlFlow<()> = self.tree.walk_spill_order(|_, spilled_buffers| {
            for slab in spilled_buffers.iter().flat_map(|spilled| &spilled.slabs) {
                by_partition[slab.spill_partition_id].push(slab.clone());
            }
            ControlFlow::Continue(())
        });
        for slab in &self.evicted {
            by_partition[slab.spill_partition_id].push(slab.clone());
        }
        by_partition
    }
}

impl Drop for PartitionEvictor {
    /// An evictor dropped while still holding spilled buffers, as a failed
    /// dataflow's is, frees them with it; they stop counting as spilled, or
    /// the memory context would go on seeing slots to reclaim that are gone.
    /// A worker waiting for one of them is woken to look again: their writes
    /// may never be counted off now, so nothing else would wake it.
    fn drop(&mut self) {
        let mut held = 0;
        let _: ControlFlow<()> = self.tree.walk_spill_order(|_, spilled_buffers| {
            held += spilled_buffers.len();
            ControlFlow::Continue(())
        });
        if held > 0 {
            memory_ctx().record_spilled_buffers_released(held);
            self.waiting_workers.wake_all();
        }
    }
}

impl Evict for SharedPartitionEvictor {
    /// Give up a buffer of this worker's, or, when it has none to give, the
    /// first one a sibling has. When there is none to take but some are still
    /// being written, this worker is recorded as waiting and the node is
    /// looked over once more before the answer says it will be woken (see
    /// [`WaitingWorkers::register`]).
    fn evict(&self) -> Evicted {
        match self.take_from_node() {
            ControlFlow::Break(buffer) => return Evicted::Buffer(buffer),
            ControlFlow::Continue(false) => return Evicted::Nothing,
            ControlFlow::Continue(true) => {}
        }
        self.waiting_workers.register(self.worker);
        match self.take_from_node() {
            ControlFlow::Break(buffer) => {
                self.waiting_workers.deregister(self.worker);
                Evicted::Buffer(buffer)
            }
            ControlFlow::Continue(false) => {
                self.waiting_workers.deregister(self.worker);
                Evicted::Nothing
            }
            ControlFlow::Continue(true) => Evicted::NothingUntilWoken,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::FileRange;
    use crate::memory::spill::WritesInFlight;
    use crate::memory::{Slab, init_test_free_pool};
    use crate::waker::worker_waker;

    /// A spilled buffer with one 1KB slab of partition 0, filed under
    /// `partition_group_slot`, with `writes_in_flight` of its writes still to
    /// land. Counted as spilled with the memory context, as the spiller would
    /// have.
    fn spilled(partition_group_slot: usize, writes_in_flight: usize) -> SpilledBuffer {
        memory_ctx().record_buffers_spilled(1);
        let buffer = Arc::new(memory_ctx().get_write_buffer(false));
        let slab = Slab::new(buffer.ptr, 1024, buffer.clone()).freeze();
        let issued = Arc::new(WritesInFlight::new());
        (0..writes_in_flight).for_each(|_| issued.count_issued_write());
        SpilledBuffer {
            writes_in_flight: issued,
            slabs: vec![SpilledSlab {
                spill_partition_id: 0,
                file_offset: 0,
                stored_len: slab.size(),
                slab: slab.downgrade(),
            }],
            buffer,
            partition_group_slot,
        }
    }

    #[test]
    fn nothing_is_evicted_while_a_buffers_writes_are_in_flight() {
        init_test_free_pool(1);
        let shared = SharedPartitionEvictor::create_without_siblings();
        let mut evictor = shared.lock_own();
        evictor.push(spilled(0, 1));

        let evicted = evictor.evict();

        assert!(evicted.is_none());
    }

    #[test]
    fn an_evicted_slabs_handle_no_longer_upgrades_but_still_says_where_its_rows_are() {
        init_test_free_pool(2);
        let shared = SharedPartitionEvictor::create_without_siblings();
        let mut evictor = shared.lock_own();
        evictor.push(spilled(0, 0));
        evictor.push(spilled(0, 1));

        evictor.evict().unwrap();
        let handles = evictor.spilled_slabs_by_partition(1).remove(0);
        let resident: Vec<bool> = handles
            .iter()
            .map(|handle| handle.slab.upgrade().is_some())
            .collect();
        let ranges: Vec<FileRange> = handles
            .iter()
            .map(|handle| FileRange::new(handle.file_offset as usize, handle.slab.size()))
            .collect();

        assert_eq!(resident, [true, false]);
        assert_eq!(ranges, [FileRange::new(0, 1024), FileRange::new(0, 1024)]);
    }

    #[test]
    fn a_buffer_whose_writes_landed_is_given_up_whole() {
        init_test_free_pool(1);
        let shared = SharedPartitionEvictor::create_without_siblings();
        let mut evictor = shared.lock_own();
        let on_disk = spilled(0, 0);
        let slot_idx = on_disk.buffer.slot_idx;
        evictor.push(on_disk);

        let evicted = evictor.evict();
        let again = evictor.evict();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(slot_idx));
        assert!(again.is_none());
    }

    #[test]
    fn eviction_takes_a_deeper_slot_before_the_root() {
        init_test_free_pool(2);
        let shared = SharedPartitionEvictor::create_without_siblings();
        let mut evictor = shared.lock_own();
        let at_root = spilled(0, 0);
        let at_right_child = spilled(2, 0);
        let (root_slot, child_slot) = (at_root.buffer.slot_idx, at_right_child.buffer.slot_idx);
        evictor.push(at_root);
        evictor.push(at_right_child);

        let first = evictor.evict().unwrap();
        let second = evictor.evict().unwrap();

        assert_eq!(first.slot_idx, child_slot);
        assert_eq!(second.slot_idx, root_slot);
    }

    #[test]
    fn a_buffer_whose_slab_a_reader_holds_is_not_evicted_until_it_is_let_go() {
        init_test_free_pool(1);
        let shared = SharedPartitionEvictor::create_without_siblings();
        let mut evictor = shared.lock_own();
        evictor.push(spilled(0, 0));
        let handle = evictor.spilled_slabs_by_partition(1).remove(0).remove(0);

        let held = handle.slab.upgrade();
        let while_held = evictor.evict();
        drop(held);
        let after = evictor.evict();

        assert!(while_held.is_none());
        assert!(after.is_some());
    }

    #[test]
    fn the_shared_handle_evicts_through_the_evict_trait() {
        init_test_free_pool(1);
        let shared = SharedPartitionEvictor::create_without_siblings();
        shared.lock_own().push(spilled(5, 0));
        let handle: Box<dyn Evict> = Box::new(shared.clone());

        let evicted = handle.evict().into_buffer();

        assert!(evicted.is_some());
    }

    #[test]
    fn a_worker_with_no_buffer_of_its_own_takes_a_siblings() {
        init_test_free_pool(1);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        let on_disk = spilled(0, 0);
        let slot_idx = on_disk.buffer.slot_idx;
        sibling.lock_own().push(on_disk);

        let evicted = worker.evict().into_buffer();
        let left_with_the_sibling = sibling.evict();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(slot_idx));
        assert!(matches!(left_with_the_sibling, Evicted::Nothing));
    }

    #[test]
    fn a_workers_own_buffer_is_taken_before_a_siblings() {
        init_test_free_pool(2);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        let own = spilled(0, 0);
        let own_slot_idx = own.buffer.slot_idx;
        sibling.lock_own().push(spilled(0, 0));
        worker.lock_own().push(own);

        let evicted = worker.evict().into_buffer();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(own_slot_idx));
    }

    #[test]
    fn a_worker_is_told_it_will_be_woken_while_a_siblings_writes_are_in_flight() {
        init_test_free_pool(1);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        sibling.lock_own().push(spilled(0, 1));

        let evicted = worker.evict();

        assert!(matches!(evicted, Evicted::NothingUntilWoken));
    }

    #[test]
    fn a_waiting_worker_is_woken_when_a_siblings_last_write_lands() {
        init_test_free_pool(1);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        let being_written = spilled(0, 2);
        let writes_in_flight = being_written.writes_in_flight.clone();
        sibling.lock_own().push(being_written);
        let wakes_before = worker_waker().ring_wake_count();

        let while_written = worker.evict();
        writes_in_flight.count_off_landed_write();
        let wakes_with_a_write_left = worker_waker().ring_wake_count();
        writes_in_flight.count_off_landed_write();
        let wakes_once_landed = worker_waker().ring_wake_count();
        let once_landed = worker.evict();

        assert!(matches!(while_written, Evicted::NothingUntilWoken));
        assert_eq!(wakes_with_a_write_left, wakes_before);
        assert_eq!(wakes_once_landed, wakes_before + 1);
        assert!(matches!(once_landed, Evicted::Buffer(_)));
    }

    #[test]
    fn a_waiting_worker_is_woken_when_the_sibling_it_waits_for_goes_away() {
        init_test_free_pool(1);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        sibling.lock_own().push(spilled(0, 1));
        let wakes_before = worker_waker().ring_wake_count();

        let while_written = worker.evict();
        drop(sibling);
        let wakes_once_gone = worker_waker().ring_wake_count();
        let once_gone = worker.evict();

        assert!(matches!(while_written, Evicted::NothingUntilWoken));
        assert_eq!(wakes_once_gone, wakes_before + 1);
        assert!(matches!(once_gone, Evicted::Nothing));
    }

    #[test]
    fn a_worker_is_not_told_to_wait_for_a_buffer_a_reader_holds() {
        init_test_free_pool(1);
        let mut workers = SharedPartitionEvictor::create_for_workers(Topology::single_node(2));
        let (sibling, worker) = (workers.remove(1), workers.remove(0));
        sibling.lock_own().push(spilled(0, 0));
        let handle = sibling
            .lock_own()
            .spilled_slabs_by_partition(1)
            .remove(0)
            .remove(0);

        let held = handle.slab.upgrade();
        let evicted = worker.evict();

        assert!(held.is_some());
        assert!(matches!(evicted, Evicted::Nothing));
    }

    #[test]
    fn a_worker_takes_nothing_from_a_worker_on_another_node() {
        init_test_free_pool(1);
        let two_nodes = Topology {
            workers_per_node: 1,
            node_count: 2,
        };
        let mut workers = SharedPartitionEvictor::create_for_workers(two_nodes);
        let (on_other_node, worker) = (workers.remove(1), workers.remove(0));
        on_other_node.lock_own().push(spilled(0, 0));

        let evicted = worker.evict();

        assert!(matches!(evicted, Evicted::Nothing));
    }
}
