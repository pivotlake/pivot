//! Hash-slot upper-bound sketch for a pushed-down grouped top-k.
//!
//! When a grouped `ORDER BY <count slot> DESC LIMIT k` is pushed into the
//! group operator, each worker also folds the ordering aggregate's partial
//! values into a fixed array indexed by the top bits of the group hash. The
//! ordering aggregate is additive and nonnegative, so after summing every
//! worker's sketch, a slot's total is an upper bound on the final value of any
//! single group hashing into that slot.
//!
//! Merge partitions cover contiguous slot ranges of the same top hash bits, so
//! each partition gets a bound: the maximum slot total in its range. The merge
//! phase runs partition jobs in descending bound order and, once any worker's
//! top-k heap holds `k` exact (fully merged) groups, skips every partition
//! whose bound is below that k-th value: no group there can displace one of
//! the `k` already found.

use crate::memory::{SlabAllocator, SlabBuffer};
use crate::operations::unary::group::GroupLimit;
use crate::operations::unary::group::values::{AggregationKind, AggregationSlot};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Sketch resolution in top hash bits. Bounds tighten as this grows (the
/// noise under any single group's bound is roughly `rows >> SKETCH_BITS`);
/// the per-worker array is `8 << SKETCH_BITS` bytes.
const SKETCH_BITS: u32 = 16;

/// Number of sketch slots.
pub(crate) const SKETCH_SLOTS: usize = 1 << SKETCH_BITS;

/// Table entries below which a worker that never drained skips building its
/// flush-time sketch. The sketch's allocation and page churn are measurable
/// against a small query's total runtime, and so few entries mean the whole
/// merge is too small for pruning to pay anyway. The merge only prunes when
/// every contributing worker reported a sketch, so a skipped worker disables
/// pruning rather than invalidating bounds. Tests all but drop the floor so
/// the pruning path stays exercised at test-sized group counts.
pub(crate) const MIN_SKETCH_ENTRIES: usize = if cfg!(test) { 1 } else { 8192 };

/// Per-worker (and, after merging, global) sums of the ordering aggregate's
/// partials by top hash bits.
///
/// Slab-backed on purpose: a fresh heap allocation this size per worker per
/// query means an mmap and munmap each time, and on a many-core box the
/// munmaps alone are a TLB-shootdown storm across every CPU. Slab memory is
/// pooled and stays mapped, so a sketch costs one memset of already-resident
/// pages.
pub struct TopKSketch {
    sums: SlabBuffer<u64>,
}

// The slab's pages are address-stable pool memory and the sketch is accessed
// by one owner at a time (a worker, then the accumulator fold, then the
// gather's final arrival).
unsafe impl Send for TopKSketch {}

impl TopKSketch {
    pub(crate) fn new(allocator: &mut SlabAllocator) -> Self {
        Self {
            sums: allocator.create_slab_buffer(SKETCH_SLOTS, true),
        }
    }

    fn as_slice(&self) -> &[u64] {
        // In-bounds: the buffer was created with SKETCH_SLOTS slots.
        unsafe { std::slice::from_raw_parts(self.sums.ptr_at_index(0), SKETCH_SLOTS) }
    }

    /// Folds one partial value of the ordering aggregate into the sketch.
    ///
    /// Saturating: a saturated slot only loosens the bound, which stays valid.
    #[inline(always)]
    pub(crate) fn add(&mut self, hash: u64, weight: u64) {
        let slot = (hash >> (u64::BITS - SKETCH_BITS)) as usize;
        self.sums[slot] = self.sums[slot].saturating_add(weight);
    }

    /// Sums another worker's sketch into this one.
    pub(crate) fn merge(&mut self, other: &TopKSketch) {
        let base = self.sums.ptr_at_index(0);
        for (slot, other_sum) in other.as_slice().iter().enumerate() {
            // One base pointer for the whole pass; `SlabBuffer` indexing would
            // reload it per slot.
            unsafe {
                let sum = base.add(slot);
                *sum = (*sum).saturating_add(*other_sum);
            }
        }
    }

    /// Copies the slot totals out of the slab, freeing it for reuse.
    pub(crate) fn into_slot_totals(self) -> Vec<u64> {
        self.as_slice().to_vec()
    }
}

/// The sketch slot a hash falls into.
#[inline(always)]
pub(crate) fn sketch_slot(hash: u64) -> usize {
    (hash >> (u64::BITS - SKETCH_BITS)) as usize
}

/// Per-slot upper bounds on any single group's final value, shared by every
/// merge job of one query.
///
/// Built from the pool-wide sketch's slot totals, each widened by the raw
/// scatter rows of the bucket covering the slot (those rows are not in the
/// sketch, and at worst one group absorbed a whole bucket of them). Bounds
/// prune at two granularities: a job whose partition's maximum bound is below
/// the k-th best skips entirely, and a job that does run skips the individual
/// source entries of slots that cannot reach it (a group's partials all share
/// a hash and therefore a slot, so slot-level skipping drops a group in full
/// or not at all).
pub(crate) struct SketchBounds {
    slot_bounds: Vec<u64>,
}

impl SketchBounds {
    /// Builds the widened per-slot bounds. `raw_bucket_rows` is the pool-wide
    /// raw scatter row count per bucket, when any worker scattered raw.
    pub(crate) fn build(sketch: TopKSketch, raw_bucket_rows: Option<&[u64]>) -> Self {
        let mut slot_bounds = sketch.into_slot_totals();
        if let Some(bucket_rows) = raw_bucket_rows {
            // Buckets are a coarser prefix of the same top hash bits, so a
            // slot's bucket is its top bits.
            let bucket_shift = SKETCH_SLOTS.trailing_zeros() - bucket_rows.len().trailing_zeros();
            for (slot, bound) in slot_bounds.iter_mut().enumerate() {
                *bound = bound.saturating_add(bucket_rows[slot >> bucket_shift]);
            }
        }
        Self { slot_bounds }
    }

    /// Whether a group hashing to `hash` could still reach `threshold`.
    #[inline(always)]
    pub(crate) fn could_reach(&self, hash: u64, threshold: u64) -> bool {
        self.slot_bounds[sketch_slot(hash)] >= threshold
    }

    /// The smallest slot bound in one partition's hash range. When it already
    /// reaches the threshold, no entry of the partition can fail
    /// [`could_reach`](Self::could_reach), so the caller skips per-entry
    /// filtering instead of paying a bounds lookup per merged entry for
    /// nothing.
    pub(crate) fn partition_min_bound(&self, partition: usize, num_partitions: usize) -> u64 {
        let start = partition * SKETCH_SLOTS / num_partitions;
        let end = ((partition + 1) * SKETCH_SLOTS / num_partitions).max(start + 1);
        self.slot_bounds[start..end]
            .iter()
            .copied()
            .min()
            .expect("a partition covers at least one slot")
    }

    /// The per-partition upper bound on any single group's final value, for a
    /// merge over `num_partitions` (a power of two) hash-prefix partitions.
    ///
    /// A partition's bound is the maximum slot bound over the slots its hash
    /// range covers (a partition finer than a slot shares that slot's bound).
    pub(crate) fn partition_bounds(&self, num_partitions: usize) -> Vec<u64> {
        (0..num_partitions)
            .map(|partition| {
                let start = partition * SKETCH_SLOTS / num_partitions;
                let end = ((partition + 1) * SKETCH_SLOTS / num_partitions).max(start + 1);
                self.slot_bounds[start..end].iter().copied().max().unwrap()
            })
            .collect()
    }
}

/// Per-NUMA-node sketch accumulators that workers fold into as they arrive at
/// the gather barrier.
///
/// Summing the per-worker sketches costs `workers x SKETCH_SLOTS` adds, which
/// grows with the pool while every other cost per worker shrinks; done by the
/// barrier's final arrival it becomes the serial tail of every large top-k
/// query on a big machine. Folding on arrival rides the staggered consume
/// finish instead: each worker adds its own 512KB into its node's accumulator
/// (node-local memory, contended only when two workers of one node arrive
/// together), and the final arrival merges just one sketch per node.
pub(crate) struct SketchAccumulators {
    /// [`STRIPES_PER_NODE`] accumulators per node, so simultaneous arrivals (a
    /// uniform query finishes every worker at once) fold in parallel instead
    /// of serializing a node's whole worker count behind one lock.
    stripes: Vec<Mutex<Option<TopKSketch>>>,
}

/// Accumulator stripes per node. Bounds the folds one straggler can serialize
/// while keeping the final arrival's stripe merge a few megabytes.
const STRIPES_PER_NODE: usize = 8;

impl SketchAccumulators {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            stripes: (0..node_count * STRIPES_PER_NODE)
                .map(|_| Mutex::new(None))
                .collect(),
        }
    }

    /// Folds one worker's sketch into one of its node's accumulators, taking
    /// the first free stripe and only blocking when all are busy.
    pub(crate) fn fold(&self, node: usize, worker: usize, sketch: TopKSketch) {
        let base = node * STRIPES_PER_NODE;
        let start = worker % STRIPES_PER_NODE;
        let mut accumulator = 'claim: {
            for offset in 0..STRIPES_PER_NODE {
                let stripe = base + (start + offset) % STRIPES_PER_NODE;
                if let Ok(accumulator) = self.stripes[stripe].try_lock() {
                    break 'claim accumulator;
                }
            }
            self.stripes[base + start].lock().unwrap()
        };
        match &mut *accumulator {
            Some(merged) => merged.merge(&sketch),
            empty => *empty = Some(sketch),
        }
    }

    /// Merges every stripe into the pool-wide sketch. Called once, by the
    /// gather barrier's final arrival, after every fold.
    pub(crate) fn take_merged(&self) -> Option<TopKSketch> {
        let mut merged: Option<TopKSketch> = None;
        for stripe in &self.stripes {
            if let Some(sketch) = stripe.lock().unwrap().take() {
                match &mut merged {
                    Some(merged) => merged.merge(&sketch),
                    empty => *empty = Some(sketch),
                }
            }
        }
        merged
    }
}

/// The pushed top-k ordering slot, when its aggregate makes slot sums valid
/// upper bounds: the partials must be nonnegative and must not merge to more
/// than their sum. `COUNT` forms qualify; `SUM` may go negative and extremes
/// are not additive, so they do not.
pub(crate) fn sketchable_slot(
    output_limit: Option<GroupLimit>,
    slots: &[AggregationSlot],
) -> Option<usize> {
    // Experiment kill switch: disables the whole sketch/pruning apparatus so
    // an A/B can attribute a regression to it or to the surrounding changes.
    if std::env::var_os("PIVOT_DISABLE_TOPK_SKETCH").is_some() {
        return None;
    }
    let (slot, limit) = match output_limit? {
        GroupLimit::TopK { slot, limit } | GroupLimit::TopKPrune { slot, limit } => (slot, limit),
        GroupLimit::First { .. } => return None,
    };
    // A zero limit emits nothing; the sketch's threshold needs a full heap.
    if limit == 0 {
        return None;
    }
    matches!(
        slots.get(slot)?.kind,
        AggregationKind::CountStar | AggregationKind::Count
    )
    .then_some(slot)
}

/// Shared lower bound on the pushed top-k's k-th best exact group value.
///
/// Every fully merged group's weight is offered from whichever worker merged
/// its partition; once `cap` groups have been seen, the smallest retained
/// weight is the true global k-th best so far, which partition jobs compare
/// their sketch bound against to skip themselves. Global rather than
/// per-worker on purpose: the k best groups land in partitions spread across
/// the pool, so no single worker's local top-k ever tightens to the real
/// k-th value.
///
/// Offers are gated on a relaxed load of the current minimum, so after the
/// first `cap` groups almost every offer is one uncontended atomic read.
pub struct TopKThreshold {
    /// The k-th best weight once `cap` groups have been offered; 0 before,
    /// which no bound is below, so nothing is skipped prematurely.
    kth_best: AtomicU64,
    /// Min-heap of the `cap` largest weights offered so far.
    top: Mutex<BinaryHeap<Reverse<u64>>>,
    cap: usize,
    /// Groups emitted so far under a plain (unordered) pushed LIMIT. Any
    /// `limit` groups satisfy it, so once this reaches the limit every
    /// remaining merge job skips itself.
    emitted_groups: AtomicUsize,
}

impl TopKThreshold {
    /// A threshold for the pushed limit; an absent limit gets a
    /// zero-capacity threshold that never fills and never skips anything.
    pub fn for_limit(output_limit: Option<GroupLimit>) -> Self {
        let cap = match output_limit {
            Some(GroupLimit::TopK { limit, .. }) | Some(GroupLimit::TopKPrune { limit, .. }) => {
                limit
            }
            _ => 0,
        };
        Self {
            kth_best: AtomicU64::new(0),
            top: Mutex::new(BinaryHeap::with_capacity(cap.min(1 << 20))),
            cap,
            emitted_groups: AtomicUsize::new(0),
        }
    }

    /// Records `groups` more emitted under a plain pushed LIMIT.
    #[inline]
    pub fn add_emitted_groups(&self, groups: usize) {
        self.emitted_groups.fetch_add(groups, Ordering::Relaxed);
    }

    /// Groups emitted so far under a plain pushed LIMIT.
    #[inline]
    pub fn emitted_groups(&self) -> usize {
        self.emitted_groups.load(Ordering::Relaxed)
    }

    /// Offers one fully merged group's weight.
    #[inline]
    pub fn offer(&self, weight: u64) {
        // Full and not an improvement: the common case after warmup.
        let kth = self.kth_best.load(Ordering::Relaxed);
        if kth != 0 && weight <= kth {
            return;
        }
        let mut top = self.top.lock().unwrap();
        if top.len() < self.cap {
            top.push(Reverse(weight));
            if top.len() == self.cap {
                self.kth_best
                    .store(top.peek().unwrap().0, Ordering::Relaxed);
            }
        } else if self.cap > 0 && weight > top.peek().unwrap().0 {
            *top.peek_mut().unwrap() = Reverse(weight);
            self.kth_best
                .store(top.peek().unwrap().0, Ordering::Relaxed);
        }
    }

    /// The current lower bound on the k-th best group value (0 until `cap`
    /// groups have been offered).
    #[inline]
    pub fn kth_best(&self) -> u64 {
        self.kth_best.load(Ordering::Relaxed)
    }
}

/// Converts a pushed ORDER BY sort key into a sketch weight, clamping into
/// `u64`. Order-preserving for the nonnegative values the sketch is gated to.
pub trait SketchWeight {
    fn saturating_weight(self) -> u64;
}

impl SketchWeight for i64 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.max(0) as u64
    }
}

impl SketchWeight for i128 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.clamp(0, u64::MAX as i128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    fn test_sketch() -> (SlabAllocator, TopKSketch) {
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(false);
        let sketch = TopKSketch::new(&mut allocator);
        (allocator, sketch)
    }

    #[test]
    fn slot_sums_accumulate_by_top_bits() {
        let (_allocator, mut sketch) = test_sketch();
        let hash_a = 3u64 << (u64::BITS - SKETCH_BITS);
        let hash_b = hash_a | 0x1fff; // Same top bits, different low bits.

        sketch.add(hash_a, 5);
        sketch.add(hash_b, 7);

        assert_eq!(sketch.sums[3], 12);
        assert_eq!(sketch.sums[2], 0);
    }

    #[test]
    fn merge_sums_workers_elementwise() {
        let (mut allocator, mut a) = test_sketch();
        let mut b = TopKSketch::new(&mut allocator);
        let hash = 9u64 << (u64::BITS - SKETCH_BITS);
        a.add(hash, 10);
        b.add(hash, 32);

        a.merge(&b);

        assert_eq!(a.sums[9], 42);
    }

    #[test]
    fn partition_bounds_take_range_maxima() {
        let (_allocator, mut sketch) = test_sketch();
        // Two slots inside partition 1 of 2: the bound takes the larger.
        let upper_half = 1u64 << (u64::BITS - 1);
        sketch.add(upper_half, 100);
        sketch.add(upper_half | (1 << (u64::BITS - SKETCH_BITS)), 250);

        let bounds = SketchBounds::build(sketch, None).partition_bounds(2);

        assert_eq!(bounds, vec![0, 250]);
    }

    #[test]
    fn partitions_finer_than_slots_share_the_slot_total() {
        let (_allocator, mut sketch) = test_sketch();
        sketch.add(0, 8);

        let bounds = SketchBounds::build(sketch, None).partition_bounds(2 * SKETCH_SLOTS);

        assert_eq!(bounds[0], 8);
        assert_eq!(bounds[1], 8);
        assert_eq!(bounds[2], 0);
    }

    #[test]
    fn count_slots_are_sketchable_and_others_are_not() {
        use arrow_schema::DataType;
        let slots = vec![
            AggregationSlot::new(AggregationKind::Sum, 0, DataType::Int64),
            AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
        ];
        let topk = |slot| Some(GroupLimit::TopK { slot, limit: 10 });

        assert_eq!(sketchable_slot(topk(1), &slots), Some(1));
        assert_eq!(sketchable_slot(topk(0), &slots), None);
        assert_eq!(
            sketchable_slot(Some(GroupLimit::First { limit: 10 }), &slots),
            None
        );
        assert_eq!(sketchable_slot(None, &slots), None);
    }
}
