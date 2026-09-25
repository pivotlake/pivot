//! The pool-wide bin totals: every worker publishes its own, and the pool
//! sums them in parallel by bin range.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::output::topk_pruning::{HASH_BINS, HashBinTotals};
use crossbeam_deque::{Injector, Steal};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bins per summing range. A range's work is `workers x BINS_PER_RANGE`
/// reads, so this keeps each claim short and the ranges plentiful enough to
/// spread over a large pool.
const BINS_PER_RANGE: usize = 64;

/// Number of summing ranges.
const RANGES: usize = HASH_BINS / BINS_PER_RANGE;

/// The pool-wide bin totals under construction: every worker publishes its
/// own totals when it finishes consuming, and after the gather barrier's
/// final arrival the pool sums them.
///
/// The sum is spread over the workers on purpose. Done by the final arrival
/// alone it would read `workers x HASH_BINS` cold, remote counters on one
/// thread, the serial tail of every large top-k query on a big machine,
/// while every other worker sits polling for the merge jobs that cannot
/// exist before the sum. Instead the final arrival opens the sum over the
/// published arrays, and every polling worker claims bin ranges and sums
/// them until none are left; the final arrival helps, then takes the sum
/// once every claimed range is done. Per-node queues keep publication
/// node-local; the sum is the one step that reads across nodes.
pub(crate) struct SharedBinTotals {
    /// Per node: the published arrays.
    published: Vec<Injector<HashBinTotals>>,
    /// The sum in progress, opened once by the final arrival.
    sum: OnceLock<SumInProgress>,
    /// Next range to claim; claims past the last range find nothing.
    next_range: AtomicUsize,
    /// Ranges summed so far.
    done_ranges: AtomicUsize,
}

/// The arrays being summed and the array they are summed into, which the
/// helpers write in disjoint ranges.
struct SumInProgress {
    sources: Vec<HashBinTotals>,
    totals: HashBinTotals,
}

impl SharedBinTotals {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            published: (0..node_count).map(|_| Injector::new()).collect(),
            sum: OnceLock::new(),
            next_range: AtomicUsize::new(0),
            done_ranges: AtomicUsize::new(0),
        }
    }

    /// Publishes one worker's totals.
    pub(crate) fn add(&self, node: usize, totals: HashBinTotals) {
        self.published[node].push(totals);
    }

    /// Sums every published array into the pool-wide totals, with the help
    /// of every worker polling for merge jobs. Called once, by the gather
    /// barrier's final arrival, after every worker's add. `None` when no
    /// worker published.
    pub(crate) fn sum(&self, allocator: &mut SlabAllocator) -> Option<HashBinTotals> {
        let mut sources = Vec::new();
        for published in &self.published {
            loop {
                match published.steal() {
                    Steal::Success(totals) => sources.push(totals),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        if sources.is_empty() {
            return None;
        }
        let opened = self.sum.set(SumInProgress {
            sources,
            totals: HashBinTotals::new(allocator),
        });
        assert!(opened.is_ok(), "the bin totals are summed once");
        while self.help() {}
        while self.done_ranges.load(Ordering::Acquire) < RANGES {
            std::hint::spin_loop();
        }
        // Every range was summed by its claimer, whose `done_ranges`
        // increment (Release) the load above saw; the sum is complete and
        // no helper touches it again. The arrays stay with the shared state
        // until the operator drops.
        let mut totals = HashBinTotals::new(allocator);
        totals.merge(&self.sum.get().expect("opened above").totals);
        Some(totals)
    }

    /// Claims and sums one range, if any is left to claim. For workers with
    /// nothing else to do while the merge phase is being planned; `false`
    /// when no sum is open or every range is claimed.
    pub(crate) fn help(&self) -> bool {
        // `OnceLock::get` acquires, so an open sum is seen complete with its
        // sources and totals.
        let Some(sum) = self.sum.get() else {
            return false;
        };
        // A plain load first: workers keep polling for jobs long after the
        // ranges are gone, and a claim attempt each time would have them all
        // hammering one cache line.
        if self.next_range.load(Ordering::Relaxed) >= RANGES {
            return false;
        }
        let range = self.next_range.fetch_add(1, Ordering::Relaxed);
        if range >= RANGES {
            return false;
        }
        let start = range * BINS_PER_RANGE;
        // SAFETY: each range is claimed by exactly one helper (the
        // fetch_add), so no two writers touch the same bins, and the sum is
        // read only once `done_ranges` reports every range.
        unsafe {
            sum.totals
                .merge_range(start..start + BINS_PER_RANGE, &sum.sources)
        };
        self.done_ranges.fetch_add(1, Ordering::Release);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::output::topk_pruning::HASH_BIN_BITS;

    fn totals_with(allocator: &mut SlabAllocator, bin: u64, weight: u64) -> HashBinTotals {
        let mut totals = HashBinTotals::new(allocator);
        totals.add(bin << (u64::BITS - HASH_BIN_BITS), weight);
        totals
    }

    #[test]
    fn the_sum_folds_every_published_array_and_helpers_have_nothing_before_it() {
        init_test_free_pool(64);
        let mut allocator = SlabAllocator::new(false);
        let shared = SharedBinTotals::new(2);
        let helped_before = shared.help();
        shared.add(0, totals_with(&mut allocator, 3, 5));
        shared.add(1, totals_with(&mut allocator, 3, 7));
        shared.add(1, totals_with(&mut allocator, HASH_BINS as u64 - 1, 1));

        let sum = shared
            .sum(&mut allocator)
            .expect("three arrays were published");

        assert!(!helped_before);
        assert_eq!(sum.into_bin_totals()[3], 12);
        assert!(!shared.help(), "every range was summed");
    }

    #[test]
    fn bins_at_both_ends_of_the_array_are_summed() {
        init_test_free_pool(64);
        let mut allocator = SlabAllocator::new(false);
        let shared = SharedBinTotals::new(1);
        shared.add(0, totals_with(&mut allocator, 0, 4));
        shared.add(0, totals_with(&mut allocator, 0, 6));
        shared.add(0, totals_with(&mut allocator, HASH_BINS as u64 - 1, 9));

        let sum = shared
            .sum(&mut allocator)
            .expect("published")
            .into_bin_totals();

        assert_eq!(sum[0], 10);
        assert_eq!(sum[HASH_BINS - 1], 9);
        assert_eq!(sum[1], 0);
    }

    #[test]
    fn no_published_array_means_no_sum() {
        init_test_free_pool(64);
        let mut allocator = SlabAllocator::new(false);
        let shared = SharedBinTotals::new(1);

        let sum = shared.sum(&mut allocator);

        assert!(sum.is_none());
    }
}
