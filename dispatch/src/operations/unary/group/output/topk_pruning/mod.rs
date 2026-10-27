//! Merge-phase pruning for a pushed-down grouped top-k.
//!
//! When a grouped `ORDER BY <count slot> DESC LIMIT k` is pushed into the
//! group operator, each worker also sums the ordering aggregate's partial
//! values into a fixed array of hash bins indexed by the top bits of the
//! group hash. The ordering aggregate is additive and nonnegative, so after
//! summing every worker's bin totals, a bin's total is an upper bound on the
//! final value of any single group hashing into that bin.
//!
//! Merge partitions cover contiguous bin ranges of the same top hash bits, so
//! each partition gets a bound: the maximum bin total in its range. The merge
//! phase runs partition jobs in descending bound order and, once any worker's
//! top-k heap holds `k` exact (fully merged) groups, skips every partition
//! whose bound is below that k-th value: no group there can displace one of
//! the `k` already found.
//!
//! [`prunable_topk`] decides whether a pushed limit qualifies. The
//! per-worker [`HashBinTotals`] are summed through the pool-wide worker summaries into
//! [`TopKBounds`], which merge jobs check against the [`TopKThreshold`] to
//! form the [`TopKCutoff`] their scans apply.

mod bin_totals;
mod bounds;
mod threshold;

pub use bin_totals::HashBinTotals;
pub(crate) use bounds::{TopKBounds, TopKCutoff};
pub use threshold::TopKThreshold;

use crate::operations::unary::group::GroupLimit;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{LiveKey, PersistedKey};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::values::{AggregationKind, AggregationSlot};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Bin resolution in top hash bits. Bounds tighten as this grows (the noise
/// under any single group's bound is roughly `rows >> HASH_BIN_BITS`); the
/// per-worker array is `8 << HASH_BIN_BITS` bytes.
pub(super) const HASH_BIN_BITS: u32 = 16;

/// Number of hash bins.
pub(crate) const HASH_BINS: usize = 1 << HASH_BIN_BITS;

/// Table entries below which a worker that never drained skips building its
/// flush-time bin totals. The array's allocation and page churn are
/// measurable against a small query's total runtime, and so few entries mean
/// the whole merge is too small for pruning to pay anyway. The merge only
/// prunes when every contributing worker reported bin totals, so a skipped
/// worker disables pruning rather than invalidating bounds. Tests all but
/// drop the floor so the pruning path stays exercised at test-sized group
/// counts.
pub(crate) const MIN_BINNED_ENTRIES: usize = if cfg!(test) { 1 } else { 8192 };

/// The hash bin a hash falls into.
#[inline(always)]
pub(crate) fn hash_bin(hash: u64) -> usize {
    (hash >> (u64::BITS - HASH_BIN_BITS)) as usize
}

/// A pushed top-k the merge can prune: its ordering slot and limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrunableTopK {
    pub slot: usize,
    pub limit: usize,
}

/// The pushed top-k, when its ordering aggregate makes bin sums valid
/// upper bounds: the partials must be nonnegative and must not merge to more
/// than their sum. `COUNT` forms qualify; `SUM` may go negative and extremes
/// are not additive, so they do not.
pub(crate) fn prunable_topk(
    output_limit: Option<GroupLimit>,
    slots: &[AggregationSlot],
) -> Option<PrunableTopK> {
    // Experiment kill switch: disables the whole pruning apparatus so an A/B
    // can attribute a regression to it or to the surrounding changes.
    if std::env::var_os("PIVOT_DISABLE_TOPK_PRUNING").is_some() {
        return None;
    }
    let (slot, limit) = match output_limit? {
        GroupLimit::TopK { slot, limit, .. } => (slot, limit),
        GroupLimit::First { .. } => return None,
    };
    // A zero limit emits nothing; the pruning threshold needs a full heap.
    if limit == 0 {
        return None;
    }
    matches!(
        slots.get(slot)?.kind,
        AggregationKind::CountStar | AggregationKind::Count
    )
    .then_some(PrunableTopK { slot, limit })
}

/// Largest pushed limit whose threshold is seeded before the merge starts.
/// Every worker keeps that many candidates and the gather dedups `workers x
/// limit` of them, so past this the seed would cost more than the merge
/// wave it saves.
pub(crate) const MAX_SEEDED_LIMIT: usize = 1024;

/// One worker's partial of one group: the group's hash and persisted key
/// with the ordering aggregate's partial value, a lower bound on the
/// group's final value.
#[derive(Clone, Copy)]
pub struct SeedCandidate<P> {
    pub hash: u64,
    pub key: P,
    pub weight: u64,
}

// SAFETY: a persisted key is a plain `Copy` value or a reference into the
// address-stable shared arena, read by whichever worker gathers the
// candidates, as the tables holding the same keys are.
unsafe impl<P: PersistedKey> Send for SeedCandidate<P> {}

/// A worker's `limit` heaviest partials, kept as its table entries fold into
/// the bin totals.
///
/// They seed the merge's k-th best threshold: before any partition merges,
/// the pool's candidates are summed per group (one group's partials come
/// from disjoint rows, so their sum never exceeds its final value) and the
/// `limit`-th largest sum is a value `limit` distinct groups are known to
/// reach. Without the seed the threshold stays at zero until whole
/// partitions have merged, so the first wave of jobs, one per worker,
/// prunes nothing.
pub struct SeedCandidates<P: Copy> {
    /// Min-heap on weight: the root is the candidate a heavier one displaces.
    heap: BinaryHeap<Reverse<ByWeight<P>>>,
    limit: usize,
}

/// A candidate ordered by weight alone, so the heap needs no order on keys.
struct ByWeight<P>(SeedCandidate<P>);

impl<P> PartialEq for ByWeight<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0.weight == other.0.weight
    }
}

impl<P> Eq for ByWeight<P> {}

impl<P> PartialOrd for ByWeight<P> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for ByWeight<P> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.weight.cmp(&other.0.weight)
    }
}

impl<P: Copy> SeedCandidates<P> {
    /// Keeps the `limit` heaviest candidates offered; `limit` must be nonzero.
    pub fn new(limit: usize) -> Self {
        assert!(limit > 0, "a seed needs at least one candidate");
        Self {
            heap: BinaryHeap::with_capacity(limit),
            limit,
        }
    }

    /// Offers one partial; kept only while it is among the `limit` heaviest.
    #[inline(always)]
    pub fn offer(&mut self, hash: u64, key: P, weight: u64) {
        let candidate = Reverse(ByWeight(SeedCandidate { hash, key, weight }));
        if self.heap.len() < self.limit {
            self.heap.push(candidate);
        } else if weight
            > self
                .heap
                .peek()
                .expect("a full heap is nonempty")
                .0
                .0
                .weight
        {
            *self.heap.peek_mut().expect("a full heap is nonempty") = candidate;
        }
    }

    pub fn into_candidates(self) -> Vec<SeedCandidate<P>> {
        self.heap
            .into_iter()
            .map(|Reverse(ByWeight(candidate))| candidate)
            .collect()
    }
}

/// The seed threshold over the pool's candidates: the `limit`-th largest
/// per-group sum of partials, or 0 when fewer than `limit` distinct groups
/// were offered (no bound is below 0, so nothing is skipped).
pub(crate) fn seed_threshold<S: StoredKey>(
    mut candidates: Vec<SeedCandidate<S::Persisted>>,
    limit: usize,
    arena: &SharedArena,
) -> u64 {
    // One group's partials all carry its hash, so equal hashes run together
    // and only those runs need their keys compared.
    candidates.sort_unstable_by_key(|candidate| candidate.hash);
    let mut group_sums: Vec<u64> = Vec::with_capacity(candidates.len());
    let mut run_start = 0;
    while run_start < candidates.len() {
        let hash = candidates[run_start].hash;
        let mut run_end = run_start;
        while run_end < candidates.len() && candidates[run_end].hash == hash {
            run_end += 1;
        }
        // The run's distinct groups: the index of each group's first
        // candidate and the sum of its partials so far.
        let mut run_groups: Vec<(usize, u64)> = Vec::new();
        for index in run_start..run_end {
            let live_key = S::resolve_persisted(arena, candidates[index].key);
            match run_groups
                .iter_mut()
                .find(|(first, _)| live_key.eq_persisted(&candidates[*first].key))
            {
                Some((_, sum)) => *sum = sum.saturating_add(candidates[index].weight),
                None => run_groups.push((index, candidates[index].weight)),
            }
        }
        group_sums.extend(run_groups.into_iter().map(|(_, sum)| sum));
        run_start = run_end;
    }
    if group_sums.len() < limit {
        return 0;
    }
    let (_, kth_largest, _) = group_sums.select_nth_unstable_by(limit - 1, |a, b| b.cmp(a));
    *kth_largest
}

/// Converts a pushed ORDER BY sort key into a pruning weight, clamping into
/// `u64`. Order-preserving for the nonnegative values pruning is gated to.
pub trait TopKWeight {
    fn saturating_weight(self) -> u64;
}

impl TopKWeight for i64 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.max(0) as u64
    }
}

impl TopKWeight for i128 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.clamp(0, u64::MAX as i128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_slots_are_prunable_and_others_are_not() {
        use arrow_schema::DataType;
        let slots = vec![
            AggregationSlot::new(AggregationKind::Sum, 0, DataType::Int64),
            AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
        ];
        let topk = |slot| {
            Some(GroupLimit::TopK {
                slot,
                limit: 10,
                with_ties: false,
            })
        };

        assert_eq!(
            prunable_topk(topk(1), &slots),
            Some(PrunableTopK { slot: 1, limit: 10 })
        );
        assert_eq!(prunable_topk(topk(0), &slots), None);
        assert_eq!(
            prunable_topk(Some(GroupLimit::First { limit: 10 }), &slots),
            None
        );
        assert_eq!(prunable_topk(None, &slots), None);
    }

    #[test]
    fn seed_is_the_kth_largest_per_group_sum_of_partials() {
        use crate::memory::init_test_free_pool;
        use crate::operations::unary::group::keys::InlineKey;
        init_test_free_pool(8);
        let arena = SharedArena::new(8);
        let partial = |key: i64, weight: u64| SeedCandidate {
            hash: key as u64 * 7919,
            key,
            weight,
        };
        // Group 1 is held by two workers; the sum of its partials counts once.
        let candidates = vec![
            partial(1, 60),
            partial(2, 70),
            partial(3, 20),
            partial(1, 50),
        ];

        let second = seed_threshold::<InlineKey<i64>>(candidates.clone(), 2, &arena);
        let fourth = seed_threshold::<InlineKey<i64>>(candidates, 4, &arena);

        assert_eq!(second, 70, "sums are 110, 70, 20: the second largest");
        assert_eq!(fourth, 0, "only three distinct groups were offered");
    }

    #[test]
    fn seed_candidates_keep_the_heaviest_partials() {
        let mut candidates = SeedCandidates::new(2);

        for (key, weight) in [(1u32, 5u64), (2, 9), (3, 1), (4, 7)] {
            candidates.offer(key as u64, key, weight);
        }

        let mut kept: Vec<u64> = candidates
            .into_candidates()
            .into_iter()
            .map(|candidate| candidate.weight)
            .collect();
        kept.sort_unstable();
        assert_eq!(kept, vec![7, 9]);
    }
}
