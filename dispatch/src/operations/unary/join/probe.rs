//! The probe side of the hash join: for each probe row whose key matches a
//! build row, emit the output columns declared in
//! [`JoinSpec::probe_output_indices`] — the listed probe columns
//! followed by the listed build columns, gathered from the stored build row
//! batches by row id.
//!
//! Matches are collected as two parallel index slices — probe row and build
//! row id — and appended into a
//! [`BatchAccumulator`](crate::arrays::accumulator::BatchAccumulator) per side, so the
//! hot loops touch only hashes, the directory, and the key/row arenas, and a
//! selective join still emits full-size batches.
//!
//! When the join is outer on the build side, the build row of each collected
//! pair is also flagged as matched, in the drain rather than the match loop:
//! the indices there are already the verified matches, so it costs one byte
//! store per output row and nothing at all per probed row. The rows still
//! unflagged once every worker has stopped probing are emitted with their probe
//! columns null by
//! [`send_out_next_unmatched_build_rows`](Probe::send_out_next_unmatched_build_rows).
//!
//! A semi join without a residual collects probe rows alone: one per probe row
//! whose arena range holds its key, found by stopping at the first one that
//! does. Only the probe accumulator ever fills, since such a join carries no
//! build columns.
//!
//! When the join is outer on the probe side, the probe rows nothing matched
//! are emitted with their build columns null once their batch is fully
//! probed. Almost every miss is known the moment the probe passes the row: a
//! row the bloom check rejects has no candidates at all, and a row whose
//! arena range verifies nothing has only false ones, so those two sites
//! record the miss on the spot. A residual predicate cannot overturn either
//! verdict (it only ever rejects pairs), but it does create one further kind
//! of miss the inline sites cannot see: a row whose verified pairs are all
//! rejected at drain time. When a residual exists, every row is therefore
//! also flagged as settled once its fate is known (miss recorded, or a pair
//! kept), and whatever the batch's last drain leaves unsettled pads too.
//! Unlike the build-side pass, none of this needs cross-worker state: a
//! probe row's matches all surface while its own batch is probed, so each
//! worker settles its own batches.
//!
//! An anti join emits only the unmatched rows of its preserved side: a
//! probe-side anti join runs the probe-side miss tracking, a build-side anti
//! join the build-side flag array and unmatched scan, and either way the
//! drain stops once the collected pairs have settled fates, so a matched
//! pair never reaches the output.
//!
//! A join with a residual predicate ([`JoinSpec::residual_filters`]) weighs
//! the collected matches once more in the drain: the matched rows of both
//! sides are gathered into one combined batch, the predicate evaluated over
//! it, and the rejected matches discarded before any flagging or output. A
//! semi join with a residual therefore collects every candidate match like an
//! inner join (the first key match may be rejected) and instead drops its
//! duplicate probe rows here.

use crate::RECORD_BATCH_SIZE;
use crate::memory::MultiSlabBuffer;
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::{filter_null_keys, split_null_keys};
use crate::operations::unary::join::build_rows;
use crate::operations::unary::join::directory::{JoinDirectory, prefetch_ptr_l2};
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::match_outputter::ProbeMatchOutputter;
use crate::operations::unary::join::residual_filter::ResidualFilter;
use crate::operations::unary::join::{JoinKind, JoinSpec, JoinTable, UnmatchedScan};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::Ordering;

const RING_SIZE: usize = 64;
const MASK: usize = RING_SIZE - 1;
const PREFETCH_LENGTH: usize = 63;

pub struct Probe<
    K: JoinKey,
    const OUTER_JOIN_BUILD_SIDE: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> {
    table: JoinTable<K::Stored>,
    hash_state: RandomState,
    spec: Arc<JoinSpec>,
    /// Accumulates matches and outputs them
    match_outputter: ProbeMatchOutputter,
    /// Shared progress of the unmatched pass. Unused by an inner join.
    unmatched: Arc<UnmatchedScan>,
    /// Whether this worker has entered `finish` and will no longer receive new record batches
    probing_done: bool,
}

impl<
    K: JoinKey,
    const OUTER_JOIN_BUILD_SIDE: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
>
    Probe<
        K,
        OUTER_JOIN_BUILD_SIDE,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: JoinTable<K::Stored>,
        hash_state: RandomState,
        spec: Arc<JoinSpec>,
        unmatched: Arc<UnmatchedScan>,
    ) -> Self {
        // Stopping at the first matching key is only safe without a residual
        // filter, which could reject that match. `join_dispatch` therefore
        // disables this shortcut for a filtered semi join.
        debug_assert!(
            !STOP_AFTER_FIRST_MATCH || spec.residual_filters.is_none(),
            "the first-match probe path cannot evaluate residual filters"
        );

        // Full outer joins are not supported, so unmatched rows may be tracked
        // on the build side or the probe side, but not both.
        debug_assert!(
            !OUTER_JOIN_BUILD_SIDE || !TRACK_UNMATCHED_PROBE_ROWS,
            "cannot track unmatched rows on both sides"
        );

        if DISCARD_MATCHED_PAIRS {
            // An anti join emits the unmatched rows from exactly one side. It
            // must inspect every candidate match, so it cannot use the semi
            // join's first-match shortcut.
            debug_assert!(
                !STOP_AFTER_FIRST_MATCH,
                "an anti join must inspect every candidate match"
            );
            debug_assert_ne!(
                OUTER_JOIN_BUILD_SIDE, TRACK_UNMATCHED_PROBE_ROWS,
                "an anti join must track unmatched rows on exactly one side"
            );
        }

        if MARK {
            // A mark join stops at the first match and marks every unmatched
            // probe row. Residual filters are not supported for mark joins.
            debug_assert!(
                STOP_AFTER_FIRST_MATCH,
                "a mark join must stop at the first match"
            );
            debug_assert!(
                TRACK_UNMATCHED_PROBE_ROWS,
                "a mark join must track unmatched probe rows"
            );
            debug_assert!(
                !DISCARD_MATCHED_PAIRS,
                "a mark join does not discard matched pairs in the drain"
            );
            debug_assert!(
                !OUTER_JOIN_BUILD_SIDE,
                "a mark join does not track unmatched build rows"
            );
            debug_assert!(
                spec.residual_filters.is_none(),
                "a mark join does not support residual filters"
            );
        }
        let residual_filters = spec
            .residual_filters
            .as_ref()
            .map(|filters| ResidualFilter::new(filters, matches!(spec.kind, JoinKind::ProbeSemi)));
        let output = ProbeMatchOutputter::new(
            &spec.probe_fields,
            &spec.build_fields,
            table.build_rows.clone(),
            residual_filters,
            // Probe outer and anti joins buffer unmatched rows for output.
            // A mark join only retains their indices to build its mark column.
            TRACK_UNMATCHED_PROBE_ROWS && !MARK,
            MARK,
        );
        Self {
            table,
            hash_state,
            spec,
            match_outputter: output,
            unmatched,
            probing_done: false,
        }
    }

    /// Claim one build row batch, scan its flags, and append its unmatched
    /// rows to the build accumulator, emitting once a full batch has
    /// gathered. Returns whether every batch has been claimed.
    fn send_out_next_unmatched_build_rows(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<bool> {
        let build_row_batch_idx = self.unmatched.cursor.fetch_add(1, Ordering::Relaxed);
        let build_rows = unsafe { &*self.table.build_rows.get() };
        if build_row_batch_idx >= build_rows.output_batches.len() {
            self.match_outputter.emit_unmatched_build_rows(sender)?;
            return Ok(true);
        }

        let batch = &build_rows.output_batches[build_row_batch_idx];
        let first_row_id = build_rows::first_row_id(build_row_batch_idx);
        self.match_outputter
            .append_unmatched_build_rows(batch, first_row_id, sender)?;
        Ok(false)
    }

    fn probe_batch(
        &mut self,
        batch: &RecordBatch,
        probe_source: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let len = batch.num_rows();
        let build_rows = unsafe { &*self.table.build_rows.get() };
        let reader = K::make_reader(batch, &self.spec.probe_key_indices, &self.hash_state);
        let verifier = K::make_verifier(&build_rows.batches, &self.spec.build_key_indices);
        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };
        if let Some(residual_filters) = &mut self.match_outputter.residual_filters {
            residual_filters.begin_probe_batch();
        }
        if TRACK_UNMATCHED_PROBE_ROWS {
            self.match_outputter.begin_unmatched_probe_tracking(len);
        }
        let probe_window = ProbeWindow::<
            K,
            OUTER_JOIN_BUILD_SIDE,
            STOP_AFTER_FIRST_MATCH,
            TRACK_UNMATCHED_PROBE_ROWS,
            DISCARD_MATCHED_PAIRS,
            MARK,
        > {
            keys,
            rows,
            reader,
            verifier,
            probe_batch: batch,
            probe_source,
            output: &mut self.match_outputter,
            sender,
        };

        ProbeArray {
            row_idx: 0,
            len,
            hash_state: self.hash_state.clone(),
            directory: unsafe { &*self.table.directory.get() },
            hashes: [0; RING_SIZE],
            matched_slots: [[(0, 0); PREFETCH_LENGTH]; 2],
            matched_size: [0; 2],
            matched_idx: 0,
            window: probe_window,
        }
        .run()?;

        // The batch is fully probed and drained, so settle its output: a mark
        // join emits the whole batch with its markers, a probe-side outer or
        // anti join finalizes the batch's unmatched rows.
        if MARK {
            let build_saw_null_key = unsafe { *self.table.build_saw_null_key.get() };
            self.match_outputter.emit_marked_batch_from_misses(
                probe_source,
                build_saw_null_key,
                sender,
            )?;
        } else if TRACK_UNMATCHED_PROBE_ROWS {
            self.match_outputter
                .finish_probe_outer_batch(probe_source, sender)?;
        }
        Ok(())
    }
}

impl<
    K: JoinKey,
    const OUTER_JOIN_BUILD_SIDE: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> Unary<RecordBatch, RecordBatch>
    for Probe<
        K,
        OUTER_JOIN_BUILD_SIDE,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        if build_rows.is_empty() {
            if MARK {
                // SQL's `IN` over an empty set is FALSE for every row, even a
                // null-keyed one.
                let probe_source = batch.project(&self.spec.probe_output_indices)?;
                return self.match_outputter.emit_marked_batch_with_constant(
                    &probe_source,
                    Some(false),
                    sender,
                );
            }
            if !TRACK_UNMATCHED_PROBE_ROWS {
                // Empty build side, nothing to match
                return Ok(());
            }
            // An empty build side matches nothing, so every probe row pads.
            let probe_source = batch.project(&self.spec.probe_output_indices)?;
            return self
                .match_outputter
                .append_probe_rows_padded(&probe_source, sender);
        }

        // A null-keyed probe row can never match. A probe-side outer or anti
        // join still owes it an output row, so it pads directly instead of
        // being dropped with the rest of the filter; a mark join owes it a
        // NULL marker (its key compares as unknown against a non-empty build
        // side).
        // The split (and its Option local) lives entirely inside the
        // const-gated arm, so every other kind compiles this method exactly
        // as if unmatched probe rows were not being tracked.
        let batch = if TRACK_UNMATCHED_PROBE_ROWS {
            let (batch, null_keyed) = split_null_keys(batch, &self.spec.probe_key_indices);
            if let Some(null_keyed) = null_keyed {
                let probe_source = null_keyed.project(&self.spec.probe_output_indices)?;
                if MARK {
                    self.match_outputter.emit_marked_batch_with_constant(
                        &probe_source,
                        None,
                        sender,
                    )?;
                } else {
                    self.match_outputter
                        .append_probe_rows_padded(&probe_source, sender)?;
                }
            }
            batch
        } else {
            filter_null_keys(batch, &self.spec.probe_key_indices)
        };
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let probe_source = batch.project(&self.spec.probe_output_indices)?;
        self.probe_batch(&batch, &probe_source, sender)
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if !self.probing_done {
            self.probing_done = true;
            if self.match_outputter.has_buffered_matches() {
                self.match_outputter.emit(sender)?;
            }
            // A mark join settled every batch as it was probed and buffers
            // nothing here.
            if TRACK_UNMATCHED_PROBE_ROWS && !MARK {
                self.match_outputter.emit_unmatched_probe_rows(sender)?;
            }
            if OUTER_JOIN_BUILD_SIDE {
                self.unmatched
                    .probes_finished
                    .fetch_sub(1, Ordering::Release);
            }
        }
        if !OUTER_JOIN_BUILD_SIDE {
            return Ok(true);
        }

        // A peer still probing can yet flag a row this worker would otherwise
        // read as unmatched, so the scan waits for all of them to arrive here.
        if self.unmatched.probes_finished.load(Ordering::Acquire) != 0 {
            return Ok(false);
        }
        let build_rows = unsafe { &*self.table.build_rows.get() };
        if build_rows.is_empty() {
            return Ok(true);
        }
        self.send_out_next_unmatched_build_rows(sender)
    }
}

/// The per-batch inputs borrowed while the persistent collector records and
/// emits matches.
struct ProbeWindow<
    'a,
    'b,
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> {
    keys: &'a MultiSlabBuffer<K::Stored>,
    rows: &'a MultiSlabBuffer<u32>,
    reader: K::Reader<'b>,
    verifier: K::Verifier<'a>,
    /// The full probed batch, which the residual predicate's probe columns
    /// gather from; `probe_source` is its output projection.
    probe_batch: &'b RecordBatch,
    probe_source: &'b RecordBatch,
    output: &'a mut ProbeMatchOutputter,
    sender: &'a mut dyn Sender<RecordBatch>,
}

impl<
    'a,
    'b,
    K: JoinKey,
    const OUTER_JOIN_BUILD_SIDE: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
>
    ProbeWindow<
        'a,
        'b,
        K,
        OUTER_JOIN_BUILD_SIDE,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >
{
    #[inline(never)]
    fn drain(&mut self) -> unary::Result<()> {
        self.output
            .drain::<
                OUTER_JOIN_BUILD_SIDE,
                STOP_AFTER_FIRST_MATCH,
                TRACK_UNMATCHED_PROBE_ROWS,
                DISCARD_MATCHED_PAIRS,
            >(
                self.probe_source,
                self.probe_batch,
                self.sender,
            )
    }

    /// Record what probe row `probe_row` matches among its slot's candidates.
    /// The regular path records every match; the first-match path returns as
    /// soon as it finds one.
    #[inline(always)]
    fn record_matches(
        &mut self,
        keys_start: usize,
        keys_end: usize,
        probe_row: usize,
        probe_key: K::Stored,
    ) -> unary::Result<()> {
        let mut row_matched_any = false;
        for key_index in keys_start..keys_end {
            let output_idx = self.output.matched;
            debug_assert!(output_idx < self.output.probe_indices.len());
            let key_matches = self.keys[key_index] == probe_key
                && K::verify(
                    &self.reader,
                    &self.verifier,
                    probe_row,
                    self.rows[key_index],
                );
            // Const-gated so the other kinds' candidate loop compiles exactly
            // as it did before this tracking existed.
            if TRACK_UNMATCHED_PROBE_ROWS {
                row_matched_any |= key_matches;
            }

            if STOP_AFTER_FIRST_MATCH {
                // This path emits at most one result per probe row, so return
                // on the first match. It records no build index because every
                // join kind using this shortcut emits only probe-side data.
                if key_matches {
                    // A mark join's match contributes nothing beyond not
                    // being a miss: the row's marker is TRUE because the
                    // miss list never received it.
                    if MARK {
                        return Ok(());
                    }
                    unsafe {
                        *self.output.probe_indices.get_unchecked_mut(output_idx) = probe_row as u32;
                    }
                    self.output.matched = output_idx + 1;
                    if self.output.matched == RECORD_BATCH_SIZE {
                        self.drain()?;
                    }
                    return Ok(());
                }
            } else {
                // In a regular join, we want to record both probe row and build row. We don't need an
                // if statement for matching since we can just increment the cursor by *whether there
                // was a match* which removes the need for branching here.
                unsafe {
                    *self.output.probe_indices.get_unchecked_mut(output_idx) = probe_row as u32;
                    *self.output.build_indices.get_unchecked_mut(output_idx) = self.rows[key_index];
                }
                self.output.matched = output_idx + key_matches as usize;
                if self.output.matched == RECORD_BATCH_SIZE {
                    self.drain()?;
                }
            }
        }
        // The range held only false candidates (the bloom tag or the hash
        // slot collided), so this row misses: a residual only ever rejects
        // pairs, and there is no pair here for it to judge.
        if TRACK_UNMATCHED_PROBE_ROWS && !row_matched_any {
            self.output.record_missed_probe_row(probe_row);
        }
        Ok(())
    }
}

/// The prefetch-pipelined probe: hashes ahead, bloom-filters into a ring of
/// matched directory slots, prefetches their arena ranges, then drains matches
/// a window behind — keeping many independent loads in flight.
struct ProbeArray<
    'a,
    'b,
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
> {
    row_idx: usize,
    /// The probed window's row count.
    len: usize,
    hash_state: RandomState,
    directory: &'a JoinDirectory,

    hashes: [u64; RING_SIZE],

    // If we made this two separate variables, e.g. matched_slots and next_matched_slots, swapping
    // became an issue. The reason for this is that because it is of constant size, it was being
    // unrolled, causing multiple pointer swaps to occur. This may seem like a small issue, and it's
    // certainly not a big one, but keeping instructions to a minimum is important for the ROB.
    matched_slots: [[(usize, usize); PREFETCH_LENGTH]; 2],
    matched_size: [usize; 2],
    matched_idx: usize,

    window: ProbeWindow<
        'a,
        'b,
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >,
}

impl<
    'a,
    'b,
    K: JoinKey,
    const BUILD_OUTER: bool,
    const STOP_AFTER_FIRST_MATCH: bool,
    const TRACK_UNMATCHED_PROBE_ROWS: bool,
    const DISCARD_MATCHED_PAIRS: bool,
    const MARK: bool,
>
    ProbeArray<
        'a,
        'b,
        K,
        BUILD_OUTER,
        STOP_AFTER_FIRST_MATCH,
        TRACK_UNMATCHED_PROBE_ROWS,
        DISCARD_MATCHED_PAIRS,
        MARK,
    >
{
    #[inline(always)]
    pub fn generate_matched_slots<const HASH: bool>(&mut self, length: usize) {
        let next_matched_slots: &mut [(usize, usize); PREFETCH_LENGTH] =
            &mut self.matched_slots[self.matched_idx];
        let mut size = self.matched_size[self.matched_idx];
        let mut row_idx = self.row_idx;
        let shift = self.directory.shift;
        for _ in 0..length {
            if HASH {
                // hash
                let hash_offset = (row_idx + PREFETCH_LENGTH) & MASK;
                self.hashes[hash_offset] = K::hash_row(
                    &self.window.reader,
                    row_idx + PREFETCH_LENGTH,
                    &self.hash_state,
                );
                let dir_slot = (self.hashes[hash_offset] >> shift) as usize;
                // Prefetch this hash from the directory; we're going to need it soon when we run bloom
                // on it
                prefetch_ptr_l2(self.directory.ptr_for_slot(dir_slot) as *const u8);
            }

            // bloom check, arena
            let bloom_offset = row_idx & MASK;
            let hash = self.hashes[bloom_offset];

            let slot = (hash >> shift) as usize;
            let stored = unsafe { *self.directory.ptr_for_slot(slot) };
            let probe = JoinDirectory::compute_tag(hash) as u64;

            if (stored & probe) == probe {
                // Given that we matched, let's prefetch the slot after ours. Cache lines are 64 bytes,
                // so there's a one in eight chance this will be relevant
                prefetch_ptr_l2(self.directory.ptr_for_slot(slot + 1) as *const u8);
                next_matched_slots[size & MASK] = (slot, row_idx);
                size += 1;
            } else if TRACK_UNMATCHED_PROBE_ROWS {
                // A row the bloom check rejects has no candidates in the
                // directory, so joins that track unmatched probe rows can
                // record the miss immediately. The else arm costs the loop no
                // extra comparison.
                self.window.output.record_missed_probe_row(row_idx);
            }

            row_idx += 1;
        }
        self.matched_size[self.matched_idx] = size;
        self.row_idx = row_idx;
    }

    #[inline(always)]
    pub fn build_output<const PREFETCH_NEXT: bool>(&mut self) -> unary::Result<()> {
        // Field-level destructuring keeps the slot-ring borrows disjoint from
        // the &mut the match recorder needs.
        let Self {
            matched_slots,
            matched_size,
            matched_idx,
            directory,
            window,
            ..
        } = self;
        let cur = *matched_idx ^ 1;
        let nxt = *matched_idx;

        let current_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &matched_slots[cur];
        let next_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &matched_slots[nxt];
        for i in 0..matched_size[cur] {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            if PREFETCH_NEXT {
                let (slot, _) = next_matched_slots[i];
                let start = directory.end_ptr(slot as isize);
                let end = directory.end_ptr((slot + 1) as isize);
                prefetch_ptr_l2(window.keys.ptr_at_index(start) as *const u8);
                prefetch_ptr_l2(window.keys.ptr_at_index(end) as *const u8);
                prefetch_ptr_l2(window.rows.ptr_at_index(start) as *const u8);
            }

            let (slot, idx) = current_matched_slots[i];
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            let probe_key = K::read_stored(&window.reader, idx);

            window.record_matches(start, end, idx, probe_key)?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn only_prefetch(&mut self) {
        let nxt = self.matched_idx;

        let next_matched_slots: &[(usize, usize)] =
            &self.matched_slots[nxt][..self.matched_size[nxt]];
        for &(slot, _) in next_matched_slots {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            let start = self.directory.end_ptr(slot as isize);
            let end = self.directory.end_ptr((slot + 1) as isize);
            prefetch_ptr_l2(self.window.keys.ptr_at_index(start) as *const u8);
            prefetch_ptr_l2(self.window.keys.ptr_at_index(end) as *const u8);
            prefetch_ptr_l2(self.window.rows.ptr_at_index(start) as *const u8);
        }
    }

    #[inline(always)]
    pub fn bootstrap_initial_hashes(&mut self) {
        for i in 0..min(PREFETCH_LENGTH, self.len) {
            self.hashes[i] = K::hash_row(&self.window.reader, i, &self.hash_state);
            prefetch_ptr_l2(
                self.directory
                    .ptr_for_slot((self.hashes[i] >> self.directory.shift) as usize)
                    as *const u8,
            );
        }
    }

    #[inline(always)]
    fn swap_matched_slots(&mut self) {
        self.matched_idx ^= 1;
        self.matched_size[self.matched_idx] = 0;
    }

    #[inline(always)]
    pub fn run(mut self) -> unary::Result<()> {
        self.bootstrap_initial_hashes();

        let len = self.len;

        // No row has a `row + PREFETCH_LENGTH`, so never use HASH=true.
        if len <= PREFETCH_LENGTH {
            if len != 0 {
                self.generate_matched_slots::<false>(len);
                self.only_prefetch();
                self.swap_matched_slots();
                self.build_output::<false>()?;
            }

            return self.window.drain();
        }

        // From here on, len > PREFETCH_LENGTH, so this cannot underflow.
        let hash_end = len - PREFETCH_LENGTH;

        // Prime the pipeline: generate first matched buffer and prefetch its arena ranges.
        let first = min(PREFETCH_LENGTH, hash_end);
        self.generate_matched_slots::<true>(first);
        self.only_prefetch();
        self.swap_matched_slots();

        while self.row_idx < hash_end {
            let n = min(PREFETCH_LENGTH, hash_end - self.row_idx);

            self.generate_matched_slots::<true>(n);
            self.build_output::<true>()?;
            self.swap_matched_slots();
        }

        // Tail rows already have hashes; do not hash-ahead.
        self.generate_matched_slots::<false>(len - self.row_idx);
        self.build_output::<true>()?;
        self.swap_matched_slots();

        // Drain final generated buffer.
        self.build_output::<false>()?;

        self.window.drain()
    }
}
