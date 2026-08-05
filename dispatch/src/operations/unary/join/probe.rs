//! The probe side of the hash join: for each probe row whose key matches a
//! build row, emit the output columns declared in
//! [`JoinOutputColumns`] — the listed probe columns
//! followed by the listed build columns, gathered from the stored build row
//! batches by row id.
//!
//! Matches are collected as two parallel index slices — probe row and build
//! row id — and appended into a [`BatchAccumulator`] per side, so the
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
//! A semi join collects probe rows alone: one per probe row whose arena range
//! holds its key, found by stopping at the first one that does. Only the probe
//! accumulator ever fills, since such a join carries no build columns.

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::{BatchAccumulator, ChunkedColumn};
use crate::memory::{MultiSlabBuffer, SlabAllocator};
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::filter_null_keys;
use crate::operations::unary::join::build_rows;
use crate::operations::unary::join::directory::{JoinDirectory, prefetch_ptr_l2};
use crate::operations::unary::join::keys::JoinKey;
use crate::operations::unary::join::{JoinOutputColumns, JoinTable, UnmatchedScan};
use ahash::RandomState;
use arrow_array::{RecordBatch, new_null_array};
use arrow_schema::{Field, Schema, SchemaRef};
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const PROBE_BATCH_SIZE: usize = 2048;
const RING_SIZE: usize = 64;
const MASK: usize = RING_SIZE - 1;
const PREFETCH_LENGTH: usize = 63;

pub struct Probe<K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool> {
    table: JoinTable<K::Stored>,
    hash_state: RandomState,
    key_columns: Vec<usize>,
    build_key_columns: Vec<usize>,
    output_columns: Arc<JoinOutputColumns>,
    allocator: SlabAllocator,

    /// Matched (probe row, build row id) pairs of the batch being
    /// probed, drained into the accumulators when full or at batch end. A semi
    /// join fills the probe indices alone; it has no build columns to gather.
    probe_indices: Vec<u32>,
    build_indices: Vec<u32>,
    outputter: Outputter,
    /// The stored build rows bound for gathering output columns. `None` only
    /// until the build side publishes, which every call that reaches it has
    /// already waited out.
    build_gather: Option<BuildGather>,
    /// Shared progress of the unmatched pass. Unused by an inner join.
    unmatched: Arc<UnmatchedScan>,
    /// Whether this worker has entered `finish` — the point it stops probing
    /// and leaves the barrier the unmatched pass waits on.
    probing_done: bool,
}

/// The stored build rows bound for reading output columns back: the batches
/// restricted to the listed build columns (for the unmatched pass's per-batch
/// scan), and the same columns prepared once as [`ChunkedColumn`]s for
/// gathering matched rows by row id.
struct BuildGather {
    build_row_batches: Vec<RecordBatch>,
    columns: Vec<ChunkedColumn>,
}

impl<K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool> Probe<K, BUILD_OUTER, SEMI> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: JoinTable<K::Stored>,
        hash_state: RandomState,
        key_columns: Vec<usize>,
        build_key_columns: Vec<usize>,
        output_columns: Arc<JoinOutputColumns>,
        probe_fields: Arc<Vec<Field>>,
        build_fields: Arc<Vec<Field>>,
        unmatched: Arc<UnmatchedScan>,
    ) -> Self {
        let mut allocator = SlabAllocator::new(false);
        let outputter = Outputter::new(&probe_fields, &build_fields, &mut allocator);
        Self {
            table,
            hash_state,
            key_columns,
            build_key_columns,
            output_columns,
            allocator,
            probe_indices: vec![0; RECORD_BATCH_SIZE],
            build_indices: vec![0; RECORD_BATCH_SIZE],
            outputter,
            build_gather: None,
            unmatched,
            probing_done: false,
        }
    }

    /// Bind the published build rows for output gathering, once.
    fn ensure_build_gather_bound(
        &mut self,
        build_rows: &build_rows::BuildRowBatches,
    ) -> unary::Result<()> {
        if self.build_gather.is_some() {
            return Ok(());
        }
        let (build_row_batches, columns) = build_rows.gather_source(&self.output_columns.build)?;
        self.build_gather = Some(BuildGather {
            build_row_batches,
            columns,
        });
        Ok(())
    }

    /// Claim one build row batch, scan its flags, and append its unmatched
    /// rows to the build accumulator, emitting once a full batch has
    /// gathered. Returns whether every batch has been claimed.
    fn send_out_next_unmatched_build_rows(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<bool> {
        let build_row_batch_idx = self.unmatched.cursor.fetch_add(1, Ordering::Relaxed);
        let gather = self
            .build_gather
            .as_ref()
            .expect("the build rows are bound before scanning");
        if build_row_batch_idx >= gather.build_row_batches.len() {
            if !self.outputter.build.is_empty() {
                self.outputter
                    .emit_unmatched_build_rows(&mut self.allocator, sender)?;
            }
            return Ok(true);
        }

        let batch = &gather.build_row_batches[build_row_batch_idx];
        let first_row_id = build_rows::first_row_id(build_row_batch_idx);
        let flags = unsafe { &*self.table.matched.get() };
        let mut found = 0;
        for row in 0..batch.num_rows() {
            // Branchless, as in the match collector: write the row and keep it
            // only if its flag is still clear. A stored batch holds at most
            // one output batch's worth of rows, so a pass never overfills the
            // accumulator.
            self.build_indices[found] = row as u32;
            found += (unsafe { *flags.ptr_at_index(first_row_id + row) } == 0) as usize;
        }
        if found > 0 {
            self.outputter.build.append_batch_by_indices(
                batch,
                &self.build_indices[..found],
                &mut self.allocator,
            );
        }
        if self.outputter.build.has_full_batch() {
            self.outputter
                .emit_unmatched_build_rows(&mut self.allocator, sender)?;
        }
        Ok(false)
    }

    fn probe_window(
        &mut self,
        window: &RecordBatch,
        window_offset: usize,
        probe_source: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        let reader = K::make_reader(window, &self.key_columns, &self.hash_state);
        let verifier = K::make_verifier(build_rows.batches(), &self.build_key_columns);
        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };
        let out = ProbeMatchCollector::<K, BUILD_OUTER, SEMI> {
            keys,
            rows,
            matched_flags: unsafe { &*self.table.matched.get() },
            reader,
            verifier,
            window_offset,
            probe_source,
            outputter: &mut self.outputter,
            build_columns: &self
                .build_gather
                .as_ref()
                .expect("the build rows are bound before probing")
                .columns,
            probe_indices: &mut self.probe_indices,
            build_indices: &mut self.build_indices,
            matched: 0,
            sender,
            allocator: &mut self.allocator,
        };

        ProbeArray {
            row_idx: 0,
            len: window.num_rows(),
            hash_state: self.hash_state.clone(),
            directory: unsafe { &*self.table.directory.get() },
            hashes: [0; RING_SIZE],
            matched_slots: [[(0, 0); PREFETCH_LENGTH]; 2],
            matched_size: [0; 2],
            matched_idx: 0,
            out,
        }
        .run()
    }
}

impl<K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool> Unary<RecordBatch, RecordBatch>
    for Probe<K, BUILD_OUTER, SEMI>
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        if build_rows.is_empty() {
            // Empty build side: nothing matches, and an outer build has no rows
            // to carry through unmatched either.
            return Ok(());
        }

        let batch = filter_null_keys(batch, &self.key_columns);
        if batch.num_rows() == 0 {
            return Ok(());
        }

        self.ensure_build_gather_bound(build_rows)?;
        let probe_source = batch.project(&self.output_columns.probe)?;

        let total = batch.num_rows();
        let mut start = 0;
        while start < total {
            let len = (total - start).min(PROBE_BATCH_SIZE);
            let window = batch.slice(start, len);
            self.probe_window(&window, start, &probe_source, sender)?;
            start += len;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if !self.probing_done {
            self.probing_done = true;
            if !self.outputter.probe.is_empty() {
                self.outputter.emit(&mut self.allocator, sender)?;
            }
            if BUILD_OUTER {
                // Release this worker's flag writes and leave the barrier. It
                // is sound to leave it here and nowhere earlier: `consume` is
                // never called again once `finish` has run.
                self.unmatched.probes_live.fetch_sub(1, Ordering::Release);
            }
        }
        if !BUILD_OUTER {
            return Ok(true);
        }

        // A peer still probing can yet flag a row this worker would otherwise
        // read as unmatched, so the scan waits for all of them to arrive here.
        if self.unmatched.probes_live.load(Ordering::Acquire) != 0 {
            return Ok(false);
        }
        let build_rows = unsafe { &*self.table.build_rows.get() };
        if build_rows.is_empty() {
            return Ok(true);
        }
        self.ensure_build_gather_bound(build_rows)?;
        self.send_out_next_unmatched_build_rows(sender)
    }
}

/// Emits the join's output: the selected probe columns and selected build
/// columns accumulate separately (their rows come from different sources),
/// and every emitted batch splices them under one schema. Both sides receive
/// the same row count per drain, so they fill and emit in lockstep — except
/// in a semi join, where the build side stays empty and contributes only its
/// (zero) columns.
struct Outputter {
    /// The probe columns listed in the output, then the build columns.
    output_schema: SchemaRef,
    probe: BatchAccumulator,
    build: BatchAccumulator,
}

impl Outputter {
    fn new(probe_fields: &[Field], build_fields: &[Field], allocator: &mut SlabAllocator) -> Self {
        let fields: Vec<Field> = probe_fields.iter().chain(build_fields).cloned().collect();
        Self {
            output_schema: Arc::new(Schema::new(fields)),
            probe: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(probe_fields.to_vec())),
                allocator,
            ),
            build: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(build_fields.to_vec())),
                allocator,
            ),
        }
    }

    /// Emit one combined batch from the two sides' accumulated rows.
    fn emit(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let probe_part = self.probe.take_batch(allocator)?;
        let build_part = self.build.take_batch(allocator)?;
        let columns = probe_part
            .columns()
            .iter()
            .chain(build_part.columns())
            .cloned()
            .collect();
        let options =
            arrow_array::RecordBatchOptions::new().with_row_count(Some(probe_part.num_rows()));
        sender.send(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &options,
        )?)?;
        Ok(())
    }

    /// Emit one batch of build rows that matched nothing: the accumulated build
    /// columns, and an all-null column of the right shape for each probe column.
    fn emit_unmatched_build_rows(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let build_part = self.build.take_batch(allocator)?;
        let rows = build_part.num_rows();
        let columns = self
            .probe
            .schema()
            .fields()
            .iter()
            .map(|field| new_null_array(field.data_type(), rows))
            .chain(build_part.columns().iter().cloned())
            .collect();
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(rows));
        sender.send(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &options,
        )?)?;
        Ok(())
    }
}

/// Collects matches produced while probing one input window.
///
/// For each candidate build-arena row, it verifies the full key, buffers the
/// corresponding `(probe row, build row id)` indices, and periodically
/// appends those rows to the probe and build output accumulators. Full output
/// batches are emitted as the accumulators fill. A semi join buffers the probe
/// row alone, once per probe row that has a match at all.
struct ProbeMatchCollector<'a, 'b, K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool> {
    keys: &'a MultiSlabBuffer<K::Stored>,
    rows: &'a MultiSlabBuffer<u32>,
    /// The build rows' matched flags, written on drain by an outer build.
    matched_flags: &'a MultiSlabBuffer<u8>,
    reader: K::Reader<'b>,
    verifier: K::Verifier<'a>,
    /// Row offset of the reader's window within the probed batch, added to
    /// every recorded probe index so the indices address `probe_source`.
    window_offset: usize,
    /// The probed batch restricted to the listed probe columns.
    probe_source: &'b RecordBatch,
    outputter: &'a mut Outputter,
    /// The stored build rows' listed columns, prepared for gathering by row
    /// id.
    build_columns: &'a [ChunkedColumn],

    probe_indices: &'a mut [u32],
    build_indices: &'a mut [u32],
    matched: usize,

    sender: &'a mut dyn Sender<RecordBatch>,
    allocator: &'a mut SlabAllocator,
}

impl<'a, 'b, K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool>
    ProbeMatchCollector<'a, 'b, K, BUILD_OUTER, SEMI>
{
    /// Append the collected pairs to the outputter, emitting if a full batch
    /// accumulated.
    #[inline(never)]
    fn drain(&mut self) -> unary::Result<()> {
        if self.matched == 0 {
            return Ok(());
        }
        if BUILD_OUTER {
            // These indices are the verified matches, so flagging them here
            // keeps the match loop itself untouched. Relaxed because the flags
            // are only read after a barrier that orders them, and atomic only
            // because peer workers write the same bytes concurrently.
            for &row in &self.build_indices[..self.matched] {
                let flag = self.matched_flags.ptr_at_index(row as usize);
                unsafe { AtomicU8::from_ptr(flag) }.store(1, Ordering::Relaxed);
            }
        }
        self.outputter.probe.append_batch_by_indices(
            self.probe_source,
            &self.probe_indices[..self.matched],
            self.allocator,
        );
        // A semi join has no build columns and buffered no build rows: its
        // build side only ever contributes its (empty) column list on emit.
        if !SEMI {
            self.outputter.build.append_chunked_by_ids(
                self.build_columns,
                build_rows::row_id_shift(),
                &self.build_indices[..self.matched],
                self.allocator,
            );
        }
        self.matched = 0;
        if self.outputter.probe.has_full_batch() {
            self.outputter.emit(self.allocator, self.sender)?;
        }
        Ok(())
    }

    /// Record what probe row `probe_row` matches among the arena range
    /// `start..end`, its slot's candidates: every one of them, or - in a semi
    /// join, where a probe row reaches the output at most once - the first.
    #[inline(always)]
    fn record_matches(
        &mut self,
        start: usize,
        end: usize,
        probe_row: usize,
        probe_key: K::Stored,
    ) -> unary::Result<()> {
        if SEMI {
            for j in start..end {
                if self.keys[j] == probe_key
                    && K::verify(&self.reader, &self.verifier, probe_row, self.rows[j])
                {
                    return self.record_probe_row(probe_row);
                }
            }
            return Ok(());
        }
        for j in start..end {
            self.record_match(j, probe_row, probe_key)?;
        }
        Ok(())
    }

    /// Buffer `probe_row` as a matched probe row, without a build row to pair
    /// it with: a semi join's whole output.
    #[inline(always)]
    fn record_probe_row(&mut self, probe_row: usize) -> unary::Result<()> {
        debug_assert!(self.matched < self.probe_indices.len());
        // SAFETY: as in `record_match` - the drain below resets the cursor the
        // moment it reaches the slice's length.
        unsafe {
            *self.probe_indices.get_unchecked_mut(self.matched) =
                (self.window_offset + probe_row) as u32;
        }
        self.matched += 1;
        if self.matched == RECORD_BATCH_SIZE {
            self.drain()?;
        }
        Ok(())
    }

    /// Record the match candidate at arena index `j` for probe row
    /// `probe_row`: write both index slices at the current cursor and advance
    /// it only when the stored key matches and the shape's verify confirms it
    /// (branchless on the match itself for a shape whose verify is constant
    /// `true`; the capacity drain branch is almost never taken).
    #[inline(always)]
    fn record_match(
        &mut self,
        j: usize,
        probe_row: usize,
        probe_key: K::Stored,
    ) -> unary::Result<()> {
        let key = self.keys[j];
        debug_assert!(self.matched < self.probe_indices.len());
        // SAFETY: `matched` stays below the slices' length: they are
        // RECORD_BATCH_SIZE long and the drain below resets the cursor the
        // moment it reaches that.
        unsafe {
            *self.probe_indices.get_unchecked_mut(self.matched) =
                (self.window_offset + probe_row) as u32;
            *self.build_indices.get_unchecked_mut(self.matched) = self.rows[j];
        }
        let key_matches =
            key == probe_key && K::verify(&self.reader, &self.verifier, probe_row, self.rows[j]);
        self.matched += key_matches as usize;
        if self.matched == RECORD_BATCH_SIZE {
            self.drain()?;
        }
        Ok(())
    }
}

/// The prefetch-pipelined probe: hashes ahead, bloom-filters into a ring of
/// matched directory slots, prefetches their arena ranges, then drains matches
/// a window behind — keeping many independent loads in flight.
struct ProbeArray<'a, 'b, K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool> {
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

    out: ProbeMatchCollector<'a, 'b, K, BUILD_OUTER, SEMI>,
}

impl<'a, 'b, K: JoinKey, const BUILD_OUTER: bool, const SEMI: bool>
    ProbeArray<'a, 'b, K, BUILD_OUTER, SEMI>
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
                    &self.out.reader,
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
            out,
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
                prefetch_ptr_l2(out.keys.ptr_at_index(start) as *const u8);
                prefetch_ptr_l2(out.keys.ptr_at_index(end) as *const u8);
                prefetch_ptr_l2(out.rows.ptr_at_index(start) as *const u8);
            }

            let (slot, idx) = current_matched_slots[i];
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            let probe_key = K::read_stored(&out.reader, idx);

            out.record_matches(start, end, idx, probe_key)?;
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
            prefetch_ptr_l2(self.out.keys.ptr_at_index(start) as *const u8);
            prefetch_ptr_l2(self.out.keys.ptr_at_index(end) as *const u8);
            prefetch_ptr_l2(self.out.rows.ptr_at_index(start) as *const u8);
        }
    }

    #[inline(always)]
    pub fn bootstrap_initial_hashes(&mut self) {
        for i in 0..min(PREFETCH_LENGTH, self.len) {
            self.hashes[i] = K::hash_row(&self.out.reader, i, &self.hash_state);
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

            return self.out.drain();
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

        self.out.drain()
    }
}
