//! The probe side of the hash join: for each probe row whose key matches a
//! build row, emit the output columns declared in
//! [`JoinOutputColumns`] — the listed probe columns
//! followed by the listed build columns (gathered from the concatenated build
//! payload batch).
//!
//! Matches are collected as two parallel index slices — probe row and build
//! payload row — and appended into a [`BatchAccumulator`] per side, so the
//! hot loops touch only hashes, the directory, and the key/row arenas, and a
//! selective join still emits full-size batches.

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::BatchAccumulator;
use crate::memory::{MultiSlabBuffer, SlabAllocator};
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::filter_null_keys;
use crate::operations::unary::join::directory::{JoinDirectory, prefetch_ptr_l2};
use crate::operations::unary::join::{JoinOutputColumns, JoinTable};
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch};
use arrow_schema::{Field, Schema, SchemaRef};
use std::cmp::min;
use std::hash::Hash;
use std::sync::Arc;

const PROBE_BATCH_SIZE: usize = 2048;
const RING_SIZE: usize = 64;
const MASK: usize = RING_SIZE - 1;
const PREFETCH_LENGTH: usize = 63;

pub struct Probe<T: ArrowPrimitiveType<Native: Hash + Eq>> {
    table: JoinTable<T::Native>,
    hash_state: RandomState,
    key_column: usize,
    output_columns: Arc<JoinOutputColumns>,
    allocator: SlabAllocator,

    /// Matched (probe row, build payload row) pairs of the batch being
    /// probed, drained into the accumulators when full or at batch end.
    probe_indices: Vec<u32>,
    build_indices: Vec<u32>,
    /// The two output sides, created on the first batch (a schema only
    /// exists then). Both receive the same row count per drain, so they fill
    /// and emit in lockstep.
    sides: Option<OutputSides>,
}

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> Probe<T> {
    pub(crate) fn new(
        table: JoinTable<T::Native>,
        hash_state: RandomState,
        key_column: usize,
        output_columns: Arc<JoinOutputColumns>,
    ) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            output_columns,
            allocator: SlabAllocator::new(false),
            probe_indices: vec![0; RECORD_BATCH_SIZE],
            build_indices: vec![0; RECORD_BATCH_SIZE],
            sides: None,
        }
    }

    fn probe_window<S: Sender<RecordBatch>>(
        &mut self,
        window: &RecordBatch,
        window_offset: usize,
        probe_source: &RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()> {
        let col = window.column(self.key_column).as_primitive::<T>();
        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };
        let out = ProbeMatchCollector {
            keys,
            rows,
            col,
            window_offset,
            probe_source,
            sides: self.sides.as_mut().expect("sides are built before probing"),
            probe_indices: &mut self.probe_indices,
            build_indices: &mut self.build_indices,
            matched: 0,
            sender,
            allocator: &mut self.allocator,
        };

        ProbeArray {
            row_idx: 0,
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

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> Unary<RecordBatch, RecordBatch> for Probe<T> {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        let Some(build_rows) = build_rows else {
            // Empty build side: an inner join emits nothing.
            return Ok(());
        };

        let batch = filter_null_keys(batch, self.key_column);
        if batch.num_rows() == 0 {
            return Ok(());
        }

        if self.sides.is_none() {
            let probe_schema = batch.schema();
            let build_schema = build_rows.schema();
            let fields: Vec<Field> = self
                .output_columns
                .probe
                .iter()
                .map(|&i| probe_schema.field(i).clone())
                .chain(
                    self.output_columns
                        .build
                        .iter()
                        .map(|&i| build_schema.field(i).clone()),
                )
                .collect();
            let build_source = build_rows.project(&self.output_columns.build)?;
            let probe_fields = Arc::new(Schema::new(
                fields[..self.output_columns.probe.len()].to_vec(),
            ));
            let build_fields = Arc::new(Schema::new(
                fields[self.output_columns.probe.len()..].to_vec(),
            ));
            self.sides = Some(OutputSides {
                output_schema: Arc::new(Schema::new(fields)),
                probe: BatchAccumulator::new(probe_fields, &mut self.allocator),
                build: BatchAccumulator::new(build_fields, &mut self.allocator),
                build_source,
            });
        }
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

    fn finish<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if let Some(sides) = &mut self.sides
            && !sides.probe.is_empty()
        {
            sides.emit(&mut self.allocator, sender)?;
        }
        Ok(true)
    }
}

/// The join's output state: the selected probe columns and selected build
/// payload columns accumulate separately (their rows come from different
/// sources), and every emitted batch splices them under one schema.
struct OutputSides {
    /// The probe columns listed in the output, then the build columns.
    output_schema: SchemaRef,
    probe: BatchAccumulator,
    build: BatchAccumulator,
    /// The build payload restricted to the listed build columns.
    build_source: RecordBatch,
}

impl OutputSides {
    /// Emit one combined batch from the two sides' accumulated rows.
    fn emit<S: Sender<RecordBatch>>(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut S,
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
}

/// Collects matches produced while probing one input window.
///
/// For each candidate build-arena row, it verifies the full key, buffers the
/// corresponding `(probe row, build payload row)` indices, and periodically
/// appends those rows to the probe and build output accumulators. Full output
/// batches are emitted as the accumulators fill.
struct ProbeMatchCollector<'a, 'b, T: ArrowPrimitiveType<Native: Hash + Eq>, S: Sender<RecordBatch>>
{
    keys: &'a MultiSlabBuffer<T::Native>,
    rows: &'a MultiSlabBuffer<u32>,
    col: &'b PrimitiveArray<T>,
    /// Row offset of `col`'s window within the probed batch, added to every
    /// recorded probe index so the indices address `probe_source`.
    window_offset: usize,
    /// The probed batch restricted to the listed probe columns.
    probe_source: &'b RecordBatch,
    sides: &'a mut OutputSides,

    probe_indices: &'a mut [u32],
    build_indices: &'a mut [u32],
    matched: usize,

    sender: &'a mut S,
    allocator: &'a mut SlabAllocator,
}

impl<'a, 'b, T: ArrowPrimitiveType<Native: Hash + Eq>, S: Sender<RecordBatch>>
    ProbeMatchCollector<'a, 'b, T, S>
{
    /// Append the collected pairs to the output sides, emitting if a full
    /// batch accumulated.
    #[inline(never)]
    fn drain(&mut self) -> unary::Result<()> {
        if self.matched == 0 {
            return Ok(());
        }
        let OutputSides {
            probe,
            build,
            build_source,
            ..
        } = &mut *self.sides;
        probe.append(self.probe_source, &self.probe_indices[..self.matched]);
        build.append(build_source, &self.build_indices[..self.matched]);
        self.matched = 0;
        if self.sides.probe.should_emit() {
            self.sides.emit(self.allocator, self.sender)?;
        }
        Ok(())
    }

    /// Record the match candidate at arena index `j` for probe row
    /// `probe_row`: write both index slices at the current cursor and
    /// advance it only when the full-width key matches (branchless on the
    /// match itself; the capacity drain branch is almost never taken).
    #[inline(always)]
    fn record_match(
        &mut self,
        j: usize,
        probe_row: usize,
        probe_key: T::Native,
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
        self.matched += (key == probe_key) as usize;
        if self.matched == RECORD_BATCH_SIZE {
            self.drain()?;
        }
        Ok(())
    }
}

/// The prefetch-pipelined probe: hashes ahead, bloom-filters into a ring of
/// matched directory slots, prefetches their arena ranges, then drains matches
/// a window behind — keeping many independent loads in flight.
struct ProbeArray<'a, 'b, T: ArrowPrimitiveType<Native: Hash + Eq>, S: Sender<RecordBatch>> {
    row_idx: usize,
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

    out: ProbeMatchCollector<'a, 'b, T, S>,
}

impl<'a, 'b, T: ArrowPrimitiveType<Native: Hash + Eq>, S: Sender<RecordBatch>>
    ProbeArray<'a, 'b, T, S>
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
                let value = unsafe { self.out.col.value_unchecked(row_idx + PREFETCH_LENGTH) };
                let hash_offset = (row_idx + PREFETCH_LENGTH) & MASK;
                self.hashes[hash_offset] = self.hash_state.hash_one(value);
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
            let probe_key = unsafe { out.col.value_unchecked(idx) };

            for j in start..end {
                out.record_match(j, idx, probe_key)?;
            }
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
        for i in 0..min(PREFETCH_LENGTH, self.out.col.len()) {
            self.hashes[i] = self
                .hash_state
                .hash_one(unsafe { self.out.col.value_unchecked(i) });
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

        let len = self.out.col.len();

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
