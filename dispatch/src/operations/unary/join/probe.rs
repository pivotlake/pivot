//! The probe side of the hash join: for each probe row whose key matches a
//! build row, emit the probe row's columns followed by the matched build row's
//! columns (gathered from the concatenated build payload batch).
//!
//! Matches are collected as two parallel selection vectors — probe row index
//! and build payload row index — and materialized per output batch with the
//! arrow `take` kernel, so the hot loops touch only hashes, the directory, and
//! the key/row arenas.

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::filter_null_keys;
use crate::operations::unary::join::directory::{Directory, PtrBuffer, prefetch_ptr_l2};
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::operations::unary::join::{JoinArena, JoinTable};
use ahash::RandomState;
use arrow::compute::take;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt32Type};
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};
use std::cmp::min;
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const RING_SIZE: usize = 64;
const MASK: usize = RING_SIZE - 1;
const PREFETCH_LENGTH: usize = 63;

pub struct Probe {
    table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    use_probe_array: bool,
    allocator: SlabAllocator,
    /// Probe fields followed by build payload fields; built on first batch.
    output_schema: Option<Arc<Schema>>,
}

impl Probe {
    pub(crate) fn new(
        table: JoinTable,
        hash_state: RandomState,
        key_column: usize,
        use_probe_array: bool,
    ) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            use_probe_array,
            allocator: SlabAllocator::new(false),
            output_schema: None,
        }
    }
}

/// The per-batch probe state: scans `col` against the directory, collecting
/// matched (probe row, build payload row) pairs into the two selection-vector
/// builders and flushing full output batches through `sender`.
struct BatchProbe<'a, 'b, S: Sender<RecordBatch>> {
    keys: &'a JoinArena<u64>,
    rows: &'a JoinArena<u32>,
    col: &'b Int64Array,
    probe_batch: &'b RecordBatch,
    build_rows: &'a RecordBatch,
    output_schema: &'a Arc<Schema>,

    probe_sel: JoinPrimitiveBuilder<UInt32Type>,
    build_sel: JoinPrimitiveBuilder<UInt32Type>,
    output_idx: usize,

    sender: &'a mut S,
    allocator: &'a mut SlabAllocator,
}

impl<'a, 'b, S: Sender<RecordBatch>> BatchProbe<'a, 'b, S> {
    #[inline(never)]
    fn flush(&mut self) -> unary::Result<()> {
        if self.output_idx == 0 {
            return Ok(());
        }
        let probe_sel = mem::replace(
            &mut self.probe_sel,
            JoinPrimitiveBuilder::<UInt32Type>::new(self.allocator, RECORD_BATCH_SIZE),
        )
        .into_array(self.output_idx);
        let build_sel = mem::replace(
            &mut self.build_sel,
            JoinPrimitiveBuilder::<UInt32Type>::new(self.allocator, RECORD_BATCH_SIZE),
        )
        .into_array(self.output_idx);

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.output_schema.fields().len());
        for column in self.probe_batch.columns() {
            columns.push(take(column, &probe_sel, None)?);
        }
        for column in self.build_rows.columns() {
            columns.push(take(column, &build_sel, None)?);
        }
        let batch = RecordBatch::try_new(self.output_schema.clone(), columns)?;
        self.sender.send(batch)?;
        self.output_idx = 0;
        Ok(())
    }

    /// Record the match candidate at arena index `j` for probe row
    /// `probe_row`: write both selection vectors at the current cursor and
    /// advance it only when the full-width key matches (branchless on the
    /// match itself; the capacity flush branch is almost never taken).
    #[inline(always)]
    fn record_match(&mut self, j: usize, probe_row: usize, probe_key: u64) -> unary::Result<()> {
        let key = self.keys[j];
        self.probe_sel.write(self.output_idx, probe_row as u32);
        self.build_sel.write(self.output_idx, self.rows[j]);
        self.output_idx += (key == probe_key) as usize;
        if self.output_idx == RECORD_BATCH_SIZE {
            self.flush()?;
        }
        Ok(())
    }

    /// Scalar reference probe: hash, bloom-check and scan each probe row's
    /// directory slot in order, with light lookahead prefetching.
    fn run_scalar<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(
        mut self,
        directory: &Directory<B>,
        hash_state: &RandomState,
    ) -> unary::Result<()> {
        let col = self.col;
        let mut batch_hashes = vec![0u64; col.len()];
        for (row_idx, hash) in batch_hashes.iter_mut().enumerate() {
            *hash = hash_state.hash_one(unsafe { col.value_unchecked(row_idx) });
        }

        for row_idx in 0..col.len() {
            const DIRECTORY_PREFETCH_DISTANCE: usize = 16;
            const ARENA_PREFETCH_DISTANCE: usize = 8;

            if row_idx + DIRECTORY_PREFETCH_DISTANCE < col.len() {
                directory.prefetch_l2(batch_hashes[row_idx + DIRECTORY_PREFETCH_DISTANCE]);
            }

            if row_idx + ARENA_PREFETCH_DISTANCE < col.len() {
                let future_hash = batch_hashes[row_idx + ARENA_PREFETCH_DISTANCE];
                let future_slot = directory.slot_for(future_hash);
                let future_start = directory.end_ptr(future_slot as isize);
                prefetch_ptr_l2(self.keys.ptr_at_index(future_start) as *const u8);
            }

            let hash = batch_hashes[row_idx];
            if !directory.matches_bloom(hash) {
                continue;
            }

            let slot = directory.slot_for(hash);
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            let probe_key = unsafe { col.value_unchecked(row_idx) } as u64;

            for j in start..end {
                self.record_match(j, row_idx, probe_key)?;
            }
        }

        self.flush()
    }
}

/// The prefetch-pipelined probe: hashes ahead, bloom-filters into a ring of
/// matched directory slots, prefetches their arena ranges, then drains matches
/// a window behind — keeping many independent loads in flight.
struct ProbeArray<
    'a,
    'b,
    B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    S: Sender<RecordBatch>,
> {
    row_idx: usize,
    hash_state: RandomState,
    directory: &'a Directory<B>,

    hashes: [u64; RING_SIZE],

    // If we made this two separate variables, e.g. matched_slots and next_matched_slots, swapping
    // became an issue. The reason for this is that because it is of constant size, it was being
    // unrolled, causing multiple pointer swaps to occur. This may seem like a small issue, and it's
    // certainly not a big one, but keeping instructions to a minimum is important for the ROB.
    matched_slots: [[(usize, usize); PREFETCH_LENGTH]; 2],
    matched_size: [usize; 2],
    matched_idx: usize,

    out: BatchProbe<'a, 'b, S>,
}

impl<'a, 'b, B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer, S: Sender<RecordBatch>>
    ProbeArray<'a, 'b, B, S>
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
            let probe = Directory::<B>::compute_tag(hash) as u64;

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
            }

            let (slot, idx) = current_matched_slots[i];
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            let probe_key = unsafe { out.col.value_unchecked(idx) } as u64;

            for j in start..end {
                out.record_match(j, idx, probe_key)?;
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub fn only_prefetch(&mut self) {
        let nxt = self.matched_idx;

        let next_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &self.matched_slots[nxt];
        for i in 0..self.matched_size[nxt] {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            let (slot, _) = next_matched_slots[i];
            let start = self.directory.end_ptr(slot as isize);
            let end = self.directory.end_ptr((slot + 1) as isize);
            prefetch_ptr_l2(self.out.keys.ptr_at_index(start) as *const u8);
            prefetch_ptr_l2(self.out.keys.ptr_at_index(end) as *const u8);
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

            return self.out.flush();
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

        self.out.flush()
    }
}

/// Run the pipelined probe over one batch: monomorphized per directory
/// backing so the inner loops carry no per-element dispatch.
fn run_probe_array<'a, 'b, B, S>(
    directory: &'a Directory<B>,
    hash_state: &RandomState,
    out: BatchProbe<'a, 'b, S>,
) -> unary::Result<()>
where
    B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    S: Sender<RecordBatch>,
{
    ProbeArray {
        row_idx: 0,
        hash_state: hash_state.clone(),
        directory,
        hashes: [0; RING_SIZE],
        matched_slots: [[(0, 0); PREFETCH_LENGTH]; 2],
        matched_size: [0; 2],
        matched_idx: 0,
        out,
    }
    .run()
}

impl Unary<RecordBatch, RecordBatch> for Probe {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()> {
        // The build dataflow's collect can return while its final stolen
        // partition job is still running on another worker; the gate flips
        // when the join table is fully populated.
        while !self.table.gate.load(Ordering::Acquire) {
            std::thread::yield_now();
        }

        let build_rows = unsafe { &*self.table.build_rows.get() };
        let Some(build_rows) = build_rows else {
            // Empty build side: an inner join emits nothing.
            return Ok(());
        };

        let batch = filter_null_keys(batch, self.key_column);
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let col = batch.column(self.key_column).as_primitive::<Int64Type>();

        let output_schema = self
            .output_schema
            .get_or_insert_with(|| {
                let fields: Vec<Field> = batch
                    .schema()
                    .fields()
                    .iter()
                    .chain(build_rows.schema().fields())
                    .map(|field| field.as_ref().clone())
                    .collect();
                Arc::new(Schema::new(fields))
            })
            .clone();

        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };
        let probe_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let build_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let out = BatchProbe {
            keys,
            rows,
            col,
            probe_batch: &batch,
            build_rows,
            output_schema: &output_schema,
            probe_sel,
            build_sel,
            output_idx: 0,
            sender,
            allocator: &mut self.allocator,
        };

        let directory = unsafe { &*self.table.directory.get() };
        if !self.use_probe_array {
            return out.run_scalar(directory, &self.hash_state);
        }
        run_probe_array(directory, &self.hash_state, out)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        Ok(true)
    }
}
