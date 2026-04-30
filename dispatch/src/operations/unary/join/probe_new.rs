use std::cell::UnsafeCell;
use std::cmp::min;
use std::hint::black_box;
use std::mem;
use std::ops::{Index, IndexMut, Sub};
use std::ptr::null;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};
use ahash::RandomState;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayAccessor, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use libc::send;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::memory::{MultiSlabBuffer, SlabAllocator, BUFFER_SIZE, RING};
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::Unary;
use crate::operations::unary::join::directory::{prefetch_ptr, prefetch_ptr_l2, Directory, JoinDirectory, PtrBuffer, PTR_SHIFT};
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::Value;
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::perf_stat::{perf_disable, perf_enable};
use crate::RECORD_BATCH_SIZE;
use crate::worker::WORKER_IDX;

const RING_SIZE: usize = 64;
const MASK: usize = RING_SIZE - 1;
const PREFETCH_LENGTH: usize = 32;

static PROBE_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("probe_idx", DataType::Int64, false),
        Field::new("build_key", DataType::Int64, false),
        // Field::new("build_payload", DataType::Int64, false),
    ]))
});

pub struct Probe {
    table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    ptrs: Box<[*const u64; RECORD_BATCH_SIZE]>,
    entries: Box<[u64; RECORD_BATCH_SIZE]>,

    matched_indexes: Box<[usize; RECORD_BATCH_SIZE]>,
    allocator: SlabAllocator,
    total: usize,
    shared_total: Arc<AtomicUsize>,
    hashes: Box<[u64; RING_SIZE]>,
    matched_slots: Box<[(usize, usize); RING_SIZE]>,
    next_matched_slots: Box<[(usize, usize); RING_SIZE]>,

    lineitem_keys: JoinPrimitiveBuilder::<Int64Type>,
    order_keys: JoinPrimitiveBuilder::<Int64Type>
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize, shared_total: Arc<AtomicUsize>) -> Self {
        let mut allocator = SlabAllocator::new(false);
        Self {
            table,
            hash_state,
            key_column,
            hashes: Box::new([0; RING_SIZE]),

            matched_slots: Box::new([(0, 0); RING_SIZE]),

            ptrs: Box::new([null(); RECORD_BATCH_SIZE]),
            entries: Box::new([0; RECORD_BATCH_SIZE]),

            matched_indexes: Box::new([0; RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
            total: 0,
            shared_total,
            next_matched_slots: Box::new([(0, 0); RING_SIZE]),
            lineitem_keys: JoinPrimitiveBuilder::<Int64Type>::new(&mut allocator, RECORD_BATCH_SIZE),
            order_keys: JoinPrimitiveBuilder::<Int64Type>::new(&mut allocator, RECORD_BATCH_SIZE),
        }
    }

    pub fn run<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, directory: &Directory<B>, col: &Int64Array) {
        self.lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        self.order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);

        let mut output = 0;


        let arena = unsafe { &*self.table.arena.get() };

        let mut hashes: [u64; RING_SIZE] = [0; RING_SIZE];

        // let mut arena_ptrs: [(usize, usize); RING_SIZE] = [(0, 0); RING_SIZE];

        let mut arena_ptrs: [[(usize, usize); RING_SIZE]; 2] = [[(0, 0); RING_SIZE]; 2];
        let [buf0, buf1] = &mut arena_ptrs;

        // Prepopulate hashes for first RING_SIZE elements
        for i in 0..min(PREFETCH_LENGTH, col.len()) {
            hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            prefetch_ptr_l2(directory.ptr_for_slot((hashes[i] >> directory.shift) as usize) as *const u8);
        }
        let mut arena_outer_idx = 0usize;

        let mut idx = 0;
        while idx + PREFETCH_LENGTH < col.len().saturating_sub(PREFETCH_LENGTH) {
            let (next_arena_ptrs, current_arena_ptrs) = if arena_outer_idx == 0 {
                (&mut *buf1, &*buf0)
            } else {
                (&mut *buf0, &*buf1)
            };

            arena_outer_idx ^= 1;
            let mut arena_size = 0;
            for _ in 0..PREFETCH_LENGTH {
                // hash
                let value = unsafe { col.value_unchecked(idx + PREFETCH_LENGTH) };
                let hash_offset = (idx + PREFETCH_LENGTH) & MASK;
                hashes[hash_offset] = self.hash_state.hash_one(value);
                let dir_slot = (hashes[hash_offset] >> directory.shift) as usize;
                prefetch_ptr_l2(directory.ptr_for_slot(dir_slot) as *const u8);

                // bloom check, arena
                let touch_offset = idx & MASK;
                let hash = hashes[touch_offset];

                let slot = directory.slot_for(hash);
                let stored = directory.entries()[slot];
                let probe = Directory::<B>::compute_tag(hash) as u64;

                if (stored & probe) == probe {
                    prefetch_ptr_l2(directory.ptr_for_slot(slot + 1) as *const u8);
                    next_arena_ptrs[arena_size & MASK] = (slot, idx);
                    arena_size += 1;
                }

                idx += 1;
            }

            // TODO: there's an unneessary iteration at first run

            for i in 0..arena_size {
                let (slot, _) = next_arena_ptrs[i];
                let start = directory.end_ptr(slot as isize);
                let end = directory.end_ptr((slot + 1) as isize);
                prefetch_ptr_l2(arena.ptr_at_index(start) as *const u8);
                prefetch_ptr_l2(arena.ptr_at_index(end) as *const u8);

                let (slot, idx) = current_arena_ptrs[i];
                let start = directory.end_ptr(slot as isize);
                let end = directory.end_ptr((slot + 1) as isize);
                let probe_key = unsafe { col.value_unchecked(idx) } as u32;

                for j in start..end {
                    let entry: Value = arena[j];
                    self.lineitem_keys.write(output, entry as i64);
                    self.order_keys.write(output, entry as i64);
                    output += (entry == probe_key) as usize;
                }
            }
        }

        let current = &mut arena_ptrs[arena_outer_idx];

        for i in 0..PREFETCH_LENGTH {
            let (slot, idx) = current[i];
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);

            for j in start..end {
                // let entry: Value = arena[j];
                let probe_key = unsafe { col.value_unchecked(idx) } as u32;
                // if entry == probe_key {
                //     self.total += 1;
                // }

                let entry: Value = arena[j];
                self.lineitem_keys.write(output, entry as i64);
                self.order_keys.write(output, entry as i64);
                output += (entry == probe_key) as usize;
            }
        }

        let end_offset = col.len().saturating_sub(PREFETCH_LENGTH);

        while idx < end_offset {
            // hash
            let value = unsafe { col.value_unchecked(idx + PREFETCH_LENGTH) };
            let hash_offset = (idx + PREFETCH_LENGTH) & MASK;
            hashes[hash_offset] = self.hash_state.hash_one(value);
            prefetch_ptr_l2(directory.ptr_for_slot((hashes[hash_offset] >> directory.shift) as usize) as *const u8);

            // bloom check, arena
            let touch_offset = idx & MASK;
            let hash = hashes[touch_offset];

            if directory.matches_bloom(hash) {
                // let slot = directory.slot_for(hash) as isize;
                // arena_ptrs[arena_size & MASK] = slot;
                // arena_size += 1;
                // self.total += 1;
            }
            idx += 1;
        }

        for j in 0..min(PREFETCH_LENGTH, col.len()) {
            let hash_offset = (end_offset + j) & MASK;
            let hash = hashes[hash_offset];
            if directory.matches_bloom(hash) {
                self.total += 1;
            }
        }

        if output > 28383828 {
            let lineitem = mem::replace(
                &mut self.order_keys,
                JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE),
            );
            let order = mem::replace(
                &mut self.lineitem_keys,
                JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE),
            );
            RecordBatch::try_new(
                PROBE_SCHEMA.clone(),
                vec![
                    lineitem.into_array(output),
                    order.into_array(output),
                ],
            );
        }
    }
}


impl Unary<RecordBatch, RecordBatch> for Probe {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()> {
        perf_enable();
        let col = batch
            .column(self.key_column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let arena = unsafe { &*self.table.arena.get() };

        let join_dir = unsafe { &*self.table.directory.get() };
        match join_dir {
            JoinDirectory::Contiguous(dir) => {
                let lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
                let order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
                ProbeArray {
                    row_idx: 0,
                    hash_state: self.hash_state.clone(),
                    col,
                    directory: dir,
                    arena,
                    hashes: [0; RING_SIZE],
                    matched_slots: [[(0, 0); PREFETCH_LENGTH]; 2],
                    matched_size: [0; 2],
                    matched_idx: 0,
                    lineitem_builder: lineitem_keys,
                    order_builder: order_keys,
                    output_idx: 0,
                    shared_total: self.shared_total.clone(),
                    total: 0,
                    sender,
                    allocator: &mut self.allocator,
                }.run()
                // self.run(dir, col)
            },
            JoinDirectory::NonContiguous(_dir) => {
                panic!("oh no")
            },
        };

        Ok(())

    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_disable();
        self.shared_total.fetch_add(mem::take(&mut self.total), Ordering::Relaxed);
        println!("Shared total: {:?}", self.shared_total.load(Ordering::Relaxed));
        Ok(true)
    }
}


struct ProbeArray<'a, 'b, B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer, S: Sender<RecordBatch>> {
    row_idx: usize,
    hash_state: RandomState,
    col: &'b Int64Array,
    directory: &'a Directory<B>,
    arena: &'a MultiSlabBuffer<Value>,

    hashes: [u64; RING_SIZE],

    // If we made this two separate variables, e.g. matched_slots and next_matched_slots, swapping
    // became an issue. The reason for this is that because it is of constant size, it was being
    // unrolled, causing multiple pointer swaps to occur. This may seem like a small issue, and it's
    // certainly not a big one, but keeping instructions to a minimum is important for the ROB.
    matched_slots: [[(usize, usize); PREFETCH_LENGTH]; 2],
    matched_size: [usize; 2],
    matched_idx: usize,

    lineitem_builder: JoinPrimitiveBuilder::<Int64Type>,
    order_builder: JoinPrimitiveBuilder::<Int64Type>,
    output_idx: usize,

    shared_total: Arc<AtomicUsize>,
    total: usize,

    sender: &'a mut S,
    allocator: &'a mut SlabAllocator,
}

impl<'a,'b, B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer, S: Sender<RecordBatch>> ProbeArray<'a, 'b, B, S> {
    #[inline(never)]
    fn flush(&mut self) {
        if self.output_idx == 0 {
            return;
        }
        let lineitem = mem::replace(
            &mut self.lineitem_builder,
            JoinPrimitiveBuilder::<Int64Type>::new(self.allocator, RECORD_BATCH_SIZE),
        );
        let order = mem::replace(
            &mut self.order_builder,
            JoinPrimitiveBuilder::<Int64Type>::new(self.allocator, RECORD_BATCH_SIZE),
        );
        let batch = RecordBatch::try_new(
            PROBE_SCHEMA.clone(),
            vec![lineitem.into_array(self.output_idx), order.into_array(self.output_idx)],
        )
        .unwrap();
        self.sender.send(batch).unwrap();
        self.output_idx = 0;
    }

    #[inline(always)]
    pub fn generate_matched_slots<const HASH: bool>(&mut self, length: usize) {
        let next = self.matched_idx;
        let next_matched_slots: &mut [(usize, usize); PREFETCH_LENGTH] = &mut self.matched_slots[next];
        let mut size = self.matched_size[next];
        let mut row_idx = self.row_idx;
        let shift = self.directory.shift;
        for _ in 0..length {
            if HASH {
                // hash
                let value = unsafe { self.col.value_unchecked(row_idx + PREFETCH_LENGTH) };
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
        self.matched_size[next] = size;
        self.row_idx = row_idx;
    }


    #[inline(always)]
    pub fn build_output<const PREFETCH_NEXT: bool>(&mut self) {
        let cur = self.matched_idx ^ 1;
        let nxt = self.matched_idx;

        let arena = self.arena;
        let mut output_idx = self.output_idx;
        let current_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &self.matched_slots[cur];
        let next_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &self.matched_slots[nxt];
        for i in 0..self.matched_size[cur] {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            if PREFETCH_NEXT {
                let (slot, _) = next_matched_slots[i];
                let start = self.directory.end_ptr(slot as isize);
                let end = self.directory.end_ptr((slot + 1) as isize);
                prefetch_ptr_l2(arena.ptr_at_index(start) as *const u8);
                prefetch_ptr_l2(arena.ptr_at_index(end) as *const u8);
            }

            let (slot, idx) = current_matched_slots[i];
            let start = self.directory.end_ptr(slot as isize);
            let end = self.directory.end_ptr((slot + 1) as isize);
            let probe_key = unsafe { self.col.value_unchecked(idx) } as u32;

            for j in start..end {
                let entry: Value = arena[j];
                self.lineitem_builder.write(output_idx, entry as i64);
                self.order_builder.write(output_idx, entry as i64);
                output_idx += (entry == probe_key) as usize;
            }
        }
        self.output_idx = output_idx;
    }

    #[inline(always)]
    pub fn bootstrap_initial_hashes(&mut self) {
        for i in 0..min(PREFETCH_LENGTH, self.col.len()) {
            self.hashes[i] = self.hash_state.hash_one(unsafe { self.col.value_unchecked(i) });
            prefetch_ptr_l2(self.directory.ptr_for_slot((self.hashes[i] >> self.directory.shift) as usize) as *const u8);
        }
    }


    #[inline(always)]
    fn swap_matched_slots(&mut self) {
        self.matched_idx ^= 1;
        self.matched_size[self.matched_idx] = 0;
    }

    #[inline(always)]
    pub fn run(mut self) {
        self.bootstrap_initial_hashes();
        self.generate_matched_slots::<true>(min(PREFETCH_LENGTH, self.col.len()));
        self.swap_matched_slots();

        while self.row_idx + PREFETCH_LENGTH < self.col.len().saturating_sub(PREFETCH_LENGTH) {
            self.generate_matched_slots::<true>(PREFETCH_LENGTH);
            // TODO: there's an unneessary iteration at first run
            self.build_output::<true>();
            self.swap_matched_slots();

        }
        self.build_output::<true>();

        self.generate_matched_slots::<true>((self.col.len() - self.row_idx).saturating_sub(PREFETCH_LENGTH));
        self.swap_matched_slots();
        self.build_output::<true>();

        self.generate_matched_slots::<false>(self.col.len() - self.row_idx);
        self.swap_matched_slots();
        self.build_output::<false>();
        self.flush();
    }
}