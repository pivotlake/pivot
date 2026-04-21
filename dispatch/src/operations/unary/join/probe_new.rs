use std::cmp::min;
use std::hint::black_box;
use std::mem;
use std::ops::{Index, IndexMut, Sub};
use std::ptr::null;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use ahash::RandomState;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayAccessor, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use libc::send;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::memory::{SlabAllocator, BUFFER_SIZE, RING};
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
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    ptrs: Box<[*const u64; RECORD_BATCH_SIZE]>,
    entries: Box<[u64; RECORD_BATCH_SIZE]>,

    matched_indexes: Box<[usize; RECORD_BATCH_SIZE]>,
    allocator: SlabAllocator,
    total: usize,
    shared_total: Arc<AtomicUsize>
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize, shared_total: Arc<AtomicUsize>) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            hashes: Box::new([0; RECORD_BATCH_SIZE]),
            ptrs: Box::new([null(); RECORD_BATCH_SIZE]),
            entries: Box::new([0; RECORD_BATCH_SIZE]),

            matched_indexes: Box::new([0; RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
            total: 0,
            shared_total,
        }
    }

    pub fn run<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, directory: &Directory<B>, col: &Int64Array) {
        const RING_SIZE: usize = 128;
        const MASK: usize = RING_SIZE - 1;
        const PREFETCH_LENGTH: usize = RING_SIZE - 1;

        let arena = unsafe { &*self.table.arena.get() };

        let mut hashes: [u64; RING_SIZE] = [0; RING_SIZE];
        let mut arena_ptrs: [(usize, usize); RING_SIZE] = [(0, 0); RING_SIZE];

        // Prepopulate hashes for first RING_SIZE elements
        for i in 0..min(PREFETCH_LENGTH, col.len()) {
            hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            prefetch_ptr_l2(directory.ptr_for_slot((hashes[i] >> directory.shift) as usize) as *const u8);
        }

        let mut idx = 0;
        while idx + PREFETCH_LENGTH < col.len().saturating_sub(PREFETCH_LENGTH) {
            let mut arena_size = 0;
            for _ in 0..PREFETCH_LENGTH {
                // hash
                let value = unsafe { col.value_unchecked(idx + PREFETCH_LENGTH) };
                let hash_offset = (idx + PREFETCH_LENGTH) & MASK;
                hashes[hash_offset] = self.hash_state.hash_one(value);
                prefetch_ptr_l2(directory.ptr_for_slot((hashes[hash_offset] >> directory.shift) as usize) as *const u8);

                // bloom check, arena
                let touch_offset = idx & MASK;
                let hash = hashes[touch_offset];

                let slot = directory.slot_for(hash);
                let stored = directory.entries()[slot];
                let probe = Directory::<B>::compute_tag(hash) as u64;
                if (stored & probe) == probe {
                    prefetch_ptr_l2(directory.ptr_for_slot(slot + 1) as *const u8);
                    arena_ptrs[arena_size & MASK] = (slot, idx);
                    arena_size += 1;
                    let start_ptr =  arena.ptr_at_index(directory.end_ptr(slot as isize)) as *const u8;
                    prefetch_ptr_l2(start_ptr);
                    self.total += 1;
                }
                idx += 1;
            }

            for i in 0..arena_size {
                let (slot, idx) = arena_ptrs[i];
                let start = directory.end_ptr(slot as isize);
                let end = directory.end_ptr((slot + 1) as isize);
                // let ptr = arena.ptr_at_index(start);
                // black_box(((start, end, unsafe {*ptr})));
                for j in start..end {
                    let entry: Value = arena[j];
                    let probe_key = unsafe { col.value_unchecked(idx) } as u32;
                    if entry == probe_key {
                        self.total += 1;
                    }
                }
            }
        }

        let end_offset = col.len().saturating_sub(PREFETCH_LENGTH);

        let mut arena_size = 0;
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
                self.total += 1;
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

        let join_dir = unsafe { &*self.table.directory.get() };
        match join_dir {
            JoinDirectory::Contiguous(dir) => {
                self.run(dir, col);
                Ok(())
            },
            JoinDirectory::NonContiguous(dir) => {
                self.run(dir, col);
                Ok(())
            },
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_disable();
        self.shared_total.fetch_add(mem::take(&mut self.total), Ordering::Relaxed);
        // println!("Shared total: {:?}", self.shared_total.load(Ordering::Relaxed));
        Ok(true)
    }
}