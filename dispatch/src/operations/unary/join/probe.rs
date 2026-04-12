use std::hint::black_box;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr::null;
use std::sync::{Arc, LazyLock};

use ahash::RandomState;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::Unary;
use crate::operations::unary::join::directory::{prefetch_ptr_l2, Directory, JoinDirectory, PtrBuffer};
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
    allocator: SlabAllocator,
    total: usize,
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            hashes: Box::new([0; RECORD_BATCH_SIZE]),
            ptrs: Box::new([null(); RECORD_BATCH_SIZE]),
            entries: Box::new([0; RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
            total: 0,
        }
    }

    #[inline(always)]
    fn compute_hashes<B: Index<usize, Output = u64> + IndexMut<usize>>(&mut self, col: &Int64Array, directory: &Directory<B>) {
        let mut i = 0;
        let length = col.len();
        while i < length {
            let hash = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            self.hashes[i] = hash;
            i += 1;
        }
    }

    fn touch<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, directory: &Directory<B>) {
        const PREFETCH_DISTANCE: usize = 64;
        for i in 0..PREFETCH_DISTANCE {
            self.ptrs[i] = directory.ptr_for_slot((self.hashes[i] >> directory.shift) as usize);
            prefetch_ptr_l2(self.ptrs[i] as *const u8);
        }

        for i in PREFETCH_DISTANCE..RECORD_BATCH_SIZE {
            self.ptrs[i] = directory.ptr_for_slot((self.hashes[i] >> directory.shift) as usize);
        }

        for i in 0..RECORD_BATCH_SIZE-PREFETCH_DISTANCE {
            prefetch_ptr_l2(self.ptrs[i + PREFETCH_DISTANCE] as *const u8);
            let a = unsafe { *self.ptrs[i] };
            black_box(a);
        }

        for i in RECORD_BATCH_SIZE-PREFETCH_DISTANCE..RECORD_BATCH_SIZE {
            let a = unsafe { *self.ptrs[i] };
            black_box(a);
        }
    }

    fn touch_size<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, size: usize, directory: &Directory<B>) {
        const PREFETCH_DISTANCE: usize = 64;

        let prefetch_size = std::cmp::min(size, PREFETCH_DISTANCE);

        for i in 0..prefetch_size {
            self.ptrs[i] = directory.ptr_for_slot((self.hashes[i] >> directory.shift) as usize);
            prefetch_ptr_l2(self.ptrs[i] as *const u8);
        }

        for i in prefetch_size..size {
            self.ptrs[i] = directory.ptr_for_slot((self.hashes[i] >> directory.shift) as usize);
        }

        for i in 0..size-prefetch_size {
            prefetch_ptr_l2(self.ptrs[i + PREFETCH_DISTANCE] as *const u8);
            let a = unsafe { *self.ptrs[i] };
            black_box(a);
        }

        for i in size-prefetch_size..size {
            let a = unsafe { *self.ptrs[i] };
            black_box(a);
        }
    }

    #[inline(always)]
    fn probe_with_dir<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        self.compute_hashes(col, directory);

        if col.len() == RECORD_BATCH_SIZE {
            self.touch(directory);
        } else {
            self.touch_size(col.len(), directory);
        }

        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        let mut i = 0;
        while i < len {
            let hash = unsafe { *self.hashes.get_unchecked(i) };

            if !directory.matches_bloom(hash) {
                i += 1;
                continue;
            }

            let slot = directory.slot_for(hash) as isize;
            let start = directory.end_ptr(slot - 1);
            let end = directory.end_ptr(slot);
            let probe_key = unsafe { col.value_unchecked(i) } as u32;

            for j in start..end {
                let entry: Value = arena[j];
                if entry == probe_key {
                    lineitem_keys.write(out, entry as i64);
                    order_keys.write(out, entry as i64);
                    out += 1;
                }
            }

            i += 1;
        }

        if out == 0 {
            return Ok(());
        }

        let result = RecordBatch::try_new(
            PROBE_SCHEMA.clone(),
            vec![
                lineitem_keys.into_array(out),
                order_keys.into_array(out),
            ],
        )?;
        sender.send(result)?;
        Ok(())
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
            JoinDirectory::Contiguous(dir) => self.probe_with_dir(dir, col, sender),
            JoinDirectory::NonContiguous(dir) => self.probe_with_dir(dir, col, sender),
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_disable();
        Ok(true)
    }
}