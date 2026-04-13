use std::hint::black_box;
use std::mem;
use std::ops::{Index, IndexMut, Sub};
use std::ptr::null;
use std::sync::{Arc, LazyLock};
use std::time::Instant;
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
use crate::operations::unary::join::directory::{prefetch_ptr_l2, Directory, JoinDirectory, PtrBuffer, PTR_SHIFT};
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::Value;
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::perf_stat::{perf_disable, perf_enable};
use crate::RECORD_BATCH_SIZE;
use crate::worker::WORKER_IDX;

// struct SearchEntry {
//     start: u64,
//     end: u64,
//     arena:
// }


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

            matched_indexes: Box::new([0; RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
            total: 0,
        }
    }

    #[inline(never)]
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
            self.entries[i] = unsafe { *self.ptrs[i] };
        }

        for i in RECORD_BATCH_SIZE-PREFETCH_DISTANCE..RECORD_BATCH_SIZE {
            self.entries[i] = unsafe { *self.ptrs[i] };
        }
    }

    #[inline(never)]
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
            self.entries[i] = unsafe { *self.ptrs[i] };
        }

        for i in size-prefetch_size..size {
            self.entries[i] = unsafe { *self.ptrs[i] };
        }
    }


    // /// The only reason this would be good is - if the CPU has several pipelines before it, and it
    // /// cannot dispatch the next pipeline, it look for another pipelin
    // #[inline(always)]
    // fn probe_groups<
    //     B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    //     S: Sender<RecordBatch>,
    // >(
    //     &mut self,
    //     directory: &Directory<B>,
    //     col: &Int64Array,
    //     sender: &mut S,
    // ) -> unary::Result<()> {
    //
    //     const GROUP_SIZE: usize = 10;
    //     let mut hashes: [u64; GROUP_SIZE] = [0; GROUP_SIZE];
    //     let mut ptrs: [*const u64; GROUP_SIZE] = [null(); GROUP_SIZE];
    //     let mut values: [u64; GROUP_SIZE] = [0; GROUP_SIZE];
    //
    //     let i = 0;
    //     while i < col.len() {
    //         for j in 0..GROUP_SIZE {
    //             hashes[j] = self.hash_state.hash_one(unsafe { col.value_unchecked(i + j) });
    //             ptrs[j] = directory.ptr_for_slot((hashes[j] >> directory.shift) as usize);
    //         }
    //
    //         // We want to separate ptr deref from hashes and ptr computations, becaus if a single
    //         // deref hits dram, we don't want all instructions after it to sit needlssly in reorder
    //         // buffer
    //         for j in 0..GROUP_SIZE {
    //             values[j] = unsafe { *ptrs[j] };
    //         }
    //
    //
    //         for j in 0..GROUP_SIZE {
    //             values[j] = directory.ptr_for_slot(j >> directory.shift);
    //         }
    //     }
    //
    //     Ok(())
    // }

    #[inline(never)]
    fn output<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>>(&mut self, directory: &Directory<B>, sender: &mut S,
              col: &Int64Array,) -> unary::Result<()> {
        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        let mut i = 0;
        while i < len {
            let hash = unsafe { *self.hashes.get_unchecked(i) };
            let entry = self.entries[i];

            let probe = Directory::<B>::compute_tag(hash) as u64;
            if entry & probe != probe {
                i += 1;
                continue;
            }

            let start = unsafe { *self.ptrs[i].sub(1) } >> PTR_SHIFT;
            let end = entry >> PTR_SHIFT;
            let probe_key = unsafe { col.value_unchecked(i) } as u32;

            for j in start..end {
                // let entry: Value = arena[j as usize];
                // if entry == probe_key {
                if j < 3_000_0000_0000 {
                    // black_box(j);
                    lineitem_keys.write(out, entry as i64);
                    order_keys.write(out, entry as i64);
                    out += 1;
                }
                // }
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
        const GROUP_SIZE: usize = 10;
        let mut hashes: [u64; GROUP_SIZE * 10] = [0; GROUP_SIZE * 10];

        let mut end_ptrs: [*const u64; GROUP_SIZE] = [null(); GROUP_SIZE];
        let mut end_entries = [0u64; GROUP_SIZE];

        let mut next_end_ptrs: [*const u64; GROUP_SIZE] = [null(); GROUP_SIZE];
        let mut next_end_entries: [u64; GROUP_SIZE] = [0; GROUP_SIZE];

        let mut start_ptrs: [(*const u64, usize); RECORD_BATCH_SIZE] = [(null(), 0); RECORD_BATCH_SIZE];

        let mut matched_index_count = 0;

        let mut i = 0;

        while i < col.len() {
            let group_size = std::cmp::min(GROUP_SIZE, col.len() - i);

            let hash_counter = i % 128;

            for j in 0..group_size {
                hashes[hash_counter + j] = self.hash_state.hash_one(unsafe { col.value_unchecked(i + j) });
                prefetch_ptr_l2(directory.ptr_for_slot((hashes[hash_counter + j] >> directory.shift) as usize) as *const u8);

                next_end_ptrs[j] = directory.ptr_for_slot((hashes[hash_counter + j + GROUP_SIZE] >> directory.shift) as usize);
            }

            for j in 0..group_size {
                next_end_entries[j] =  unsafe { *next_end_ptrs[j] };
            }

            for j in 0..group_size {
                let entry = end_entries[j];
                let probe = Directory::<B>::compute_tag(entry) as u64;
                if entry & probe != probe {
                    continue;
                }

                start_ptrs[matched_index_count] = (unsafe { end_ptrs[j].sub(1) }, i + j);
                matched_index_count += 1;
            }

            i += group_size;
            mem::swap(&mut end_entries, &mut next_end_entries);
            mem::swap(&mut end_ptrs, &mut next_end_ptrs);
        }

        let mut arenas_ptrs: [(usize, usize, usize); GROUP_SIZE] = [(0, 0, 0); GROUP_SIZE];
        let mut load_arenas_ptrs: [(usize, usize, usize); GROUP_SIZE] = [(0, 0, 0); GROUP_SIZE];

        let arena = unsafe { &*self.table.arena.get() };

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        const PREFETCH_DISTANCE: usize = 64;
        for i in 0..matched_index_count {
            let group_size = std::cmp::min(GROUP_SIZE, col.len() - i);

            if i + PREFETCH_DISTANCE * 2 < matched_index_count {
                let (ptr, _) = start_ptrs[i + PREFETCH_DISTANCE];
                prefetch_ptr_l2(ptr as *const u8);
            }

            for j in 0..group_size {
                let (ptr, idx) = start_ptrs[i + j];
                let value = unsafe { *ptr };
                let entry = unsafe { *ptr.add(1) };
                let start = (value >> PTR_SHIFT) as usize;
                let end = (entry >> PTR_SHIFT) as usize;
                arenas_ptrs[j] = (start, end, idx);
            }

            for j in 0..group_size {
                let (start, _, _) = arenas_ptrs[j];
                black_box(arena[start]);
            }

            for j in 0..group_size {
                let (start, end, idx) = load_arenas_ptrs[j];
                let probe_key = unsafe { col.value_unchecked(idx) } as u32;
                for k in start..end {
                    let entry: Value = arena[k];
                    if entry == probe_key {
                        lineitem_keys.write(out, entry as i64);
                        order_keys.write(out, entry as i64);
                        out += 1;
                    }
                }
            }
        }

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
            JoinDirectory::NonContiguous(dir) => panic!("Oh on"),
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_disable();
        println!("total {:?}", mem::take(&mut self.total));
        Ok(true)
    }
}