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
const BATCH: usize = 1;
const RING_SIZE: usize = 128;

// struct SearchEntry {
//     start: u64,
//     end: u64,
//     arena:
// }
const GROUP_SIZE: usize = 40;
const HASH_RING_SIZE: usize = 120;


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

    #[inline(never)]
    fn compute_hashes(&mut self, col: &Int64Array) {
        let mut i = 0;
        let length = col.len();
        while i < length {
            let hash = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            self.hashes[i] = hash;
            i += 1;
        }
    }


    #[inline(never)]
    fn compute_hashes_len(&mut self, col: &Int64Array, len: usize) {
        let mut i = 0;
        let length = min(col.len(), len);
        while i < length {
            let hash = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            self.hashes[i] = hash;
            black_box(self.hashes[i]);
            i += 1;
        }
    }

    #[inline(always)]
    fn compute_hashes_len2<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, directory: &Directory<B>, col: &Int64Array, mut idx: usize, offset: usize, len: usize) {
        let mut i = offset;
        let total = len + i;
        while i < total {
            let hash = self.hash_state.hash_one(unsafe { col.value_unchecked(idx) });
            self.hashes[i] = hash;
            prefetch_ptr_l2(directory.ptr_for_slot((hash >> directory.shift) as usize) as *const u8);
            idx += 1;
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
            black_box(unsafe { *self.ptrs[i] });
        }

        for i in size-prefetch_size..size {
            // self.entries[i] = unsafe { *self.ptrs[i] };
            black_box(unsafe { *self.ptrs[i] });
        }
    }

    #[inline(never)]
    fn touch_size_simple<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, size: usize, directory: &Directory<B>) {
        // for i in 0..size {
        //     self.ptrs[i] = directory.ptr_for_slot((self.hashes[i] >> directory.shift) as usize);
        // }

        for i in 0..size {
            if directory.matches_bloom(self.hashes[i]) {
                self.total += 1;
            }
        }
    }

    #[inline(always)]
    fn touch_size2<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(&mut self, hash_idx: usize, directory: &Directory<B>) {
        // for i in 0..BATCH {
        //     self.ptrs[i] = directory.ptr_for_slot((self.hashes[hash_idx + i] >> directory.shift) as usize);
        // }

        for i in 0..BATCH {
            // black_box(unsafe { *self.ptrs[i] });
            // let res = unsafe { *self.ptrs[i] };
            // black_box(self.hashes[hash_idx + i]);
            if directory.matches_bloom(self.hashes[hash_idx + i]) {
                self.total += 1;
            }
        }
    }

    // fn bloom_filters
    //

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

    #[inline(never)]
    fn probe_old_with_hashing<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        self.compute_hashes(col);
        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);

        let mut out = 0;

        let mut i = 0;
        while i < len {
            const PREFETCH_DISTANCE: usize = 16;
            if i + PREFETCH_DISTANCE < len {
                let ph = self.hashes[i + PREFETCH_DISTANCE];
                directory.prefetch_l1(ph);
            }

            // if i + 8 < len {
            //     let h = unsafe { *self.hashes.get_unchecked(i + 8) };
            //     let slot = directory.slot_for(h) as isize;
            //     let start = directory.end_ptr(slot - 1);
            //     let ptr = arena.ptr_at_index(start) as *const i8;
            //
            //     #[cfg(target_arch = "x86_64")]
            //     unsafe {
            //         std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
            //     }
            // }

            let hash = unsafe { *self.hashes.get_unchecked(i) };

            if !directory.matches_bloom(hash) {
                i += 1;
                continue;
            }
            self.total += 1;
            black_box(hash);
            // //
            // let slot = directory.slot_for(hash) as isize;
            // let start = directory.end_ptr(slot - 1);
            // let end = directory.end_ptr(slot);
            //
            // let probe_key = unsafe { col.value_unchecked(i) } as u32;
            //
            // for j in start..end {
            //     let entry: Value = arena[j];
            //     // if entry == probe_key {
            //         lineitem_keys.write(out, entry as i64);
            //         order_keys.write(out, entry as i64);
            //         out += 1;
            //     // }
            // }

            i += 1;
        }

        if out == 0 {
            return Ok(());
        }

        if out > RECORD_BATCH_SIZE {
            panic!("OH no!");
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

    #[inline(never)]
    fn probe_old<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        self.compute_hashes(col);
        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);

        let mut out = 0;

        let mut i = 0;
        while i < len {
            const PREFETCH_DISTANCE: usize = 16;
            if i + PREFETCH_DISTANCE < len {
                let ph = self.hashes[i + PREFETCH_DISTANCE];
                directory.prefetch_l1(ph);
            }

            // if i + 8 < len {
            //     let h = unsafe { *self.hashes.get_unchecked(i + 8) };
            //     let slot = directory.slot_for(h) as isize;
            //     let start = directory.end_ptr(slot - 1);
            //     let ptr = arena.ptr_at_index(start) as *const i8;
            //
            //     #[cfg(target_arch = "x86_64")]
            //     unsafe {
            //         std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
            //     }
            // }

            let hash = unsafe { *self.hashes.get_unchecked(i) };

            if !directory.matches_bloom(hash) {
                i += 1;
                continue;
            }
            self.total += 1;
            black_box(hash);
            // //
            // let slot = directory.slot_for(hash) as isize;
            // let start = directory.end_ptr(slot - 1);
            // let end = directory.end_ptr(slot);
            //
            // let probe_key = unsafe { col.value_unchecked(i) } as u32;
            //
            // for j in start..end {
            //     let entry: Value = arena[j];
            //     // if entry == probe_key {
            //         lineitem_keys.write(out, entry as i64);
            //         order_keys.write(out, entry as i64);
            //         out += 1;
            //     // }
            // }

            i += 1;
        }

        if out == 0 {
            return Ok(());
        }

        if out > RECORD_BATCH_SIZE {
            panic!("OH no!");
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

    // fn probe_old<
    //     B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    //     S: Sender<RecordBatch>,
    // >(
    //     &mut self,
    //     directory: &Directory<B>,
    //     col: &Int64Array,
    //     sender: &mut S,
    // ) -> unary::Result<()> {
    //     self.compute_hashes(col);
    //     let arena = unsafe { &*self.table.arena.get() };
    //     let len = col.len();
    //
    //     let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
    //     let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
    //
    //     let mut out = 0;
    //
    //     let mut i = 0;
    //     while i < len {
    //         const PREFETCH_DISTANCE: usize = 16;
    //         if i + PREFETCH_DISTANCE < len {
    //             let ph = self.hashes[i + PREFETCH_DISTANCE];
    //             directory.prefetch_l1(ph);
    //         }
    //         //
    //         // if i + 8 < len {
    //         //     let h = unsafe { *self.hashes.get_unchecked(i + 8) };
    //         //     let slot = directory.slot_for(h) as isize;
    //         //     let start = directory.end_ptr(slot - 1);
    //         //     let ptr = arena.ptr_at_index(start) as *const i8;
    //         //
    //         //     #[cfg(target_arch = "x86_64")]
    //         //     unsafe {
    //         //         std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
    //         //     }
    //         // }
    //
    //         let hash = unsafe { *self.hashes.get_unchecked(i) };
    //
    //         if !directory.matches_bloom(hash) {
    //             i += 1;
    //             continue;
    //         }
    //         black_box(hash);
    //
    //         // let slot = directory.slot_for(hash) as isize;
    //         // let start = directory.end_ptr(slot - 1);
    //         // let end = directory.end_ptr(slot);
    //         //
    //         // let probe_key = unsafe { col.value_unchecked(i) } as u32;
    //         //
    //         // for j in start..end {
    //         //     let entry: Value = arena[j];
    //         //     if entry == probe_key {
    //         //         lineitem_keys.write(out, entry as i64);
    //         //         order_keys.write(out, entry as i64);
    //         //         out += 1;
    //         //     }
    //         // }
    //
    //         i += 1;
    //     }
    //
    //     if out == 0 {
    //         return Ok(());
    //     }
    //
    //     let result = RecordBatch::try_new(
    //         PROBE_SCHEMA.clone(),
    //         vec![
    //             lineitem_keys.into_array(out),
    //             order_keys.into_array(out),
    //         ],
    //     )?;
    //     sender.send(result)?;
    //     Ok(())
    // }

    #[inline(never)]
    fn simple_load< S: Sender<RecordBatch>>(&mut self,  sender: &mut S, matched_count: usize, start_ptrs: &[(*const u64, usize); RECORD_BATCH_SIZE], col: &Int64Array) -> unary::Result<()> {
        let arena = unsafe { &*self.table.arena.get() };

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        const PREFETCH_DISTANCE: usize = 8;

        for i in 0..matched_count {
            // if i + PREFETCH_DISTANCE < matched_count {
            //     let ph = start_ptrs[i + PREFETCH_DISTANCE];
            //     directory.prefetch_l1(ph);
            // }
            //
            // if i + PREFETCH_DISTANCE < matched_count {
            //     let (ptr, _) = start_ptrs[i + PREFETCH_DISTANCE];
            //     let start = (unsafe { *ptr } >> PTR_SHIFT) as usize;
            //     let val =  arena.ptr_at_index(start);
            //     prefetch_ptr(val as *const u8);
            // }

            let (ptr, idx) = start_ptrs[i];
            let start = (unsafe { *ptr.sub(1) } >> PTR_SHIFT) as usize;
            let end = (unsafe { *ptr } >> PTR_SHIFT) as usize;
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

    // fn run_probe_once<
    //     const RUN_HASHES: bool,
    //
    //     B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    //     S: Sender<RecordBatch>,
    // >(
    //     &mut self,
    //     col_index: &mut usize,
    //     directory: &Directory<B>,
    //     col: &Int64Array,
    //
    //     hashes: &mut [u64; HASH_RING_SIZE],
    //     end_ptrs: &mut [*const u64; GROUP_SIZE],
    //     next_end_ptrs: &mut [*const u64; GROUP_SIZE],
    //     end_entries: &mut [u64; GROUP_SIZE],
    //     next_end_entries: &mut [u64; GROUP_SIZE],
    //     start_ptrs: &mut [(*const u64, usize); RECORD_BATCH_SIZE],
    //     matched_index_count: &mut usize
    // ) {
    //     let group_size = std::cmp::min(GROUP_SIZE, col.len() - *col_index);
    //
    //     let hash_counter = (*col_index + GROUP_SIZE) % HASH_RING_SIZE;
    //     let matching_group_counter = (*col_index - GROUP_SIZE) % HASH_RING_SIZE;
    //     let next_group_counter = *col_index % HASH_RING_SIZE;
    //
    //     if RUN_HASHES {
    //         for j in 0..group_size {
    //             hashes[hash_counter + j] = self.hash_state.hash_one(unsafe { col.value_unchecked(*col_index + j) });
    //             prefetch_ptr_l2(directory.ptr_for_slot((hashes[hash_counter + j] >> directory.shift) as usize) as *const u8);
    //         }
    //     }
    //
    //     for j in 0..GROUP_SIZE {
    //         next_end_ptrs[j] = directory.ptr_for_slot((hashes[next_group_counter + j] >> directory.shift) as usize);
    //     }
    //
    //     // Interesting if we'll see lots of cycles at end of previous loop? Maybe we could
    //     // overlap with next section if we have another layer of depth in pipeline with
    //     // something like "next_next_end_ptrs"
    //     for j in 0..GROUP_SIZE {
    //         next_end_entries[j] =  unsafe { *next_end_ptrs[j] };
    //     }
    //
    //     for j in 0..GROUP_SIZE {
    //         let entry = end_entries[j];
    //         let probe = Directory::<B>::compute_tag(hashes[matching_group_counter + j]) as u64;
    //         if entry & probe != probe {
    //             continue;
    //         }
    //
    //         start_ptrs[*matched_index_count] = (unsafe { end_ptrs[j].sub(1) }, *col_index + j - 2 * GROUP_SIZE);
    //         *matched_index_count += 1;
    //     }
    //
    //     *col_index += group_size;
    //     mem::swap(end_entries, next_end_entries);
    //     mem::swap(end_ptrs, next_end_ptrs);
    // }

    #[inline(never)]
    fn probe_with_dir<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        /// [ GROUP-HASHING (+ Prefetch l2) ] [ GROUP-MATCHING ] [GROUP-LOADING]
        let mut hashes: [u64; HASH_RING_SIZE] = [0; HASH_RING_SIZE];

        let mut end_ptrs: [*const u64; GROUP_SIZE] = [null(); GROUP_SIZE];
        let mut end_entries = [0u64; GROUP_SIZE];

        let mut next_end_ptrs: [*const u64; GROUP_SIZE] = [null(); GROUP_SIZE];
        let mut next_end_entries: [u64; GROUP_SIZE] = [0; GROUP_SIZE];

        let mut start_ptrs: [(*const u64, usize); RECORD_BATCH_SIZE] = [(null(), 0); RECORD_BATCH_SIZE];

        let mut matched_index_count = 0;

        for i in GROUP_SIZE..HASH_RING_SIZE {
            hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i - GROUP_SIZE) });
            prefetch_ptr_l2(directory.ptr_for_slot((hashes[i] >> directory.shift) as usize) as *const u8);
        }
        for j in 0..GROUP_SIZE {
            let pos = GROUP_SIZE + j;
            end_ptrs[j] = directory.ptr_for_slot((hashes[pos] >> directory.shift) as usize);
        }
        for j in 0..GROUP_SIZE {
            end_entries[j] = unsafe { *end_ptrs[j] };
        }

        let mut i = HASH_RING_SIZE - GROUP_SIZE;

        while i < col.len() {
            let group_size = std::cmp::min(GROUP_SIZE, col.len() - i);

            let hash_counter = (i + GROUP_SIZE) % HASH_RING_SIZE;
            let matching_group_counter = (i - GROUP_SIZE) % HASH_RING_SIZE;
            let next_group_counter = i % HASH_RING_SIZE;

            for j in 0..group_size {
                hashes[hash_counter + j] = self.hash_state.hash_one(unsafe { col.value_unchecked(i + j) });
                // prefetch_ptr(directory.ptr_for_slot((hashes[hash_counter + j] >> directory.shift) as usize) as *const u8);
            }

            for j in 0..GROUP_SIZE {
                next_end_ptrs[j] = directory.ptr_for_slot((hashes[next_group_counter + j] >> directory.shift) as usize);
            }

            // Interesting if we'll see lots of cycles at end of previous loop? Maybe we could
            // overlap with next section if we have another layer of depth in pipeline with
            // something like "next_next_end_ptrs"
            for j in 0..GROUP_SIZE {
                next_end_entries[j] =  unsafe { *next_end_ptrs[j] };
                black_box(next_end_entries[j]);
            }
            //
            // for j in 0..GROUP_SIZE {
            //     let entry = end_entries[j];
            //     let probe = Directory::<B>::compute_tag(hashes[matching_group_counter + j]) as u64;
            //     if entry & probe != probe {
            //         continue;
            //     }
            //
            //     start_ptrs[matched_index_count] = (unsafe { end_ptrs[j].sub(1) }, i + j - 2 * GROUP_SIZE);
            //     matched_index_count += 1;
            // }

            i += group_size;
            mem::swap(&mut end_entries, &mut next_end_entries);
            mem::swap(&mut end_ptrs, &mut next_end_ptrs);
        }

        for ptr in start_ptrs {
            black_box(ptr);
        }

        // /// We will finish at a state something like this:
        // /// [ HASHES ] [ HASHES ]
        // /// [ end_ptrs]
        // ///
        // /// We need to do one loop to create entries for last hashes, and two loops to generate start_ptrs
        // let last_group_size = match col.len() % GROUP_SIZE {
        //     0 => GROUP_SIZE,
        //     r => r,
        // };
        // let last_i = i - last_group_size;
        //
        // let next_group_counter = (last_i + GROUP_SIZE) % HASH_RING_SIZE;
        // let mut matching_group_counter = last_i % HASH_RING_SIZE;
        // for j in 0..last_group_size {
        //     next_end_ptrs[j] = directory.ptr_for_slot((hashes[next_group_counter + j] >> directory.shift) as usize);
        // }
        // for j in 0..last_group_size {
        //     next_end_entries[j] =  unsafe { *next_end_ptrs[j] };
        // }
        //
        // for j in 0..GROUP_SIZE {
        //     let entry = end_entries[j];
        //     let probe = Directory::<B>::compute_tag(hashes[matching_group_counter + j]) as u64;
        //     if entry & probe != probe {
        //         continue;
        //     }
        //
        //     start_ptrs[matched_index_count] = (unsafe { end_ptrs[j].sub(1) }, last_i - GROUP_SIZE + j);
        //     matched_index_count += 1;
        // }
        //
        // mem::swap(&mut end_entries, &mut next_end_entries);
        // mem::swap(&mut end_ptrs, &mut next_end_ptrs);
        //
        // matching_group_counter =  (matching_group_counter + GROUP_SIZE) % HASH_RING_SIZE;
        //
        // for j in 0..last_group_size {
        //     let entry = end_entries[j];
        //     let probe = Directory::<B>::compute_tag(hashes[matching_group_counter + j]) as u64;
        //     if entry & probe != probe {
        //         continue;
        //     }
        //
        //     start_ptrs[matched_index_count] = (unsafe { end_ptrs[j].sub(1) }, i + j - last_group_size);
        //     matched_index_count += 1;
        // }
        //
        // self.total += matched_index_count;
        // for p in start_ptrs {
        //     black_box(p);
        // }
        Ok(())
    }

    #[inline(never)]
    fn probe_with_dir2<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        const BATCH: usize = 256;


        let len = col.len();
        let mut i = 0;
        while i + BATCH <= len {
            let chunk = col.slice(i, BATCH);
            self.compute_hashes_len(&chunk, BATCH);
            self.touch_size(BATCH, directory);
            i += BATCH;
        }
        if i < len {
            let rem = len - i;
            let chunk = col.slice(i, rem);
            self.compute_hashes_len(&chunk, rem);
            self.touch_size(rem, directory);
        }
        Ok(())
    }

    #[inline(never)]
    fn probe_with_dir3<
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
        S: Sender<RecordBatch>,
    >(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {

        self.compute_hashes_len2(directory, &col, 0, 0, RING_SIZE - BATCH);

        let len = col.len();
        let mut i = 0;

        while i + BATCH <= (len - (RING_SIZE - BATCH))  {
            let hash_offset = (i + (RING_SIZE - BATCH)) % RING_SIZE;
            self.compute_hashes_len2(directory, &col, i + (RING_SIZE - BATCH), hash_offset, BATCH);

            let touch_offset = i % RING_SIZE;
            self.touch_size2(touch_offset, directory);
            i += BATCH;
        }

        while i + BATCH <= len {
            let touch_offset = i % RING_SIZE;
            self.touch_size2(touch_offset, directory);
            i += BATCH;
        }

        if i < len {
            let rem = len - i;
            let chunk = col.slice(i, rem);
            self.compute_hashes_len(&chunk, rem);
            self.touch_size_simple(rem, directory);
            // self.touch_size(rem, directory);
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
            JoinDirectory::Contiguous(dir) => {
                if col.len() <= RING_SIZE {
                    self.compute_hashes_len2(dir, col, 0, 0, col.len());
                    self.touch_size_simple(col.len(), dir);
                } else {
                    self.probe_with_dir3(dir, col, sender);
                }
                Ok(())
            },
            JoinDirectory::NonContiguous(dir) => {
                if col.len() <= RING_SIZE {
                    self.compute_hashes_len2(dir, col, 0, 0, col.len());
                    self.touch_size_simple(col.len(), dir);
                } else {
                    self.probe_with_dir3(dir, col, sender);
                }
                Ok(())
            },
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> unary::Result<bool> {
        perf_disable();
        self.shared_total.fetch_add(mem::take(&mut self.total), Ordering::Relaxed);
        println!("Shared total: {:?}", self.shared_total.load(Ordering::Relaxed));
        Ok(true)
    }
}