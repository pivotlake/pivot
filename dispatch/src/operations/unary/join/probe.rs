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
        const BUF_SIZE: usize = GROUP_SIZE * 10;
        let len = col.len();
        if len == 0 {
            return Ok(());
        }

        // Circular buffer for hashes and directory ptrs
        let mut hashes: [u64; BUF_SIZE] = [0; BUF_SIZE];
        let mut dir_ptrs: [*const u64; BUF_SIZE] = [null(); BUF_SIZE];
        // Entries loaded by tight deref loop, indexed same as circular buffer
        let mut entries: [u64; BUF_SIZE] = [0; BUF_SIZE];

        // Bloom-matched: (ptr to slot entry, col index)
        let mut start_ptrs: [(*const u64, usize); RECORD_BATCH_SIZE] = [(null(), 0); RECORD_BATCH_SIZE];
        let mut matched_count = 0;

        // --- Bootstrap ---
        // Fill circular buffer: hash up to BUF_SIZE elements, prefetch all
        let bootstrap_size = std::cmp::min(BUF_SIZE, len);
        for k in 0..bootstrap_size {
            hashes[k] = self.hash_state.hash_one(unsafe { col.value_unchecked(k) });
            dir_ptrs[k] = directory.ptr_for_slot((hashes[k] >> directory.shift) as usize);
            prefetch_ptr_l2(dir_ptrs[k] as *const u8);
        }

        // Deref group 0 (tight loop) so bloom can start on first main iteration
        let g0_size = std::cmp::min(GROUP_SIZE, len);
        for j in 0..g0_size {
            entries[j] = unsafe { *dir_ptrs[j] };
        }

        // Previous group position in circular buffer for bloom check
        let mut bloom_off: usize = 0;
        let mut bloom_col: usize = 0;
        let mut bloom_size: usize = g0_size;

        let mut write_pos = bootstrap_size;

        // --- Main loop ---
        // deref_col tracks the col index of the group we're deref-ing
        let mut deref_col = GROUP_SIZE;
        while deref_col < len {
            let deref_size = std::cmp::min(GROUP_SIZE, len - deref_col);
            let deref_off = deref_col % BUF_SIZE;

            // Stage 1: Bloom check previous group (pure ALU, no loads)
            for j in 0..bloom_size {
                let tag = Directory::<B>::compute_tag(hashes[bloom_off + j]) as u64;
                if entries[bloom_off + j] & tag != tag {
                    continue;
                }
                start_ptrs[matched_count] = (dir_ptrs[bloom_off + j], bloom_col + j);
                matched_count += 1;
            }

            // Stage 2: Deref directory ptrs for current group (tight loop)
            for j in 0..deref_size {
                entries[deref_off + j] = unsafe { *dir_ptrs[deref_off + j] };
            }

            // Stage 3: Hash + prefetch next GROUP_SIZE into circular buffer
            let hash_size = std::cmp::min(GROUP_SIZE, len.saturating_sub(write_pos));
            for j in 0..hash_size {
                let slot = write_pos % BUF_SIZE;
                hashes[slot] = self.hash_state.hash_one(unsafe { col.value_unchecked(write_pos) });
                dir_ptrs[slot] = directory.ptr_for_slot((hashes[slot] >> directory.shift) as usize);
                prefetch_ptr_l2(dir_ptrs[slot] as *const u8);
                write_pos += 1;
            }

            bloom_off = deref_off;
            bloom_col = deref_col;
            bloom_size = deref_size;
            deref_col += deref_size;
        }

        // Drain: bloom check the last deref'd group
        for j in 0..bloom_size {
            let tag = Directory::<B>::compute_tag(hashes[bloom_off + j]) as u64;
            if entries[bloom_off + j] & tag != tag {
                continue;
            }
            start_ptrs[matched_count] = (dir_ptrs[bloom_off + j], bloom_col + j);
            matched_count += 1;
        }

        // --- Phase 2: arena pipeline ---
        // 3 stages, all reading directly from start_ptrs:
        //   L2 prefetch: 64 matched entries ahead
        //   L1 load (black_box): current group (1 group ahead of probe)
        //   Probe: previous group

        let arena = unsafe { &*self.table.arena.get() };

        let mut lineitem_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut order_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        const L2_DISTANCE: usize = 64;

        // L2 prefetch bootstrap: prefetch first 64 entries
        let l2_bootstrap = std::cmp::min(L2_DISTANCE, matched_count);
        for k in 0..l2_bootstrap {
            let (ptr, _) = start_ptrs[k];
            let start = (unsafe { *ptr.sub(1) } >> PTR_SHIFT) as usize;
            prefetch_ptr_l2(arena.ptr_at_index(start) as *const u8);
        }

        // L1 load bootstrap: black_box load first group
        let g0 = std::cmp::min(GROUP_SIZE, matched_count);
        for j in 0..g0 {
            let (ptr, _) = start_ptrs[j];
            let start = (unsafe { *ptr.sub(1) } >> PTR_SHIFT) as usize;
            black_box(unsafe { *(arena.ptr_at_index(start) as *const u32) });
        }

        let mut prev_start = 0usize;
        let mut prev_size = g0;
        let mut load_col = GROUP_SIZE;

        while load_col < matched_count {
            let load_size = std::cmp::min(GROUP_SIZE, matched_count - load_col);

            // Stage 1: L2 prefetch 64 ahead (tight loop)
            let l2_start = load_col + L2_DISTANCE;
            let l2_end = std::cmp::min(l2_start + load_size, matched_count);
            for k in l2_start..l2_end {
                let (ptr, _) = start_ptrs[k];
                let start = (unsafe { *ptr.sub(1) } >> PTR_SHIFT) as usize;
                prefetch_ptr_l2(arena.ptr_at_index(start) as *const u8);
            }

            // Stage 2: L1 load current group via black_box (tight loop, warm in L2)
            for j in 0..load_size {
                let (ptr, _) = start_ptrs[load_col + j];
                let start = (unsafe { *ptr.sub(1) } >> PTR_SHIFT) as usize;
                black_box(unsafe { *(arena.ptr_at_index(start) as *const u32) });
            }

            // Stage 3: Probe previous group (in L1)
            for j in 0..prev_size {
                let (ptr, idx) = start_ptrs[prev_start + j];
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

            prev_start = load_col;
            prev_size = load_size;
            load_col += load_size;
        }

        // Drain: probe the last group (already in L1)
        for j in 0..prev_size {
            let (ptr, idx) = start_ptrs[prev_start + j];
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