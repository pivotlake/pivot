use std::ops::{Index, IndexMut};
use std::sync::{Arc, LazyLock};

use ahash::RandomState;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::Unary;
use crate::operations::unary::join::directory::{Directory, JoinDirectory};
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::Value;
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::RECORD_BATCH_SIZE;

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
    allocator: SlabAllocator,
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            hashes: Box::new([0; RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
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

    #[inline(always)]
    fn probe_with_dir<B: Index<usize, Output = u64> + IndexMut<usize>, S: Sender<RecordBatch>>(
        &mut self,
        directory: &Directory<B>,
        col: &Int64Array,
        sender: &mut S,
    ) -> unary::Result<()> {
        self.compute_hashes(col, directory);

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
                directory.prefetch(ph);
            }

            if i + 8 < len {
                let h = unsafe { *self.hashes.get_unchecked(i + 8) };
                let slot = directory.slot_for(h) as isize;
                let start = directory.end_ptr(slot - 1);
                let ptr = unsafe {arena.as_ptr().add(start) } as *const i8;

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
                }
            }

            let hash = unsafe { *self.hashes.get_unchecked(i) };
            let slot = directory.slot_for(hash) as isize;
            let start = directory.end_ptr(slot - 1);
            let end = directory.end_ptr(slot);

            let probe_key = unsafe { col.value_unchecked(i) } as u32;

            for j in start..end {
                let entry: Value = unsafe { *arena.get_unchecked(j) };
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
}
