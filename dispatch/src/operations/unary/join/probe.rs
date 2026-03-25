use std::sync::{Arc, LazyLock};

use ahash::RandomState;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::Unary;
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::Value;
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::RECORD_BATCH_SIZE;

static PROBE_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("probe_idx", DataType::Int64, false),
        Field::new("build_key", DataType::Int64, false),
        Field::new("build_payload", DataType::Int64, false),
    ]))
});

pub struct Probe {
    table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    batch_ptrs: Box<[(usize, usize); RECORD_BATCH_SIZE]>,
    allocator: SlabAllocator,
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            batch_ptrs: Box::new([(0usize, 0usize); RECORD_BATCH_SIZE]),
            allocator: SlabAllocator::new(false),
        }
    }

    #[inline(always)]
    fn compute_hashes(&mut self, col: &Int64Array) {
        let directory = unsafe { &*self.table.directory.get() };
        let mut i = 0;
        let length = col.len();
        while i < length {
            let hash = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
            let slot = directory.slot_for(hash);
            let end = directory.end_ptr(slot as isize);
            let start = directory.end_ptr(slot as isize - 1);
            self.batch_ptrs[i] = (start, end);
            i += 1;
        }
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
        self.compute_hashes(col);

        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut probe_indices = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut build_keys = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut build_payloads = JoinPrimitiveBuilder::<Int64Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut out = 0;

        let mut i = 0;
        while i < len {
            let (start, end) = unsafe { *self.batch_ptrs.get_unchecked(i) };
            let probe_key = unsafe { col.value_unchecked(i) } as u64;

            for j in start..end {
                let entry: &Value = unsafe { arena.get_unchecked(j) };
                if entry.0 == probe_key {
                    probe_indices.write(out, i as i64);
                    build_keys.write(out, entry.0 as i64);
                    build_payloads.write(out, entry.1 as i64);
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
                probe_indices.into_array(out),
                build_keys.into_array(out),
                build_payloads.into_array(out),
            ],
        )?;
        sender.send(result)?;
        Ok(())
    }
}
