use ahash::RandomState;
use arrow_array::builder::{ArrayBuilder, Int64Builder};
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::Unary;
use crate::operations::unary::join::directory::Directory;
use crate::operations::unary::join::JoinTable;
use crate::operations::unary::join::Value;
use crate::RECORD_BATCH_SIZE;

pub struct Probe {
    table: JoinTable,
    hash_state: RandomState,
    key_column: usize,
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
}

impl Probe {
    pub fn new(table: JoinTable, hash_state: RandomState, key_column: usize) -> Self {
        Self {
            table,
            hash_state,
            key_column,
            hashes: Box::new([0u64; RECORD_BATCH_SIZE]),
        }
    }

    #[inline(always)]
    fn compute_hashes(&mut self, col: &Int64Array) {
        let mut i = 0;
        let length = col.len();
        while i < length {
            self.hashes[i] = self.hash_state.hash_one(unsafe { col.value_unchecked(i) });
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

        let directory = unsafe { &*self.table.directory.get() };
        let arena = unsafe { &*self.table.arena.get() };
        let len = col.len();

        let mut probe_indices = Int64Builder::with_capacity(len);
        let mut build_keys = Int64Builder::with_capacity(len);
        let mut build_payloads = Int64Builder::with_capacity(len);

        let mut i = 0;
        while i < len {
            let hash = unsafe { *self.hashes.get_unchecked(i) };

            if directory.matches_bloom(hash) {
                let slot = directory.slot_for(hash);
                let end = directory.end_ptr(slot as isize);
                let start = directory.end_ptr(slot as isize - 1);
                let probe_key = unsafe { col.value_unchecked(i) } as u64;

                for j in start..end {
                    let entry: &Value = unsafe { arena.get_unchecked(j) };
                    if entry.0 == probe_key {
                        probe_indices.append_value(i as i64);
                        build_keys.append_value(entry.0 as i64);
                        build_payloads.append_value(entry.1 as i64);
                    }
                }
            }

            i += 1;
        }

        if probe_indices.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("probe_idx", DataType::Int64, false),
            Field::new("build_key", DataType::Int64, false),
            Field::new("build_payload", DataType::Int64, false),
        ]));

        let result = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(probe_indices.finish()),
                Arc::new(build_keys.finish()),
                Arc::new(build_payloads.finish()),
            ],
        )?;

        sender.send(result)?;
        Ok(())
    }
}
