mod aggregations;
mod allocation;
mod arena;
mod arena_key;
mod hashtable;

use crate::io::OperationIOSubmitter;
use crate::operations::group::arena::ByteArena;
pub use crate::operations::group::arena_key::{ArenaKey, StringKey};
pub use crate::operations::group::hashtable::{Entry, HashTable as BaseHashTable, Value};
use crate::operations::{ConsumeContext, Operation, Output, PipelineBreaker};
use aggregations::Count;
use ahash::RandomState;
use arrow_array::builder::{StringBuilder, UInt64Builder};
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use tracing::debug;

type HashTable<K, V> = BaseHashTable<K, V, Vec<Entry<K, V>>>;

fn map_to_record_batch(map: &HashTable<ArenaKey, Count>) -> Result<RecordBatch, ArrowError> {
    let mut key_b = StringBuilder::with_capacity(map.len(), 4096);
    let mut val_b = UInt64Builder::with_capacity(map.len());

    for entry in map.iter() {
        key_b.append_value(unsafe { str::from_utf8_unchecked(entry.key().as_ref()) });
        val_b.append_value(entry.value().value as u64);
    }

    let keys: ArrayRef = Arc::new(key_b.finish());
    let vals: ArrayRef = Arc::new(val_b.finish());

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::UInt64, false),
    ]));

    RecordBatch::try_new(schema, vec![keys, vals])
}

const PARTITIONS: usize = 256;
const DEFAULT_CAPACITY: usize = 128;

type GroupState = (ByteArena, Vec<HashTable<ArenaKey, Count>>);

pub struct Group {
    arena: ByteArena,
    group_column: usize,
    output: Box<dyn Output>,

    maps: Vec<HashTable<ArenaKey, Count>>,
    hash_state: RandomState,
    injector: Arc<Injector<PartitionJob>>,
    sender: Sender<GroupState>,
    receiver: Option<Receiver<GroupState>>,
    partition_jobs_injected: Arc<AtomicBool>,
}

impl Group {
    pub fn new(
        state: RandomState,
        injector: Arc<Injector<PartitionJob>>,
        output: Box<dyn Output>,
        group_column: usize,
        sender: Sender<GroupState>,
        receiver: Option<Receiver<GroupState>>,
        partition_jobs_injected: Arc<AtomicBool>,
    ) -> Self {
        assert_eq!(size_of::<Entry<ArenaKey, Count>>(), 32);
        Self {
            arena: ByteArena::new(),
            group_column,
            sender,
            receiver,
            output,
            hash_state: state,
            maps: (0..PARTITIONS)
                .map(|_| HashTable::new(DEFAULT_CAPACITY))
                .collect(),
            injector,
            partition_jobs_injected,
        }
    }

    fn compute_hashes(&mut self, c: &StringViewArray) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(c.len());
        let mut i = 0;
        let length = c.len();
        while i < length {
            hashes.push(self.hash_state.hash_one(unsafe { c.value_unchecked(i) }));
            i += 1;
        }
        hashes
    }
}

impl Operation for Group {
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> crate::operations::Result<Option<RecordBatch>> {
        let col = batch
            .column(self.group_column)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();

        let hashes = self.compute_hashes(col);

        let length = hashes.len();
        let mut i = 0;

        while i < length {
            let hash = hashes[i];
            const USIZE_BITS: usize = 64;

            let idx =
                (hash as usize >> (USIZE_BITS - PARTITIONS.ilog2() as usize)) & (PARTITIONS - 1);
            let map = &mut self.maps[idx];

            let live_key = StringKey::new(&mut self.arena, unsafe { col.value_unchecked(i) });
            map.merge(hash, live_key, Count::single());
            i += 1;
        }

        Ok(None)
    }
}

pub struct PartitionJob {
    /// These maps are in the same partition
    maps: Vec<HashTable<ArenaKey, Count>>,
    /// Arenas are held by all so that no ArenaKey will be invalid (if worker 1 finishes, drops it's
    /// arena, and another worker is using worker 1's HashTable, they will get pointers to
    /// deallocated data)
    _arenas: Arc<Vec<ByteArena>>,
}

impl PartitionJob {
    pub fn run(mut self, output: &mut Box<dyn Output>) -> crate::operations::Result<()> {
        let (idx, _) = self
            .maps
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.capacity())
            .unwrap();
        let mut result_map = self.maps.swap_remove(idx);
        for map in self.maps {
            let map_iterator = map.iter();
            for entry in map_iterator {
                result_map.merge(entry.hash(), *entry.key(), *entry.value());
            }
        }

        output.write(map_to_record_batch(&result_map)?);
        Ok(())
    }
}

impl PipelineBreaker for Group {
    fn finish(mut self: Box<Self>) -> crate::operations::Result<()> {
        self.sender.send((self.arena, self.maps)).unwrap();
        drop(self.sender);

        if let Some(rx) = self.receiver {
            // We are the leader - we must populate the injector so everyone can begin working...
            let (arenas, mut partitions): (Vec<_>, Vec<_>) = rx.into_iter().unzip();
            // It's critical we don't drop the arenas until everyone finished working (if an arena
            // drops, someone could end up touching deallocated memory - our unsafe work in ArenaKey
            // means Rust does not protect us
            let arenas = Arc::new(arenas);

            let partitions_iterator = (0..PARTITIONS).map(|_| {
                partitions
                    .iter_mut()
                    .map(|w| w.pop().unwrap())
                    .collect::<Vec<_>>()
            });
            debug!("Pushing in partitions!");
            for maps in partitions_iterator {
                self.injector.push(PartitionJob {
                    maps,
                    _arenas: arenas.clone(),
                });
            }
            self.partition_jobs_injected.store(true, Ordering::Relaxed);
        }

        loop {
            let steal = self.injector.steal();
            match steal {
                Steal::Success(job) => {
                    job.run(&mut self.output)?;
                }
                Steal::Empty => {
                    if self.partition_jobs_injected.load(Ordering::Relaxed) {
                        // Guaranteed to be no more work to do
                        break;
                    }
                }
                Steal::Retry => continue,
            }
        }

        self.output.finish();
        Ok(())
    }
}
