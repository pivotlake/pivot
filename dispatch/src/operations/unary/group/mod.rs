mod aggregations;
mod allocation;
mod arena;
mod arena_key;
mod hashtable;

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::group::arena::ByteArena;
pub use crate::operations::unary::group::arena_key::{ArenaKey, StringKey};
pub use crate::operations::unary::group::hashtable::{Entry, HashTable as BaseHashTable, Value};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use aggregations::Count;
use ahash::RandomState;
use arrow_array::builder::{StringBuilder, UInt64Builder};
use arrow_array::{Array, ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use thiserror::Error;
use tracing::debug;

type HashTable<K, V> = BaseHashTable<K, V, Vec<Entry<K, V>>>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] crate::operations::channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

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

pub struct GroupFactory {
    group_column: usize,
    hash_state: RandomState,
    injector: Arc<Injector<PartitionJob>>,
    partition_jobs_injected: Arc<AtomicBool>,

    sender: mpsc::Sender<GroupState>,
    receiver: Option<mpsc::Receiver<GroupState>>,
}

impl GroupFactory {
    pub fn create_for_workers(
        group_column: usize,
        worker_count: usize,
    ) -> impl IntoIterator<Item = GroupFactory> {
        let hash_state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<GroupState>();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| GroupFactory {
            group_column,
            hash_state: hash_state.clone(),
            injector: injector.clone(),
            partition_jobs_injected: partition_jobs_injected.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for GroupFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Group>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, Group> {
        PipelineBreaker::Consuming(Group::new(
            self.hash_state,
            self.injector,
            self.group_column,
            self.sender,
            self.receiver.take(),
            self.partition_jobs_injected,
        ))
    }
}

pub struct Group {
    arena: ByteArena,
    group_column: usize,

    maps: Vec<HashTable<ArenaKey, Count>>,
    hash_state: RandomState,

    sender: mpsc::Sender<GroupState>,
    outputter: GroupOutputter,
}

impl Group {
    pub fn new(
        state: RandomState,
        injector: Arc<Injector<PartitionJob>>,
        group_column: usize,
        sender: mpsc::Sender<GroupState>,
        receiver: Option<mpsc::Receiver<GroupState>>,
        partition_jobs_injected: Arc<AtomicBool>,
    ) -> Self {
        assert_eq!(size_of::<Entry<ArenaKey, Count>>(), 32);
        Self {
            arena: ByteArena::new(),
            group_column,
            outputter: GroupOutputter {
                injector,
                receiver,
                partition_jobs_injected,
            },
            hash_state: state,
            maps: (0..PARTITIONS)
                .map(|_| HashTable::new(DEFAULT_CAPACITY))
                .collect(),
            sender,
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

impl Consumer<RecordBatch, RecordBatch> for Group {
    type Outputter = GroupOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
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

            // We want to take the top bits instead of the bottom bits so we don't have bit overlap
            // within the HashTable itself (i.e., using same bits to decide the map and to place the
            // item within the table) as we'll have many collisions
            let idx =
                (hash as usize >> (USIZE_BITS - PARTITIONS.ilog2() as usize)) & (PARTITIONS - 1);
            let map = &mut self.maps[idx];

            let live_key = StringKey::new(&mut self.arena, unsafe { col.value_unchecked(i) });
            map.merge(hash, live_key, Count::single());
            i += 1;
        }

        Ok(())
    }

    fn into_outputter(self) -> crate::operations::unary::Result<Option<Self::Outputter>> {
        self.sender.send((self.arena, self.maps)).unwrap();
        Ok(Some(self.outputter))
    }
}

pub struct GroupOutputter {
    injector: Arc<Injector<PartitionJob>>,
    receiver: Option<mpsc::Receiver<GroupState>>,
    partition_jobs_injected: Arc<AtomicBool>,
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
    pub fn run<S: Sender<RecordBatch>>(mut self, sender: &mut S) -> Result<()> {
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

        sender.send(map_to_record_batch(&result_map)?)?;
        Ok(())
    }
}

impl Outputter<RecordBatch> for GroupOutputter {
    fn output<S: Sender<RecordBatch>>(
        &mut self,
        sender: &mut S,
    ) -> crate::operations::unary::Result<bool> {
        if let Some(rx) = self.receiver.take() {
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

        let steal = self.injector.steal();
        match steal {
            Steal::Success(job) => {
                job.run(sender).map_err(unary::Error::from)?;
            }
            Steal::Empty => {
                if self.partition_jobs_injected.load(Ordering::Relaxed) {
                    // Guaranteed to be no more work to do
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}
