use crate::pipeline::Pipeline;
use crate::worker::Worker;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Barrier, OnceLock};
use std::thread::JoinHandle;
use tikv_jemallocator::Jemalloc;
use tracing::info;

mod env;
mod functions;
mod identified;
mod input;
mod io;
mod memory_source;
mod operations;
mod pipeline;
mod record_batch_metadata;
mod table;
mod worker;

pub use functions::Contains;
pub use memory_source::{MemoryInput, MemoryOutput};
pub use operations::{
    ConsumeContext, Count, Filter, Materializer, Operation, OrderBy, OrderByLimit, Output,
    OutputOperation, PipelineBreaker, StdOutOutput,
};
pub use table::{RowGroupMetadataHandle, Table, TableInput, TableSource};
use worker::PipelineHandle;

#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8] = b"percpu_arena:percpu,oversize_threshold:0,\
muzzy_decay_ms:5000,dirty_decay_ms:10000,\
lg_extent_max_active_fit:8,background_thread:true\0";


#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub static DISPATCHER: OnceLock<Dispatcher> = OnceLock::new();

pub fn dispatcher() -> &'static Dispatcher {
    DISPATCHER
        .get()
        .expect("Dispatcher has not been initialized")
}

pub fn init() {
    DISPATCHER.get_or_init(Dispatcher::new);
}

/// The architecture is based on the paper: https://db.in.tum.de/~leis/papers/morsels.pdf
/// Where the basic idea is to have a thread per core running a worker which continuously requests
/// work from a global dispatcher.
///
/// Each worker is meant to run in large part "alone" on pipelines - its operations are all
/// parallelism-aware, and workers synchronize with each other only at the deepest level
/// (a running `Operation`).
///
/// Workers are meant to largely run on their own output, thus ensuring locality (both cache and in the future, NUMA nodes)
///
/// The main execution is done through sending `Pipelines` to workers. For a given query, pipelines
/// are sent out to all workers, which contain operations that can synchronize the work. For example,
/// in a Source -> Filter -> Count query, pipelines will be sent to all workers; Each pipeline will
/// contain the Count operation which has a shared `Barrier` and `AtomicUsize` with the other
/// pipelines. Each worker will run its own instance of the Pipeline, continuously updating an
/// internal count. Once the pipeline is complete, each count operation will update the atomic and
/// wait on their barrier, where the leader will output it.
///
/// All pipelines and operations have identifiers. These identifiers are unique within a single worker,
/// but *shared* across workers. Every "sibling" pipeline/operation has the same identifier; this is
/// so to allow work stealing.
///
/// The `Dispatcher` controls spinning up workers and dispatching work. The Dispatcher is the main
/// entry-point for working with the `dispatch` library. Despite its name, the dispatcher does not
///  "run" actively, but is instead called by its workers to prevent unneeded context switches.
pub struct Dispatcher {
    pipeline_senders: Vec<Sender<PipelineHandle>>,
    handles: Vec<JoinHandle<()>>,
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
    }

impl Dispatcher {
    pub fn new() -> Self {
        let cores = core_affinity::get_core_ids().unwrap();
        let worker_count: usize = std::env::var("WORKER_COUNT")
            .unwrap_or(cores.len().to_string())
            .parse()
            .expect("Non integer WORKER_COUNT given");
        if worker_count > cores.len() {
            panic!("Invalid worker count (worker count must be below number of cores");
        }

        let cores: Vec<_> = cores.into_iter().take(worker_count).collect();
        let mut threads = vec![];
        info!("Starting workers...");
        let barrier = Arc::new(Barrier::new(cores.len() + 1));
        let mut senders = vec![];
        for core in cores {
            let (tx, rx) = channel();
            senders.push(tx);
            threads.push(Worker::create(core, rx, barrier.clone()));
        }
        barrier.wait();
        info!("All workers have begun...");

        Dispatcher {
                pipeline_senders: senders,
            handles: threads,
        }
    }

    pub fn workers(&self) -> usize {
        self.handles.len()
    }

    /// Enter a new pipeline for all the workers to execute.
    pub fn push_pipeline<F: FnMut() -> Pipeline>(&self, mut pipeline_builder: F) {
        let mut cb_workers: Vec<_> = (0..self.pipeline_senders.len())
            .map(|_| crossbeam_deque::Worker::new_fifo())
            .collect();
        let mut stealers: Vec<_> = (0..self.pipeline_senders.len())
            .map(|_| cb_workers.iter().map(|w| w.stealer()).collect())
            .collect();
        for sender in &self.pipeline_senders {
            sender
                .send(PipelineHandle::new(
                    pipeline_builder(),
                    cb_workers.pop().unwrap(),
                    stealers.pop().unwrap(),
                ))
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::get_env_var_with_default;
    use crate::identified::Identified;
    use crate::memory_source::{MemoryInput, MemoryOutput, MemorySource};
    use crate::operations::{Count, Filter, Operation};
    use crate::pipeline::Pipeline;
    use crate::table::{Table, TableInput, TableSource};
    use crate::{
        Contains, Dispatcher, Materializer, OrderBy, OrderByLimit, Output, PipelineBreaker,
    };
    use arrow::compute::concat;
    use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
    use parquetd::Projection;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;
    use std::{mem, vec};
    use tracing_subscriber::{EnvFilter, fmt};

    use std::sync::mpsc::channel;
    use std::sync::{Arc, Barrier, Condvar, LazyLock, Mutex};

    struct SharedState {
        batches: Vec<RecordBatch>,
        active_writers: usize,
    }

    pub struct TestOutputWaiter {
        inner: Arc<(Mutex<SharedState>, Condvar)>,
    }

    pub struct TestOutputWriter {
        inner: Arc<(Mutex<SharedState>, Condvar)>,
    }

    pub fn create_test_output() -> (TestOutputWriter, TestOutputWaiter) {
        let inner = Arc::new((
            Mutex::new(SharedState {
                batches: Vec::new(),
                active_writers: 1,
            }),
            Condvar::new(),
        ));

        (
            TestOutputWriter {
                inner: Arc::clone(&inner),
            },
            TestOutputWaiter { inner },
        )
    }

    impl TestOutputWaiter {
        pub fn wait_for_finish(self) -> Vec<RecordBatch> {
            let (lock, cvar) = &*self.inner;
            let mut state = lock.lock().unwrap();

            // Wait while there are still active writers
            state = cvar
                .wait_while(state, |s| s.active_writers > 0)
                .expect("Lock poisoned");

            mem::take(&mut state.batches)
        }
    }

    impl Output for TestOutputWriter {
        fn write(&mut self, batch: RecordBatch) {
            let (lock, _) = &*self.inner;
            lock.lock().unwrap().batches.push(batch);
        }

        fn finish(&mut self) {
            let (lock, cvar) = &*self.inner;
            let mut state = lock.lock().unwrap();
            state.active_writers -= 1;

            if state.active_writers == 0 {
                cvar.notify_all();
            }
        }
    }

    impl Clone for TestOutputWriter {
        fn clone(&self) -> Self {
            let (lock, _) = &*self.inner;
            lock.lock().unwrap().active_writers += 1;
            Self {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    static SOURCE_DIRECTORY: LazyLock<PathBuf> =
        LazyLock::new(|| PathBuf::from(std::env::var("SOURCE_DIRECTORY").unwrap()));

    // Check the performance of "SELECT count(*) FROM hits WHERE URL LIKE '%google%'"
    fn run_query_20(table: Arc<Table>, dispatcher: Arc<Dispatcher>, workers: usize) {
        let count_barrier = Arc::new(Barrier::new(workers));
        let shared_count = Arc::new(AtomicUsize::new(0));
        let pipeline_count = Arc::new(AtomicUsize::new(workers));
        let table_source = Arc::new(TableSource::from(&table));
        let (output, reader) = create_test_output();
        let mut output = Box::new(output);

        dispatcher.push_pipeline(|| {
            let mut contains = Contains::new("google");
            Pipeline::new(
                1,
                vec![Identified::new(
                    0,
                    Box::new(TableInput::new(
                        table_source.clone(),
                        Some(Projection::columns([13])),
                    )),
                )],
                vec![Identified::new(
                    1,
                    Box::new(Filter::new(move |batch| {
                        let col = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringViewArray>()
                            .unwrap();
                        contains.run(col)
                    })) as Box<dyn Operation>,
                )],
                vec![Identified::new(
                    2,
                    Box::new(Count::new(
                        shared_count.clone(),
                        count_barrier.clone(),
                        output.clone(),
                    )) as Box<dyn PipelineBreaker>,
                )],
                vec![(0, vec![1]), (1, vec![2])].into_iter().collect(),
                pipeline_count.clone(),
            )
        });

        output.finish();
        let results = reader.wait_for_finish();
        assert_eq!(results.len(), 1);
        let column = results[0].column(0);
        assert_eq!(
            column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            15911
        )
    }

    // SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;
    fn run_query_23(table: Arc<Table>, dispatcher: Arc<Dispatcher>, workers: usize) {
        let (output, waiter) = create_test_output();
        let mut output = Box::new(output);

        let record_batch_source = Arc::new(MemorySource::new());
        let mut record_batch_output = Box::new(MemoryOutput::new(record_batch_source.clone()));

        let table_source = Arc::new(TableSource::from(&table));
        let (tx, rx) = channel();
        let pipeline_count = Arc::new(AtomicUsize::new(workers));

        let mut rx_opt = Some(rx);
        dispatcher.push_pipeline(|| {
            let mut contains = Contains::new("google");
            Pipeline::new(
                0,
                vec![Identified::new(
                    0,
                    Box::new(TableInput::new(
                        table_source.clone(),
                        Some(Projection::columns([4, 13])),
                    )),
                )],
                vec![Identified::new(
                    1,
                    Box::new(Filter::new(move |batch| {
                        let col = batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<StringViewArray>()
                            .unwrap();
                        contains.run(col)
                    })) as Box<dyn Operation>,
                )],
                vec![Identified::new(
                    2,
                    Box::new(OrderByLimit::new(
                        10,
                        record_batch_output.clone(),
                        tx.clone(),
                        mem::take(&mut rx_opt),
                        vec![OrderBy::new(0, false, false)],
                    )),
                )],
                vec![(0, vec![1]), (1, vec![2])].into_iter().collect(),
                pipeline_count.clone(),
            )
        });
        drop(tx);

        let (tx, rx) = channel();
        let mut rx_opt = Some(rx);
        let pipeline_count = Arc::new(AtomicUsize::new(workers));
        dispatcher.push_pipeline(|| {
            Pipeline::new(
                1,
                vec![Identified::new(
                    0,
                    Box::new(MemoryInput::new(record_batch_source.clone())),
                )],
                vec![Identified::new(
                    1,
                    Box::new(Materializer::new(None, table.clone())) as Box<dyn Operation>,
                )],
                vec![Identified::new(
                    2,
                    Box::new(OrderByLimit::new(
                        10,
                        output.clone(),
                        tx.clone(),
                        mem::take(&mut rx_opt),
                        vec![OrderBy::new(4, false, false)],
                    )),
                )],
                vec![(0, vec![1]), (1, vec![2])].into_iter().collect(),
                pipeline_count.clone(),
            )
        });
        drop(tx);

        output.finish();
        record_batch_output.finish();

        let batches = waiter.wait_for_finish();
        if batches.len() == 0 {
            panic!("Unexpected empty batches");
        }

        let expected_watch_ids: Vec<i64> = vec![
            7675678523794456216,
            6147260061318473746,
            5972689683963797854,
            8008688361303225116,
            7436461208655480623,
            5564518777317455184,
            7381524648140977766,
            8614832219462424183,
            7168314068394418899,
            8235569889353442646,
        ];

        let column_segments: Vec<&dyn Array> = batches
            .iter()
            .map(|batch| batch.column(0).as_ref())
            .collect();

        // 3. Concatenate all segments into one single Array
        let concatenated_array = concat(&column_segments).expect("Failed to concatenate column 0");
        let actual_ids = concatenated_array
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Column 0 is not an Int64Array");

        println!("Actual ids: {:?}", actual_ids);

        assert_eq!(
            actual_ids.len(),
            expected_watch_ids.len(),
            "Total row count across all batches mismatch"
        );

        for (i, &expected_val) in expected_watch_ids.iter().enumerate() {
            assert_eq!(
                actual_ids.value(i),
                expected_val,
                "Mismatch at global row index {}",
                i
            );
        }
    }

    #[rstest]
    fn test_dispatcher() {
        fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_writer(std::io::stdout)
            .init();

        let (dispatcher, handles) = Dispatcher::new();
        let table =
            Arc::new(Table::from_directory(&SOURCE_DIRECTORY).expect("Could not create source"));

        for _ in 0..get_env_var_with_default("QUERY_TEST_COUNT", 20) {
            let start = Instant::now();
            match get_env_var_with_default("QUERY", 20) {
                20 => run_query_20(table.clone(), dispatcher.clone(), handles.len()),
                23 => run_query_23(table.clone(), dispatcher.clone(), handles.len()),
                _ => panic!("No such query"),
            }
            println!("Elapsed {:?}", start.elapsed().as_millis())
        }
    }
}
