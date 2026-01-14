use crate::pipeline::Pipeline;
use crate::worker::Worker;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Barrier, OnceLock};
use std::thread::JoinHandle;
use tikv_jemallocator::Jemalloc;
use tracing::info;

mod api;
mod env;
mod functions;
mod identified;
mod input;
mod io;
mod memory_source;
mod operations;
mod pipeline;
mod record_batch_metadata;
pub mod table;
mod worker;

pub use api::*;
pub use functions::*;

pub use operations::OrderBy;
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
    use crate::table::Table;
    use crate::{Contains, OrderBy, PipelineSpec, init};
    use arrow::compute::concat;
    use arrow_array::{
        Array, ArrayRef, Int64Array, RecordBatch, StringArray, StringViewArray, UInt64Array,
    };
    use parquetd::Projection;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};
    use std::vec;
    use tracing_subscriber::{EnvFilter, fmt};

    use crate::api::MemoryFeed;
    use std::sync::{Arc, LazyLock};
    use std::thread::sleep;
    use tikv_jemalloc_ctl::{epoch, stats};

    fn jemalloc_snapshot() {
        epoch::advance().unwrap(); // refresh stats
        eprintln!("allocated: {}", stats::allocated::read().unwrap());
        eprintln!("active:    {}", stats::active::read().unwrap());
        eprintln!("resident:  {}", stats::resident::read().unwrap());
        eprintln!("mapped:    {}", stats::mapped::read().unwrap());
        eprintln!("retained:    {}", stats::retained::read().unwrap());
    }

    static SOURCE_DIRECTORY: LazyLock<PathBuf> =
        LazyLock::new(|| PathBuf::from(std::env::var("SOURCE_DIRECTORY").unwrap()));

    // Check the performance of "SELECT count(*) FROM hits WHERE URL LIKE '%google%'"
    fn run_query_20(table: Arc<Table>) {
        let feed = MemoryFeed::new();

        PipelineSpec::new()
            .table_input(table, Some(Projection::columns([13])))
            .filter_builder(|| {
                let mut contains = Contains::new("google");
                Box::new(move |batch| {
                    let col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    contains.run(col)
                })
            })
            .count(feed.output())
            .execute();

        let results = feed.collect();
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
    fn run_query_23(table: Arc<Table>) {
        let pipeline_feed = MemoryFeed::new();

        PipelineSpec::new()
            .table_input(table.clone(), Some(Projection::columns([4, 13])))
            .filter_builder(|| {
                let mut contains = Contains::new("google");
                Box::new(move |batch| {
                    let col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    contains.run(col)
                })
            })
            .order_by_limit([OrderBy::new(0, false, false)], 10, pipeline_feed.output())
            .execute();

        let output_feed = MemoryFeed::new();

        PipelineSpec::new()
            .memory_input(pipeline_feed.source())
            .materialize(table.clone(), None)
            .order_by_limit([OrderBy::new(4, false, false)], 10, output_feed.output())
            .execute();

        let batches = output_feed.collect();

        let concatenated_array: ArrayRef = concat(
            &batches
                .iter()
                .map(|batch| batch.column(0).as_ref())
                .collect::<Vec<_>>(),
        )
        .expect("Failed to concatenate column 0");

        let actual_ids = concatenated_array
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Column 0 is not an Int64Array");

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

    // Check the performance of "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;"
    fn run_query_33(table: Arc<Table>) {
        let pipeline_feed = MemoryFeed::new();

        PipelineSpec::new()
            .table_input(table, Some(Projection::columns([13])))
            .group_by_count(0, pipeline_feed.output())
            .execute();

        let output_feed = MemoryFeed::new();
        PipelineSpec::new()
            .memory_input(pipeline_feed.source())
            .order_by_limit([OrderBy::new(1, true, false)], 10, output_feed.output())
            .execute();

        let results = output_feed.collect();
        assert_eq!(results.len(), 1);
        let result = &results[0];

        let expected = RecordBatch::try_from_iter(vec![
            (
                "key",
                Arc::new(StringArray::from(vec![
                    "http://liver.ru/belgorod/page/1006.jки/доп_приборы",
                    "http://kinopoisk.ru",
                    "http://bdsm_po_yers=0&with_video",
                    "http://video.yandex",
                    "http://smeshariki.ru/region",
                    "http://auto_fiat_dlya-bluzki%2F8536.30.18&he=900&with",
                    "http://liver.ru/place_rukodel=365115eb7bbb90",
                    "http://kinopoisk.ru/vladimir.irr.ru",
                    "http://video.yandex.ru/search/?jenre=50&s_yers",
                    "http://tienskaia-moda",
                ])) as ArrayRef,
            ),
            (
                "value",
                Arc::new(UInt64Array::from(vec![
                    3288173, 1625250, 791465, 582400, 514984, 507995, 359893, 354690, 318979,
                    289355,
                ])) as ArrayRef,
            ),
        ])
        .unwrap();

        assert_eq!(result, &expected);
    }

    #[rstest]
    fn test_dispatcher() {
        init();

        fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_writer(std::io::stdout)
            .init();

        let table =
            Arc::new(Table::from_directory(&SOURCE_DIRECTORY).expect("Could not create source"));

        for _ in 0..get_env_var_with_default("QUERY_TEST_COUNT", 20) {
            let start = Instant::now();
            match get_env_var_with_default("QUERY", 20) {
                20 => run_query_20(table.clone()),
                23 => run_query_23(table.clone()),
                33 => run_query_33(table.clone()),
                _ => panic!("No such query"),
            }
            println!("Elapsed {:?}", start.elapsed().as_millis());
            jemalloc_snapshot();
            if let Ok(a) = std::env::var("SLEEP") {
                sleep(Duration::from_secs(a.parse().unwrap()));
            }
        }
    }
}
