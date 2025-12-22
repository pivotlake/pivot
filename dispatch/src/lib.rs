use crate::pipeline::PipelineConfig;
use crate::worker::Worker;
use crossbeam_deque::Injector;
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use tracing::info;

mod env;
mod operations;
mod pipeline;
mod source;
mod worker;

pub use operations::*;
/// The architecture is mainly based on the paper: https://db.in.tum.de/~leis/papers/morsels.pdf
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
/// in a Source -> Filter -> Count query, pipelines will be sent to all workers. Each pipeline will
/// contain the Count operation will have a shared `Barrier` and `AtomicUsize` with the other
/// pipelines. Each worker will run its own instance of the Pipeline, continuously updating an
/// internal count. Once the pipeline is complete, each count operation will updating the atomic and
/// wait on their barrier, where the leader will output it
///
/// The `Dispatcher` controls spinning up workers and dispatching work. The Dispatcher is the main
/// entry-point for working with the `dispatch` library. Despite its name, the dispatcher does not
///  "run" actively, but is instead called by its workers to prevent unneeded context switches.
pub struct Dispatcher {
    pipelines: Injector<PipelineConfig>,
}

impl Dispatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Dispatcher {
            pipelines: Default::default(),
        })
    }

    /// Enter a new pipeline for one of the workers to execute. The next worker to be "free"
    /// will execute it.
    pub fn push_pipeline(&self, pipeline_config: PipelineConfig) {
        self.pipelines.push(pipeline_config);
    }

    /// Start a worker per CPU core, and wait for all workers to begin-
    /// this method should ideally only be called once.
    pub fn start_worker_per_cpu(self: &Arc<Self>) -> Vec<JoinHandle<()>> {
        let cores = core_affinity::get_core_ids().unwrap();
        let mut threads = vec![];
        info!("Starting workers...");
        let barrier = Arc::new(Barrier::new(cores.len() + 1));
        for core in cores {
            threads.push(Worker::create(core, self.clone(), barrier.clone()));
        }
        barrier.wait();
        info!("All workers have begun...");
        threads
    }
}

#[cfg(test)]
mod tests {
    use crate::Dispatcher;
    use crate::env::get_env_var_with_default;
    use crate::operations::{Count, Filter, Operation};
    use crate::pipeline::PipelineConfig;
    use crate::source::Source;
    use arrow::compute::like;
    use arrow_array::{Scalar, StringViewArray};
    use parquetd::Projection;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier, LazyLock};
    use std::time::Instant;
    use std::vec;
    use tracing_subscriber::{EnvFilter, fmt};

    static SOURCE_DIRECTORY: LazyLock<PathBuf> =
        LazyLock::new(|| PathBuf::from(std::env::var("SOURCE_DIRECTORY").unwrap()));

    // Check the performance of "SELECT count(*) FROM hits WHERE URL LIKE '%google%'"
    fn run_query_filter_count(dispatcher: Arc<Dispatcher>, workers: usize) {
        let start = Instant::now();
        let source = Arc::new(Source::from_directory(&SOURCE_DIRECTORY));
        let barrier = Arc::new(Barrier::new(workers + 1));

        let count_barrier = Arc::new(Barrier::new(workers));
        let shared_count = Arc::new(AtomicUsize::new(0));

        for _ in 0..workers {
            dispatcher.push_pipeline(PipelineConfig {
                source: source.clone(),
                projection: Some(Projection::columns([13])),
                operations: vec![(
                    1,
                    Box::new(Count::new(1, shared_count.clone(), count_barrier.clone()))
                        as Box<dyn Operation>,
                )]
                .into_iter()
                .collect(),
                initial_operations: vec![Box::new(Filter::new(0, move |batch| {
                    let col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    let pat = Scalar::new(StringViewArray::from(vec![Some("%google%")]));
                    like(&col, &pat).unwrap()
                }))],
                publishers_to_subscribers: vec![((0, vec![1]))].into_iter().collect(),
                barrier: barrier.clone(),
            })
        }
        barrier.wait();
    }

    #[rstest]
    fn test_dispatcher() {
        fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_writer(std::io::stdout)
            .init();

        let dispatcher = Dispatcher::new();
        let handles = dispatcher.start_worker_per_cpu();

        for _ in 0..get_env_var_with_default("QUERY_TEST_COUNT", 20) {
            let start = Instant::now();
            run_query_filter_count(dispatcher.clone(), handles.len());
            println!("Elapsed {:?}", start.elapsed().as_millis())
        }
    }
}
