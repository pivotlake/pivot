//! Source stage: a work-stealing queue that hands out every input file exactly
//! once, to whatever worker steals it next. Shared by every dataflow that reads
//! a list of files over the pool: a table's footer load, a whole-object fetch.

use crate::DataFile;
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::sync::Arc;

/// Builds a [`FileInjector`] per worker, all sharing one queue of the files.
#[derive(Clone)]
pub struct FileInjectorFactory<T = DataFile> {
    files: Arc<Injector<T>>,
}

impl<T: Clone> FileInjectorFactory<T> {
    pub fn new(files: &[T]) -> Self {
        let injector = Injector::new();
        for file in files {
            injector.push(file.clone());
        }
        Self {
            files: Arc::new(injector),
        }
    }
}

impl<T: Clone + Send + Sync + 'static> RootChannelFactory<T> for FileInjectorFactory<T> {
    type Receiver = FileInjector<T>;

    fn build(self) -> FileInjector<T> {
        FileInjector { files: self.files }
    }
}

pub struct FileInjector<T = DataFile> {
    files: Arc<Injector<T>>,
}

impl<T> FileInjector<T> {
    /// Wake the pool if this claim was the one that emptied the queue.
    ///
    /// A worker whose `try_finish` saw the queue non-empty (so it did not
    /// decrement its finish counter) can park in the window before it re-checks,
    /// just as another worker claims the final file. Claiming a file emits its
    /// result only to the worker consuming it, so nothing else wakes that
    /// parked worker to run its finish. Broadcasting when the queue empties wakes
    /// it (and, via the wake-count bump, any worker mid-park) so every worker
    /// reaches its finish and the stage's sibling counter can drain to zero.
    /// Without it the query hangs with the pool parked one step short of done.
    ///
    /// The queue is filled once at construction and only drains, so the claim
    /// that empties it always sees `is_empty` here; no separate counter is needed.
    /// Two final claims racing may both observe it and broadcast twice, which
    /// costs only a redundant wake.
    fn wake_if_drained(&self) {
        if self.files.is_empty() {
            dispatch::waker::waker_set().notify_all();
        }
    }
}

impl<T: Send + Sync> Receiver<T> for FileInjector<T> {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn try_recv(&self) -> Option<T> {
        // Drive the fetch from the eager `run_cpu_work` path (which calls
        // `try_recv`), not only the worker's idle `steal` path; otherwise each
        // worker fetches one file at a time and a large catch-up (many new
        // files committed since the last query) serialises into ~1s. A single
        // non-spinning attempt keeps the hot loop from spinning on `Steal::Retry`;
        // the next iteration retries.
        match self.files.steal() {
            Steal::Success(file) => {
                self.wake_if_drained();
                Some(file)
            }
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<T> {
        loop {
            match self.files.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(file) => {
                    self.wake_if_drained();
                    return Some(file);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataFileLocation, FileRef, ObjectPath};
    use dispatch::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};
    use std::path::PathBuf;

    /// The injector only ever hands these out, so they need no file behind them.
    fn data_file(name: &str) -> DataFile {
        DataFile {
            file: FileRef {
                path: ObjectPath::new(name),
                size: 0,
            },
            source: DataFileLocation::Local(PathBuf::from(name)),
        }
    }

    /// Install a waker this thread can read the wake count off, standing in for
    /// the pool a worker would be parked on.
    fn observable_waker() -> Arc<WorkerWaker> {
        let waker = Arc::new(WorkerWaker::new(1));
        init_worker_waker(&waker);
        init_waker_set(WakerSet::new(vec![waker.clone()], 1));
        waker
    }

    /// Regression: draining the queue has to wake the pool.
    ///
    /// A worker whose `try_finish` saw the queue non-empty returns without
    /// decrementing its stage's sibling counter. If a peer then claims the last
    /// file, that worker parks on an empty queue still owing its decrement, and
    /// claiming a file only sends its result to the worker consuming it, so
    /// nothing else would wake it. Without this broadcast the sibling counter
    /// never reaches zero and a table load hangs with the pool parked one step
    /// short of done.
    #[test]
    fn claiming_the_last_file_wakes_the_pool() {
        let waker = observable_waker();
        let injector = FileInjectorFactory::new(&[data_file("a"), data_file("b")]).build();

        let before = waker.wake_count();
        injector.steal().expect("first file");
        let after_first = waker.wake_count();
        injector.steal().expect("last file");
        let after_last = waker.wake_count();

        assert_eq!(
            after_first, before,
            "a claim that leaves work behind should wake nobody"
        );
        assert!(
            after_last > after_first,
            "claiming the last file must wake the pool"
        );
    }

    /// The eager `try_recv` claim path must broadcast on drain too: it is the one
    /// the fetcher's `run_cpu_work` actually uses.
    #[test]
    fn the_eager_claim_path_also_wakes_the_pool_on_drain() {
        let waker = observable_waker();
        let injector = FileInjectorFactory::new(&[data_file("only")]).build();

        let before = waker.wake_count();
        // `try_recv` is deliberately single-shot, so retry past a lost steal race.
        (0..1_000)
            .find_map(|_| injector.try_recv())
            .expect("only file");

        assert!(
            waker.wake_count() > before,
            "draining via try_recv must wake the pool"
        );
    }
}
