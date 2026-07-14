//! Bounded async map stage whose completions re-enter the owning worker.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};

use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

use super::{Result, Unary, UnaryFactory};
use crate::WorkStatus;
use crate::operations::channels::Sender;
use crate::worker::WorkerWaker;

static ASYNC_RUNTIME: OnceLock<Runtime> = OnceLock::new();
static NEXT_TASK_ID: AtomicUsize = AtomicUsize::new(0);

fn async_runtime() -> &'static Runtime {
    ASYNC_RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("pivot-dispatch-async")
            .build()
            .expect("failed to create dispatch async runtime")
    })
}

/// Builds one worker-local async map with a bounded number of futures.
pub struct AsyncMapFactory<F> {
    map: F,
    max_in_flight: usize,
    waker: Arc<WorkerWaker>,
}

impl<F> AsyncMapFactory<F> {
    pub(crate) fn new(map: F, max_in_flight: usize, waker: Arc<WorkerWaker>) -> Self {
        Self {
            map,
            max_in_flight,
            waker,
        }
    }
}

impl<I, O, F, Fut> UnaryFactory<I, O> for AsyncMapFactory<F>
where
    I: 'static,
    O: Send + 'static,
    F: FnMut(I) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    type Unary = AsyncMap<F, O>;

    fn build_unary(self) -> Self::Unary {
        let (completion_tx, completion_rx) = mpsc::channel();
        AsyncMap {
            map: self.map,
            max_in_flight: self.max_in_flight,
            completion_tx,
            completion_rx,
            tasks: HashMap::new(),
            waker: self.waker,
        }
    }
}

enum Completion<O> {
    Finished { id: usize, result: Result<O> },
    Failed { id: usize },
}

struct CompletionGuard<O> {
    id: usize,
    completion_tx: mpsc::Sender<Completion<O>>,
    waker: Arc<WorkerWaker>,
    completed: bool,
}

impl<O> CompletionGuard<O> {
    fn finish(mut self, result: Result<O>) {
        self.completed = true;
        let _ = self.completion_tx.send(Completion::Finished {
            id: self.id,
            result,
        });
        self.waker.notify();
    }
}

impl<O> Drop for CompletionGuard<O> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.completion_tx.send(Completion::Failed { id: self.id });
            self.waker.notify();
        }
    }
}

/// One worker's async futures and completion queue.
pub struct AsyncMap<F, O> {
    map: F,
    max_in_flight: usize,
    completion_tx: mpsc::Sender<Completion<O>>,
    completion_rx: mpsc::Receiver<Completion<O>>,
    tasks: HashMap<usize, JoinHandle<()>>,
    waker: Arc<WorkerWaker>,
}

impl<I, O, F, Fut> Unary<I, O> for AsyncMap<F, O>
where
    O: Send + 'static,
    F: FnMut(I) -> Fut + Send,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    fn consume<S: Sender<O>>(&mut self, item: I, _sender: &mut S) -> Result<()> {
        let future = (self.map)(item);
        let id = NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed);
        let guard = CompletionGuard {
            id,
            completion_tx: self.completion_tx.clone(),
            waker: self.waker.clone(),
            completed: false,
        };
        let task = async_runtime().spawn(async move {
            let result = future.await;
            guard.finish(result);
        });
        self.tasks.insert(id, task);
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.tasks.len() < self.max_in_flight
    }

    fn ready_to_finish(&self) -> bool {
        self.tasks.is_empty()
    }

    fn run<S: Sender<O>>(&mut self, sender: &mut S) -> Result<WorkStatus> {
        let completion = match self.completion_rx.try_recv() {
            Ok(completion) => completion,
            Err(mpsc::TryRecvError::Empty) => return Ok(WorkStatus::Pending),
            Err(mpsc::TryRecvError::Disconnected) => {
                unreachable!("async map owns a completion sender")
            }
        };
        match completion {
            Completion::Finished { id, result } => {
                self.tasks.remove(&id);
                sender.send(result?)?;
            }
            Completion::Failed { id } => {
                self.tasks.remove(&id);
                return Err(super::Error::Operator(Box::new(AsyncTaskFailed)));
            }
        }
        Ok(WorkStatus::Ran)
    }
}

impl<F, O> Drop for AsyncMap<F, O> {
    fn drop(&mut self) {
        let tasks = self.tasks.drain().map(|(_, task)| task).collect::<Vec<_>>();
        // Cancellation and errors must not let an upload outlive its transaction
        // rollback. Await every future before this dataflow releases its output
        // sender and reports completion to the caller. Letting tasks drain also
        // covers filesystem operations that an executor cannot cancel once the
        // kernel has accepted them.
        async_runtime().block_on(async {
            for task in tasks {
                let _ = task.await;
            }
        });
    }
}

#[derive(Debug, thiserror::Error)]
#[error("dispatch async task panicked or was cancelled")]
struct AsyncTaskFailed;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::{Dispatch, values_input};

    #[test]
    fn maps_futures_and_bounds_concurrency() {
        let dispatch = Dispatch::spin_up(1, 16, None);
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut output = values_input(dispatch.dispatcher(), 0..8)
            .map_each_async(2, {
                let active = active.clone();
                let peak = peak.clone();
                move |value| {
                    let active = active.clone();
                    let peak = peak.clone();
                    async move {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(2)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(value * 2)
                    }
                }
            })
            .collect()
            .unwrap();
        output.sort_unstable();

        assert_eq!(output, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        assert!(peak.load(Ordering::SeqCst) <= 2);
        dispatch.exit();
    }

    #[test]
    fn cancellation_drains_started_futures_before_closing() {
        let dispatch = Dispatch::spin_up(1, 16, None);
        let started = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        let handle = values_input(dispatch.dispatcher(), [1])
            .map_each_async(1, {
                let started = started.clone();
                let finished = finished.clone();
                move |value| {
                    let started = started.clone();
                    let finished = finished.clone();
                    async move {
                        started.store(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        finished.store(1, Ordering::SeqCst);
                        Ok(value)
                    }
                }
            })
            .execute();
        while started.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }

        handle.cancel();
        handle.collect().unwrap();

        assert_eq!(finished.load(Ordering::SeqCst), 1);
        dispatch.exit();
    }
}
