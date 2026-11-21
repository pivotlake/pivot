//! The background tasks the datastore runs for itself, and how they stop.
//!
//! The datastore spawns each of its maintenance loops through one
//! [`MaintenanceTasks`], which hands the loop a [`StopSignal`]. The loop waits
//! on the signal beside whatever else wakes it (a timer tick, a command) and
//! returns once the signal is raised, but only between sweeps: a sweep that
//! has started is awaited to its end, since its body runs on the blocking
//! pool, where dropping its handle would not stop it.
//! [`MaintenanceTasks::stop`] raises the signal and waits for every loop to
//! return; [`MaintenanceTasks::request_stop`] raises it and waits for none,
//! for a caller that cannot count on a sweep in flight ever finishing.

use std::future::Future;
use std::sync::Mutex;

use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Why a maintenance task was not spawned: its datastore's maintenance has
/// been stopped, and a stopped datastore runs none again.
#[derive(Debug, thiserror::Error)]
#[error("the datastore's background maintenance has been stopped")]
pub(crate) struct MaintenanceStopped;

/// The maintenance tasks of one datastore: spawned through it, stopped and
/// awaited through it.
#[derive(Debug)]
pub(crate) struct MaintenanceTasks {
    stop: watch::Sender<bool>,
    /// The tasks spawned so far; `None` once [`stop`](Self::stop) or
    /// [`request_stop`](Self::request_stop) has taken them, after which
    /// nothing is spawned.
    running: Mutex<Option<Vec<JoinHandle<()>>>>,
}

impl MaintenanceTasks {
    pub(crate) fn new() -> Self {
        let (stop, _) = watch::channel(false);
        Self {
            stop,
            running: Mutex::new(Some(Vec::new())),
        }
    }

    /// Spawn onto the ambient runtime the task `build_task` makes of this
    /// datastore's stop signal. The task must return once the signal is
    /// raised, having finished whatever it had in flight.
    pub(crate) fn spawn<Task>(
        &self,
        build_task: impl FnOnce(StopSignal) -> Task,
    ) -> Result<(), MaintenanceStopped>
    where
        Task: Future<Output = ()> + Send + 'static,
    {
        let mut running = self.running.lock().unwrap();
        let tasks = running.as_mut().ok_or(MaintenanceStopped)?;
        let signal = StopSignal(self.stop.subscribe());
        tasks.push(tokio::spawn(build_task(signal)));
        Ok(())
    }

    /// Raise the stop signal and return at once, waiting for no task: each
    /// starts no further sweep, and one in flight is left to end by itself.
    /// Nothing is spawned through this afterwards, and a later
    /// [`stop`](Self::stop) finds no task to wait for. For a caller that
    /// cannot count on a sweep in flight ever finishing.
    pub(crate) fn request_stop(&self) {
        self.running.lock().unwrap().take();
        self.stop.send_replace(true);
    }

    /// Raise the stop signal and wait for every task spawned through this to
    /// return. Nothing is spawned through this afterwards. Meant to be called
    /// once: a second call finds no task to wait for and returns at once.
    pub(crate) async fn stop(&self) {
        let tasks = self.running.lock().unwrap().take();
        self.stop.send_replace(true);
        for task in tasks.into_iter().flatten() {
            if let Err(error) = task.await {
                tracing::warn!(error = %error, "maintenance task panicked");
            }
        }
    }
}

/// Tells a maintenance loop that its datastore is stopping.
#[derive(Clone, Debug)]
pub(crate) struct StopSignal(watch::Receiver<bool>);

impl StopSignal {
    /// A signal nothing ever raises, for a loop run on its own.
    pub(crate) fn never() -> Self {
        let (_, raised) = watch::channel(false);
        Self(raised)
    }

    pub(crate) fn is_raised(&self) -> bool {
        *self.0.borrow()
    }

    /// Wait until the signal is raised. A signal whose
    /// [`MaintenanceTasks`] was dropped unraised never is.
    pub(crate) async fn wait_until_raised(&mut self) {
        if self.0.wait_for(|raised| *raised).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

    fn build_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap()
    }

    /// The sweep outlasts the raising of the stop signal, so only a stop that
    /// waits for it can find it finished.
    #[test]
    fn stop_waits_for_the_sweep_in_flight() {
        let runtime = build_runtime();
        let tasks = MaintenanceTasks::new();
        let sweep_finished = Arc::new(AtomicBool::new(false));
        let mark_finished = Arc::clone(&sweep_finished);
        let (report_started, sweep_started) = tokio::sync::oneshot::channel();
        let _runtime = runtime.enter();
        tasks
            .spawn(|stop| async move {
                let sweep = tokio::task::spawn_blocking(move || {
                    report_started.send(()).unwrap();
                    while !stop.is_raised() {
                        std::thread::yield_now();
                    }
                    std::thread::sleep(Duration::from_millis(50));
                    mark_finished.store(true, Ordering::SeqCst);
                });
                sweep.await.unwrap();
            })
            .unwrap();
        runtime.block_on(sweep_started).unwrap();

        runtime.block_on(tasks.stop());

        assert!(sweep_finished.load(Ordering::SeqCst));
    }

    /// The task never returns, so only a request that waits for nothing can
    /// return, and the task tells that the signal reached it.
    #[test]
    fn request_stop_raises_the_signal_without_waiting_and_ends_spawning() {
        let runtime = build_runtime();
        let tasks = MaintenanceTasks::new();
        let (report_signal, signal_seen) = tokio::sync::oneshot::channel();
        let _runtime = runtime.enter();
        tasks
            .spawn(|mut stop| async move {
                stop.wait_until_raised().await;
                report_signal.send(()).unwrap();
                std::future::pending::<()>().await;
            })
            .unwrap();

        tasks.request_stop();

        runtime.block_on(signal_seen).unwrap();
        assert!(tasks.spawn(|_stop| async {}).is_err());
    }

    #[test]
    fn nothing_is_spawned_after_stop() {
        let runtime = build_runtime();
        let tasks = MaintenanceTasks::new();
        runtime.block_on(tasks.stop());

        let _runtime = runtime.enter();
        let spawned = tasks.spawn(|_stop| async {});

        assert!(spawned.is_err());
    }
}
