//! Regression test for the silent build-failure deadlock.
//!
//! Every stage's `siblings_left` counter is initialised to the full worker
//! count when a dataflow is *specced*, not when it is built. If one worker's
//! `DataFlowBuilder::build` then fails (e.g. a build-time panic), that worker
//! used to just log a warning and skip the dataflow — while the remaining
//! workers processed all of the data via work stealing and then waited
//! forever for the missing sibling's `try_finish` decrements: every worker
//! parked on the [`WorkerWaker`], the collector blocked on the output
//! channel, and no error surfaced anywhere. This was observed in the wild as
//! an intermittent whole-query hang (staged-scan q31 on the 100M-row
//! ClickBench box: an "Evicting" panic inside a build-time
//! `SlabAllocator::new` when the buffer pools were momentarily empty between
//! queries).
//!
//! The fix makes a failed build cancel the dataflow and queue the error, so
//! the query fails cleanly instead. This test builds a pipeline whose unary
//! factory panics on exactly one worker and asserts the query errors rather
//! than hanging.
//!
//! [`WorkerWaker`]: dispatch::worker::WorkerWaker

mod common;

use common::*;
use dispatch::worker::WORKER_IDX;
use dispatch::{Sender, Unary, UnaryFactory, UnaryResult, stealable, values_input};
use std::time::Duration;

/// Pass-through unary (the stage's behaviour is irrelevant — only its
/// factory's failure mode matters).
struct Identity;

impl Unary<i64, i64> for Identity {
    fn consume<S: Sender<i64>>(&mut self, v: i64, sender: &mut S) -> UnaryResult<()> {
        sender.send(v)?;
        Ok(())
    }
}

/// Builds fine everywhere except worker 1, where it panics — the build-time
/// failure shape of e.g. an allocator that cannot get pool memory.
struct PanicsOnWorkerOne;

impl UnaryFactory<i64, i64> for PanicsOnWorkerOne {
    type Unary = Identity;

    fn build_unary(self) -> Identity {
        if WORKER_IDX.get() == 1 {
            panic!("synthetic build failure");
        }
        Identity
    }
}

#[test]
fn failed_build_on_one_worker_errors_instead_of_hanging() {
    const WORKERS: usize = 4;
    let dispatch = dispatch(WORKERS);

    let spec = values_input(&dispatch, 0i64..10_000).chain(
        stealable::<i64>(WORKERS).into_iter().collect(),
        (0..WORKERS).map(|_| PanicsOnWorkerOne).collect(),
    );

    // Run the query on a watchdog: pre-fix it never completes (all workers
    // parked waiting for worker 1's sibling decrements), so a timeout — not a
    // wedged test harness — is the failure mode.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(spec.collect());
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(result) => {
            let err = result.expect_err("a failed build must fail the query");
            assert!(
                err.to_string().contains("synthetic build failure"),
                "error should carry the build panic, got: {err}"
            );
        }
        Err(_) => panic!(
            "query hung: a worker that fails to build a dataflow must cancel it, \
             not silently strand its siblings' counters"
        ),
    }
}
