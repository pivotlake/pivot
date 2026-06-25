//! Regression test for the pipeline-breaker finish/park deadlock.
//!
//! A pipeline breaker emits its results over several `finish()`/`run()` calls,
//! returning `false` until it has drained. The bug: `UnaryOperator::try_finish`
//! reported that still-flushing `false` the same as "not ready", and the
//! worker's finish pass does not set `did_work` — so a worker that called
//! `finish()`, got `false`, and had nothing else advance the wake count could
//! park one stage short of finishing and never wake, hanging the whole dataflow.
//! The fix makes `try_finish` return `FinishStatus::Working`, which the worker
//! counts as work and re-drives via `run` instead of parking.
//!
//! Triggering it needs the worker to reach a send-less `false` step while
//! "caught up" (its `last_seen` wake count equals the current one) and with no
//! sibling about to notify — otherwise the spin-before-park, a stale `last_seen`,
//! or a send/finish notify rescues it. So this uses a breaker that emits NOTHING
//! (no send-notifies during the flush) over a tiny source (minimal consume-phase
//! notifies, so workers stay caught up). With the bug, the workers park on the
//! breaker's send-less flush step and the dataflow never completes; we detect
//! that as a timeout. With the fix, every run finishes immediately.

mod common;

use common::dispatch;
use dispatch::{
    Consumer, Outputter, PipelineBreaker, Sender, UnaryFactory, UnaryResult, stealable,
    values_input,
};
use std::sync::mpsc;
use std::time::Duration;

const WORKERS: usize = 2;

/// A pipeline breaker that swallows its input and emits nothing — it just takes
/// a few send-less `output()` steps before reporting done.
#[derive(Default)]
struct SilentBreaker;

impl Consumer<i64, i64> for SilentBreaker {
    type Outputter = SilentOutputter;

    fn consume<S: Sender<i64>>(&mut self, _item: i64, _sender: &mut S) -> UnaryResult<()> {
        Ok(())
    }

    fn into_outputter(self) -> UnaryResult<Option<SilentOutputter>> {
        // Always produce an outputter (even for an empty partition) so every
        // worker drives the send-less flush — that's the step that wedged.
        Ok(Some(SilentOutputter { steps_left: 3 }))
    }
}

/// Emits nothing; returns `false` (still working) a few times, then `true`.
/// No sends means no send-notifies during the flush, so `wake_count` doesn't
/// advance and a worker that parks on one of these `false` steps has
/// `last_seen == wake_count` — and would otherwise sleep forever.
struct SilentOutputter {
    steps_left: u32,
}

impl Outputter<i64> for SilentOutputter {
    fn output<S: Sender<i64>>(&mut self, _sender: &mut S) -> UnaryResult<bool> {
        if self.steps_left > 0 {
            self.steps_left -= 1;
            return Ok(false);
        }
        Ok(true)
    }
}

struct SilentBreakerFactory;

impl UnaryFactory<i64, i64> for SilentBreakerFactory {
    type Unary = PipelineBreaker<i64, i64, SilentBreaker>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(SilentBreaker)
    }
}

#[test]
fn pipeline_breaker_flush_does_not_hang_the_pool() {
    const RUNS: usize = 50;
    const TIMEOUT: Duration = Duration::from_secs(20);

    let dispatch = dispatch(WORKERS);

    for run in 0..RUNS {
        // Run the dataflow on a helper thread so the main thread can detect a
        // hang via a timeout instead of blocking forever with it.
        let d = (*dispatch).clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let channels: Vec<_> = stealable::<i64>(WORKERS).into_iter().collect();
            let factories: Vec<_> = (0..WORKERS).map(|_| SilentBreakerFactory).collect();
            // Tiny source: just enough to drive the breaker, with minimal
            // consume-phase notify traffic so the workers stay caught up.
            let result = values_input(&d, 0..WORKERS as i64)
                .chain(channels, factories)
                .collect();
            let _ = tx.send(result);
        });

        let out = rx
            .recv_timeout(TIMEOUT)
            .unwrap_or_else(|_| {
                panic!("run {run}: dataflow hung — pipeline-breaker finish/park deadlock")
            })
            .expect("dataflow returned an error");

        assert!(
            out.is_empty(),
            "run {run}: silent breaker should emit nothing"
        );
    }
}
