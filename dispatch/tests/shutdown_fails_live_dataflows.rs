//! A pool that shuts down under a running dataflow fails it. Without that, the
//! dataflow's output channel closes like one that finished, and a collector
//! takes the rows it received so far for the whole result.

mod common;

use dispatch::{
    Consumer, Dispatch, Outputter, PipelineBreaker, Sender, UnaryFactory, UnaryResult, stealable,
    values_input,
};
use std::sync::mpsc;
use std::time::Duration;

const WORKERS: usize = 2;

/// Swallows its input and then reports "still working" forever.
#[derive(Default)]
struct EndlessBreaker;

impl Consumer<i64, i64> for EndlessBreaker {
    type Outputter = EndlessOutputter;

    fn consume(&mut self, _item: i64, _sender: &mut dyn Sender<i64>) -> UnaryResult<()> {
        Ok(())
    }

    fn into_outputter(self) -> UnaryResult<Option<EndlessOutputter>> {
        Ok(Some(EndlessOutputter))
    }
}

struct EndlessOutputter;

impl Outputter<i64> for EndlessOutputter {
    fn output(&mut self, _sender: &mut dyn Sender<i64>) -> UnaryResult<bool> {
        Ok(false)
    }
}

struct EndlessBreakerFactory;

impl UnaryFactory<i64, i64> for EndlessBreakerFactory {
    type Unary = PipelineBreaker<i64, i64, EndlessBreaker>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(EndlessBreaker)
    }
}

#[test]
fn shutting_down_fails_a_dataflow_still_running() {
    let dispatch = Dispatch::spin_up(WORKERS, 32, None);
    let dispatcher = dispatch.dispatcher().clone();
    let (result_tx, result_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let channels: Vec<_> = stealable::<i64>(dispatcher.topology())
            .into_iter()
            .collect();
        let factories: Vec<_> = (0..WORKERS).map(|_| EndlessBreakerFactory).collect();
        let result = values_input(&dispatcher, 0..WORKERS as i64)
            .chain(channels, factories)
            .collect();
        let _ = result_tx.send(result);
    });
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        result_rx.try_recv().is_err(),
        "the dataflow must still be running when the pool shuts down"
    );

    dispatch.exit();

    let result = result_rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the collector returns once the pool has shut down");
    let error = result.expect_err("a dataflow cut short by shutdown must not look finished");
    assert!(
        error.to_string().contains("shut down"),
        "unexpected error: {error}"
    );
}
