//! Tests for a query's live stats: the messages a [`LiveStatsSubscription`]
//! receives about a dataflow, first its operators and then every worker's
//! stats while it runs and when it ends.

mod common;

use common::*;
use dispatch::{
    BuildContext, Built, BuiltUnary, CustomStats, DataFlowDispatcher, LiveStatsMessage,
    LiveStatsSubscription, OperatorDescription, OperatorIO, Sender, Unary, UnaryFactory,
    UnaryResult, WorkerLiveStats, channel_input, stealable, values_input,
};
use std::sync::mpsc;
use std::time::Duration;

const WORKER_COUNT: usize = 2;

/// How often a worker may send its stats in these tests.
const INTERVAL: Duration = Duration::from_millis(1);

/// A dispatcher whose dataflows send their live stats to the returned
/// receiver, each worker at most once per [`INTERVAL`].
fn subscribe_to_live_stats(
    dispatcher: &DataFlowDispatcher,
) -> (DataFlowDispatcher, mpsc::Receiver<LiveStatsMessage>) {
    let (sender, receiver) = mpsc::channel();
    let subscription = LiveStatsSubscription {
        interval: INTERVAL,
        sender,
    };
    (dispatcher.clone().with_live_stats(subscription), receiver)
}

/// The workers' stats among the messages received so far.
fn take_worker_stats(live_stats: &mpsc::Receiver<LiveStatsMessage>) -> Vec<WorkerLiveStats> {
    live_stats
        .try_iter()
        .filter_map(|message| match message {
            LiveStatsMessage::Worker(worker) => Some(worker),
            LiveStatsMessage::Operators(_) => None,
        })
        .collect()
}

/// The name of every custom stat the workers sent so far, with the index of
/// the operator it was sent under.
fn take_custom_stat_names_by_operator(
    live_stats: &mpsc::Receiver<LiveStatsMessage>,
) -> Vec<(usize, &'static str)> {
    take_worker_stats(live_stats)
        .into_iter()
        .flat_map(|worker| worker.operators.into_iter().enumerate())
        .flat_map(|(operator_idx, operator)| {
            operator
                .custom_stats
                .into_iter()
                .map(move |stat| (operator_idx, stat.name))
        })
        .collect()
}

/// Passes every item through, reporting each one as a custom stat.
struct ItemReporter {
    custom_stats: CustomStats,
}

impl Unary<i64, i64> for ItemReporter {
    fn consume(
        &mut self,
        item: i64,
        sender: &mut dyn Sender<i64>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        self.custom_stats
            .report("item_seen", || serde_json::json!({ "item": item }));
        sender.send(item)?;
        Ok(())
    }

    fn custom_stats(&self) -> Option<CustomStats> {
        Some(self.custom_stats.clone())
    }
}

struct ItemReporterFactory {
    custom_stats: CustomStats,
}

impl UnaryFactory<i64, i64> for ItemReporterFactory {
    type Unary = ItemReporter;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(ItemReporter {
            custom_stats: self.custom_stats,
        })
    }
}

#[test]
fn the_first_message_describes_the_operators_and_their_inputs() {
    let dispatch = dispatch(WORKER_COUNT);
    let (subscribed, live_stats) = subscribe_to_live_stats(&dispatch);

    values_input(&subscribed, 0..10i64)
        .map_each(|x| Ok(x * 2))
        .execute()
        .collect_with_stats()
        .unwrap();

    let first_message = live_stats.try_recv().unwrap();
    assert_eq!(
        first_message,
        LiveStatsMessage::Operators(vec![
            OperatorDescription {
                name: "Forward",
                inputs: vec![],
            },
            OperatorDescription {
                name: "Map",
                inputs: vec![0],
            },
        ])
    );
}

#[test]
fn every_worker_ends_with_a_message_holding_each_operators_counters() {
    let dispatch = dispatch(WORKER_COUNT);
    let (subscribed, live_stats) = subscribe_to_live_stats(&dispatch);

    values_input(&subscribed, 0..1000i64)
        .map_each(|x| Ok(x * 2))
        .execute()
        .collect_with_stats()
        .unwrap();

    let last_messages: Vec<WorkerLiveStats> = take_worker_stats(&live_stats)
        .into_iter()
        .filter(|worker| worker.finished)
        .collect();
    let mut workers: Vec<usize> = last_messages.iter().map(|m| m.worker_idx).collect();
    workers.sort_unstable();
    assert_eq!(workers, vec![0, 1]);
    let map_cpu: Duration = last_messages
        .iter()
        .map(|worker| worker.operators[1].counters.cpu)
        .sum();
    assert!(map_cpu > Duration::ZERO);
}

#[test]
fn a_custom_stat_an_operator_reports_is_sent_once_under_that_operator() {
    let dispatch = dispatch(WORKER_COUNT);
    let (subscribed, live_stats) = subscribe_to_live_stats(&dispatch);

    values_input(&subscribed, 0..10i64)
        .chain(
            stealable::<i64>(subscribed.topology())
                .into_iter()
                .collect(),
            (0..WORKER_COUNT)
                .map(|_| ItemReporterFactory {
                    custom_stats: subscribed.create_custom_stats(),
                })
                .collect(),
        )
        .execute()
        .collect_with_stats()
        .unwrap();

    let reported = take_custom_stat_names_by_operator(&live_stats);
    assert_eq!(reported, vec![(1, "item_seen"); 10]);
}

#[test]
fn a_custom_stat_reported_before_the_dataflow_runs_is_sent() {
    let dispatch = dispatch(WORKER_COUNT);
    let (subscribed, live_stats) = subscribe_to_live_stats(&dispatch);
    let custom_stats_per_worker: Vec<CustomStats> = (0..WORKER_COUNT)
        .map(|_| subscribed.create_custom_stats())
        .collect();
    custom_stats_per_worker[0].report("input_skipped", || serde_json::json!({ "input": 3 }));

    values_input(&subscribed, Vec::<i64>::new())
        .chain(
            stealable::<i64>(subscribed.topology())
                .into_iter()
                .collect(),
            custom_stats_per_worker
                .into_iter()
                .map(|custom_stats| ItemReporterFactory { custom_stats })
                .collect(),
        )
        .execute()
        .collect_with_stats()
        .unwrap();

    let reported = take_custom_stat_names_by_operator(&live_stats);
    assert_eq!(reported, vec![(1, "input_skipped")]);
}

#[test]
fn a_dataflow_without_live_stats_keeps_no_custom_stats() {
    let dispatch = dispatch(WORKER_COUNT);
    let custom_stats = dispatch.create_custom_stats();
    let mut built_values = 0;

    custom_stats.report("item_seen", || {
        built_values += 1;
        serde_json::json!({})
    });

    assert_eq!(built_values, 0);
}

#[test]
fn a_worker_sends_its_stats_while_the_dataflow_is_still_running() {
    let dispatch = dispatch(WORKER_COUNT);
    let (subscribed, live_stats) = subscribe_to_live_stats(&dispatch);
    let (input, source) = channel_input::<i64>(&subscribed, 16, Box::new(|| {}));
    let handle = source.map_each(Ok).execute();

    // Two items an interval apart: whichever worker takes the second one has
    // stats that are due by then.
    assert!(input.try_send(7).is_ok());
    std::thread::sleep(INTERVAL * 20);
    assert!(input.try_send(8).is_ok());
    let worker_stats = std::iter::repeat_with(|| {
        live_stats
            .recv_timeout(Duration::from_secs(30))
            .expect("a worker sends its stats before the dataflow ends")
    })
    .find_map(|message| match message {
        LiveStatsMessage::Worker(worker) => Some(worker),
        LiveStatsMessage::Operators(_) => None,
    })
    .unwrap();

    assert!(!worker_stats.finished);
    input.close();
    let mut items = handle.collect().unwrap();
    items.sort_unstable();
    assert_eq!(items, vec![7, 8]);
}
