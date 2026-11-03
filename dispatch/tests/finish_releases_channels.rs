//! A stage whose workers report to one collecting worker over a channel has
//! to let that channel close once every worker has sent: the collecting
//! worker finishes only when it sees the end of its input, so a sending end
//! kept anywhere else holds the whole dataflow open.

mod common;

use arrow_schema::DataType;
use common::{dispatch, extract_count, strings_and_ints};
use dispatch::{AggregationKind, AggregationSlot, values_input};
use std::sync::mpsc;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(20);

#[test]
fn a_global_count_over_two_workers_finishes() {
    let dispatch = dispatch(2);
    let dispatcher = (*dispatch).clone();
    let batch = strings_and_ints(&["a", "b", "c"], &[1, 2, 3]);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let results = values_input(&dispatcher, vec![batch])
            .record_batches()
            .aggregate::<i64>(vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )])
            .collect();
        let _ = tx.send(results);
    });

    let results = rx
        .recv_timeout(TIMEOUT)
        .expect("the count never finished")
        .unwrap();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn a_limit_over_two_workers_finishes() {
    let dispatch = dispatch(2);
    let dispatcher = (*dispatch).clone();
    let batch = strings_and_ints(&["a", "b", "c"], &[1, 2, 3]);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let results = values_input(&dispatcher, vec![batch])
            .record_batches()
            .limit(2, 0)
            .collect();
        let _ = tx.send(results);
    });

    let results = rx
        .recv_timeout(TIMEOUT)
        .expect("the limit never finished")
        .unwrap();

    let rows: usize = results.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(rows, 2);
}
