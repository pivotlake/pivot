//! Ring initialization modes. An [`RingInit::OnDemand`] ring hands its slots to
//! the free pools untouched, so the process only becomes as resident as the work
//! it actually runs.

mod common;

use arrow_schema::DataType;
use common::*;
use dispatch::{AggregationKind, AggregationSlot, BUFFER_SIZE, Dispatch, RingInit, values_input};

/// A ring far larger than the query below needs, so faulting it in up front
/// would be unmistakable in the process's resident size.
const RING_BUFFERS: usize = 2048;

/// This process's resident set size, in bytes.
fn resident_bytes() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let rss = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .expect("VmRSS in /proc/self/status");
    let kilobytes: usize = rss.split_whitespace().nth(1).unwrap().parse().unwrap();
    kilobytes * 1024
}

#[test]
fn an_on_demand_ring_serves_queries_without_becoming_resident() {
    let batch = strings_and_ints(&["a", "b", "c", "d", "e"], &[1, 2, 3, 4, 5]);
    let baseline = resident_bytes();

    let dispatch = Dispatch::spin_up_with_ring_init(1, RING_BUFFERS, None, RingInit::OnDemand);
    let results = values_input(dispatch.dispatcher(), vec![batch])
        .record_batches()
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();
    let growth = resident_bytes().saturating_sub(baseline);
    dispatch.exit();

    assert_eq!(extract_count(&results), 5);
    let ring_bytes = RING_BUFFERS * BUFFER_SIZE;
    assert!(
        growth < ring_bytes / 10,
        "resident size grew by {growth} bytes, most of a {ring_bytes}-byte ring"
    );
}
