//! A GROUP BY whose scatter rows do not fit the ring. In a pool with a spill
//! directory the workers have to spill buffers to disk as they consume, the
//! memory context has to free spilled buffers when it runs out of slots, and
//! the merge has to read the evicted rows back. Nothing here drives any of
//! that directly; the worker loop does, from the memory watermark down. In a
//! pool without one the workers spill nothing, and the query fails for want
//! of memory.

mod common;

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{DataFlowError, Dispatch, Distinct, IntKeyExtractor, SpillDirectory, values_input};

/// A 2MB ring slot per entry. The scatter rows of a worker sit in one slab
/// per partition, and a buffer can be spilled only once every slab in it has
/// filled, so with 4096 partitions a worker holds about 256MB of open slabs
/// before anything is spillable. The ring is sized just above that.
const RING_SLOTS: usize = 384;
/// Unique keys, each a 16-byte scatter row (hash and key): 384MB of rows, the
/// whole ring, of which the sealed buffers must go to disk.
const KEYS: i64 = 36_000_000;
const ROWS_PER_BATCH: i64 = 8_192;

fn int64_batch(from: i64, to: i64) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from_iter_values(from..to))],
    )
    .unwrap()
}

/// Group `KEYS` distinct ids by id on `dispatcher`'s workers.
fn group_distinct_ids(dispatcher: &DispatchGuard) -> Result<Vec<RecordBatch>, DataFlowError> {
    let batches: Vec<RecordBatch> = (0..KEYS)
        .step_by(ROWS_PER_BATCH as usize)
        .map(|from| int64_batch(from, (from + ROWS_PER_BATCH).min(KEYS)))
        .collect();
    values_input(dispatcher, batches)
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Distinct>(vec![0], Vec::new(), None, ())
        .collect()
}

#[test]
fn a_group_by_larger_than_the_ring_spills_and_reads_back() {
    let spill_root = tempfile::tempdir().unwrap();
    let spill_directory = SpillDirectory::open(spill_root.path()).unwrap();
    let d = DispatchGuard::new(Dispatch::spin_up_with_spill_directory(
        1,
        RING_SLOTS,
        None,
        Some(spill_directory),
    ));

    let results = group_distinct_ids(&d).unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort_unstable();
    assert_eq!(ids.len() as i64, KEYS);
    assert_eq!(ids.first(), Some(&0));
    assert_eq!(ids.last(), Some(&(KEYS - 1)));
}

#[test]
fn a_group_by_larger_than_the_ring_fails_without_a_spill_directory() {
    let d = DispatchGuard::new(Dispatch::spin_up(1, RING_SLOTS, None));

    let failure = group_distinct_ids(&d).unwrap_err().to_string();

    assert!(failure.contains("no spill directory set"), "{failure}");
}
