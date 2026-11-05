//! A GROUP BY whose scatter rows do not fit the ring: the workers have to
//! spill buffers to disk as they consume, the memory context has to free
//! spilled buffers when it runs out of slots, and the merge has to read the
//! evicted rows back. Nothing here drives any of that directly; the worker
//! loop does, from the memory watermark down.

mod common;

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{Dispatch, Distinct, IntKeyExtractor, values_input};

/// A 2MB ring slot per entry. A worker's scatter rows sit in one slab per
/// partition, growing to 64KB, and a buffer can be spilled only once every
/// slab in it has closed. Partitions filled at the same rate open their next
/// slab in the same short stretch, so with 4096 partitions a worker holds two
/// generations of slabs, about 512MB, before the older one is spillable, and
/// the spill check keeps 512MB more reclaimable ahead of that. The ring holds
/// both with a little to spare.
const RING_SLOTS: usize = 768;
/// Unique keys, each a 16-byte scatter row (hash and key), 1GB of rows in
/// slabs, of which most of the sealed buffers must go to disk.
const KEYS: i64 = 64_000_000;
const ROWS_PER_BATCH: i64 = 8_192;

fn int64_batch(from: i64, to: i64) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from_iter_values(from..to))],
    )
    .unwrap()
}

/// Spilling opens one file per partition, thousands of them, where a login
/// shell's default soft limit is 1024. The server raises its own limit at
/// startup; this process is the test binary, so it does the same.
fn raise_open_file_limit() {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: plain libc calls on a struct this function owns.
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit), 0);
        if limit.rlim_cur < limit.rlim_max {
            limit.rlim_cur = limit.rlim_max;
            assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &limit), 0);
        }
    }
}

#[test]
fn a_group_by_larger_than_the_ring_spills_and_reads_back() {
    raise_open_file_limit();
    let d = DispatchGuard::new(Dispatch::spin_up(1, RING_SLOTS, None));
    let batches: Vec<RecordBatch> = (0..KEYS)
        .step_by(ROWS_PER_BATCH as usize)
        .map(|from| int64_batch(from, (from + ROWS_PER_BATCH).min(KEYS)))
        .collect();

    let results = values_input(&d, batches)
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Distinct>(vec![0], Vec::new(), None, ())
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort_unstable();
    assert_eq!(ids.len() as i64, KEYS);
    assert_eq!(ids.first(), Some(&0));
    assert_eq!(ids.last(), Some(&(KEYS - 1)));
}
