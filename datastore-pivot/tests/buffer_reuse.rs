mod common;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{AggregationKind, AggregationSlot, Projection, RECORD_BATCH_SIZE};
use parquet_engine::table_input;

#[test]
fn subsequent_batches_reuse_write_buffer() {
    let dispatcher = dispatch(1);
    let n = 10_000; // > RECORD_BATCH_SIZE (8192) to produce multiple batches from one page
    let names: Vec<&str> = (0..n).map(|_| "x").collect();
    let values: Vec<i64> = (0..n as i64).collect();
    let (_dir, table) = parquet_table(&dispatcher, &[strings_and_ints(&names, &values)], true);

    let results = table_input(&dispatcher, &table, Projection::columns([1]), false)
        .map(|| |d| d.column(0).to_data().buffers()[0].as_ptr() as usize)
        .collect()
        .unwrap();

    assert!(
        results.len() >= 2,
        "expected at least 2 batches, got {}",
        results.len()
    );
    let ptr0 = results[0];
    let ptr1 = results[1];
    assert_eq!(
        ptr1 - ptr0,
        RECORD_BATCH_SIZE * 8,
        "subsequent batches should be contiguous in the same WriteBuffer"
    );
}

/// Not yet implemented: SlabAllocator needs to reclaim space when the most
/// recent allocation is dropped (bump pointer rollback), so that the next
/// batch can reuse the same region.
#[test]
#[ignore]
fn dropped_batch_memory_is_reused() {
    let dispatcher = dispatch(1);
    let n = 10_000;
    let names: Vec<&str> = (0..n).map(|_| "x").collect();
    let values: Vec<i64> = (0..n as i64).collect();
    let (_dir, table) = parquet_table(&dispatcher, &[strings_and_ints(&names, &values)], true);

    let ptrs: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(vec![]));
    let ptrs_clone = ptrs.clone();

    table_input(&dispatcher, &table, Projection::columns([1]), false)
        .project(|| {
            let ptrs = ptrs_clone.clone();
            move |batch: RecordBatch| {
                ptrs.lock()
                    .unwrap()
                    .push(batch.column(0).to_data().buffers()[0].as_ptr() as usize);
                let len = batch.num_rows();
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("d", DataType::Int64, false)])),
                    vec![Arc::new(Int64Array::from(vec![0i64; len]))],
                )
                .unwrap()
            }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    let ptrs = ptrs.lock().unwrap();
    assert!(
        ptrs.len() >= 2,
        "expected at least 2 batches, got {}",
        ptrs.len()
    );
    assert_eq!(
        ptrs[0], ptrs[1],
        "second batch should reuse memory from dropped first batch"
    );
}

#[test]
fn decompressed_string_buffer_is_reused() {
    // This pins the underlying cross-page decompression-buffer recycling. The
    // decompressed-page cache retains each page's buffer, which would mask that
    // reuse, so disable it here (its own behaviour is covered by the decompressor
    // unit tests). Set before the dispatcher is built; the cache reads this at
    // construction.
    unsafe { std::env::set_var("PIVOT_DECOMPRESSED_CACHE", "false") };
    let dispatcher = dispatch(1);
    // 100K unique strings over 12 bytes each are five Parquet pages. Each page
    // decompresses into a WriteBuffer that goes back to the pool once the
    // batches over it are dropped, so the scan cycles through a bounded set of
    // buffers rather than taking a fresh one per page. A decode range that
    // straddles a page boundary keeps the earlier page alive while the next
    // one is decompressed, so the bound is two.
    let n = 100_000;
    let owned: Vec<String> = (0..n)
        .map(|i| format!("long-string-value-{i:06}"))
        .collect();
    let names: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "name",
            DataType::Utf8View,
            false,
        )])),
        vec![Arc::new(StringViewArray::from(names))],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatcher, &[batch], false);

    let ptrs: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(vec![]));
    let ptrs_clone = ptrs.clone();

    let res = table_input(&dispatcher, &table, Projection::columns([0]), false)
        .project(|| {
            let ptrs = ptrs_clone.clone();
            move |batch: RecordBatch| {
                let sv = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                ptrs.lock()
                    .unwrap()
                    .push(sv.data_buffers()[0].as_ptr() as usize);
                let len = batch.num_rows();
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("d", DataType::Int64, false)])),
                    vec![Arc::new(Int64Array::from(vec![0i64; len]))],
                )
                .unwrap()
            }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&res), n as i64);

    let ptrs = ptrs.lock().unwrap();
    let distinct: HashSet<usize> = ptrs.iter().copied().collect();
    assert!(distinct.len() <= 2, "{} distinct buffers", distinct.len());
}
