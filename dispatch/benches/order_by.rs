//! Benchmarks for the full ORDER BY operator (no LIMIT) over large inputs.
//!
//! The `order_by/*_limit` scenarios in `operators.rs` measure the bounded
//! top-N operator, whose working set is a few rows per worker. Here the whole
//! input is sorted and re-emitted, so run formation, the merge phases and the
//! per-column gather all run at input scale. Every scenario streams at least
//! `PIVOT_BENCH_SORT_MB` (default 1024, i.e. 1 GiB) of Arrow data through a
//! single `order_by` dataflow: large enough that the sort works far outside
//! cache and the merge output is written batch by batch across the ring.
//!
//! # Scenarios
//!
//! * `i64_random` - an `i64` key drawn from the whole 64-bit range plus one
//!   `i64` payload (16 B/row). Key comparisons are single integer compares,
//!   so run sorting and merge traversal dominate.
//! * `i64_presorted` - the same shape with globally ascending keys. Batches
//!   arrive already ordered and extend runs untouched; this is the ordered
//!   input fast path and the cost floor of re-emitting the input.
//! * `i64_wide_payload` - the `i64` key with four `i64` payload columns and a
//!   ~32 B string payload column. Compare cost matches `i64_random`, but each
//!   reordered row drags ~80 B through the gather, so take/scatter dominates.
//! * `string_random` - a ~32 B `Utf8View` key (dictionary-shared views whose
//!   bytes diverge early) plus one `i64` payload; byte-wise key comparisons
//!   dominate.
//!
//! Scenario sizes count logical row bytes (fixed widths; views plus payload
//! bytes for strings) independent of dictionary sharing, and generation stops
//! at the first batch to cross the target, so reported bytes/s throughput is
//! against the target size and accurate to within one batch.
//!
//! # Running
//!
//! ```sh
//! # All scenarios:
//! cargo bench --bench order_by
//!
//! # A/B across two builds of the operator (run, rebuild, run):
//! cargo bench --bench order_by -- --save-baseline before
//! cargo bench --bench order_by -- --baseline before
//! ```
//!
//! Tunables (env): `PIVOT_BENCH_SORT_MB` (MiB of data per scenario, default
//! 1024), `PIVOT_BENCH_WORKERS` (default: all cores), `PIVOT_BENCH_BUFFERS`
//! (ring slots of 2 MiB each, default 8192).
//!
//! Sizing the ring: the sorted output and the intermediate merge runs live on
//! ring buffers, so the ring must hold several multiples of the input size
//! (an 8 GiB ring evicts on the 1 GiB default; the default 8192 slots, a
//! 16 GiB ring, do not). Scale `PIVOT_BENCH_BUFFERS` with
//! `PIVOT_BENCH_SORT_MB`, and keep `buffers x 2 MiB` (the ring is one
//! pre-faulted mmap; the caches share its slots) plus the generated input
//! under box RAM.

use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, Criterion, SamplingMode, Throughput, black_box};

use dispatch::{DataFlowDispatcher, Dispatch, OrderBy, memory_ctx, values_input};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Rows per `RecordBatch`. Mirrors a scan morsel: many independent batches fan
/// out across workers through the work-stealing injector.
const BATCH_ROWS: usize = 64 * 1024;

fn read_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Bytes of Arrow data each scenario streams through the sort.
fn target_sort_bytes() -> usize {
    read_env_usize("PIVOT_BENCH_SORT_MB", 1024) * (1 << 20)
}

fn worker_count() -> usize {
    read_env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}

fn ring_buffers() -> usize {
    read_env_usize("PIVOT_BENCH_BUFFERS", 8192)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64), so both sides of an A/B sort identical data.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Column generators (one batch's worth at a time)
// ---------------------------------------------------------------------------

/// `i64` column of full-range random values (effectively all-distinct keys).
fn generate_i64_random(rng: &mut Rng, n: usize) -> ArrayRef {
    let v: Int64Array = (0..n).map(|_| rng.next_u64() as i64).collect();
    Arc::new(v)
}

/// `i64` column continuing a globally ascending sequence from `*next`.
fn generate_i64_ascending(next: &mut i64, n: usize) -> ArrayRef {
    let v: Int64Array = (0..n)
        .map(|_| {
            let k = *next;
            *next += 3;
            k
        })
        .collect();
    Arc::new(v)
}

/// Build `n_distinct` distinct strings averaging ~`avg_len` bytes whose bytes
/// diverge within the first few characters (a short shared lead, then a
/// per-entry token), so key comparisons short-circuit early like real ids and
/// URLs instead of scanning a long common prefix. All are >12 B so the views
/// are buffer-backed rather than inlined.
fn build_string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    let lead: String = prefix.bytes().take(7).map(|b| b as char).collect();
    (0..n_distinct)
        .map(|i| {
            let token = i.wrapping_mul(2_654_435_761) % 100_000_000;
            let mut s = format!("{lead}{token}");
            let target = (avg_len / 2).max(8) + (i % avg_len.max(1));
            let mut k = i as u64 + 1;
            while s.len() < target {
                k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                s.push('/');
                s.push_str(&(k % 100_000).to_string());
            }
            s.truncate(target.max(8));
            s
        })
        .collect()
}

/// A dictionary block: every distinct string stored once in a single backing
/// buffer, with the `(offset, len)` of each entry recorded. Columns then emit
/// dictionary-shared views into it, repeated rows pointing at the same bytes,
/// exactly as Parquet dictionary decoding produces.
struct DictBlock {
    buffer: arrow_buffer::Buffer,
    /// `(offset, len)` into `buffer` for each distinct dictionary entry.
    spans: Vec<(u32, u32)>,
}

fn build_dict_block(dict: &[String]) -> DictBlock {
    let mut data: Vec<u8> = Vec::new();
    let mut spans = Vec::with_capacity(dict.len());
    for s in dict {
        spans.push((data.len() as u32, s.len() as u32));
        data.extend_from_slice(s.as_bytes());
    }
    DictBlock {
        buffer: arrow_buffer::Buffer::from_vec(data),
        spans,
    }
}

/// An `n`-row `StringViewArray` of uniformly drawn dictionary-shared views.
fn generate_string_col(rng: &mut Rng, n: usize, block: &DictBlock) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    let blk = b.append_block(block.buffer.clone());
    for _ in 0..n {
        let (off, len) = block.spans[rng.below(block.spans.len())];
        b.try_append_view(blk, off, len).unwrap();
    }
    Arc::new(b.finish())
}

fn make_schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn make_batch(schema: &Arc<Schema>, cols: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), cols).unwrap()
}

/// Logical bytes of one batch: fixed widths for primitives, view plus payload
/// bytes for strings. Independent of buffer sharing, so it is a stable
/// denominator for bytes/s throughput.
fn count_logical_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            DataType::Int64 => 8 * c.len(),
            DataType::Utf8View => {
                let a = c.as_any().downcast_ref::<StringViewArray>().unwrap();
                16 * a.len()
                    + a.views()
                        .iter()
                        .map(|&v| (v as u32) as usize)
                        .sum::<usize>()
            }
            other => panic!("no logical size rule for bench column type {other}"),
        })
        .sum()
}

/// Build `BATCH_ROWS`-sized batches until their logical bytes reach `target`.
fn generate_until(
    target: usize,
    schema: &Arc<Schema>,
    mut make_cols: impl FnMut(usize) -> Vec<ArrayRef>,
) -> Vec<RecordBatch> {
    let mut total = 0;
    let mut batches = Vec::new();
    while total < target {
        let b = make_batch(schema, make_cols(BATCH_ROWS));
        total += count_logical_bytes(&b);
        batches.push(b);
    }
    batches
}

// ---------------------------------------------------------------------------
// Run helper: stream the prebuilt batches through the pipeline and drain.
// `batches` is cloned per iteration (cheap Arc bumps); the engine is reused.
// ---------------------------------------------------------------------------

fn run_sort(d: &DataFlowDispatcher, batches: &[RecordBatch], order_by: &[OrderBy]) {
    let spec = values_input(d, batches.to_vec()).record_batches();
    let out = spec.order_by(order_by.to_vec()).collect().unwrap();
    black_box(out);
}

/// Register one scenario. `make_data` runs only when the benchmark is selected
/// and its result is cached across iterations, so a filtered run builds just
/// the one scenario's dataset.
fn register<D>(
    c: &mut Criterion,
    d: &DataFlowDispatcher,
    name: &str,
    make_data: D,
    keys: Vec<OrderBy>,
) where
    D: FnOnce() -> Vec<RecordBatch>,
{
    let make_data = RefCell::new(Some(make_data));
    let cache: RefCell<Option<Vec<RecordBatch>>> = RefCell::new(None);
    let mut g = c.benchmark_group("order_by");
    g.throughput(Throughput::Bytes(target_sort_bytes() as u64));
    // Whole-input sorts run seconds per iteration: a handful of flat samples
    // beats criterion's default hundred.
    g.sample_size(10);
    g.sampling_mode(SamplingMode::Flat);
    g.bench_function(name, |b| {
        if cache.borrow().is_none() {
            let data = make_data.borrow_mut().take().expect("make_data missing")();
            *cache.borrow_mut() = Some(data);
        }
        let cached = cache.borrow();
        let batches = cached.as_ref().unwrap();
        // Per-iteration setup (UNTIMED): re-zero the buffers the previous
        // query dirtied, so allocation cost stays out of the measured window.
        b.iter_batched(
            || {
                d.run_on_workers(|| {
                    memory_ctx().zero_dirty_buffers();
                })
            },
            |_| run_sort(d, batches, &keys),
            BatchSize::PerIteration,
        );
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

fn bench_order_by_large(c: &mut Criterion, d: &DataFlowDispatcher) {
    let target = target_sort_bytes();

    register(
        c,
        d,
        "i64_random",
        || {
            let sch = make_schema(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]);
            let mut rng = Rng::new(1);
            generate_until(target, &sch, |n| {
                vec![
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        c,
        d,
        "i64_presorted",
        || {
            let sch = make_schema(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]);
            let mut rng = Rng::new(2);
            let mut next_key = 0;
            generate_until(target, &sch, |n| {
                vec![
                    generate_i64_ascending(&mut next_key, n),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        c,
        d,
        "i64_wide_payload",
        || {
            let sch = make_schema(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v0", DataType::Int64, false),
                Field::new("v1", DataType::Int64, false),
                Field::new("v2", DataType::Int64, false),
                Field::new("v3", DataType::Int64, false),
                Field::new("s", DataType::Utf8View, false),
            ]);
            let dict = build_string_dict(1_000_000, "payload ", 32);
            let block = build_dict_block(&dict);
            let mut rng = Rng::new(3);
            generate_until(target, &sch, |n| {
                vec![
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_string_col(&mut rng, n, &block),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        c,
        d,
        "string_random",
        || {
            let sch = make_schema(vec![
                Field::new("k", DataType::Utf8View, false),
                Field::new("v", DataType::Int64, false),
            ]);
            let dict = build_string_dict(1_000_000, "series/", 32);
            let block = build_dict_block(&dict);
            let mut rng = Rng::new(4);
            generate_until(target, &sch, |n| {
                vec![
                    generate_string_col(&mut rng, n, &block),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );
}

// ---------------------------------------------------------------------------
// Harness: one shared Dispatch (worker pool) for every scenario.
// ---------------------------------------------------------------------------

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    eprintln!(
        "order_by benches: {workers} workers, {buffers} ring buffers, {} MiB per scenario",
        target_sort_bytes() >> 20
    );

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let dispatcher = dispatch.dispatcher().clone();

    let mut c = Criterion::default().configure_from_args();
    bench_order_by_large(&mut c, &dispatcher);
    c.final_summary();

    dispatch.exit();
}
