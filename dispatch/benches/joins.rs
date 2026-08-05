//! Microbenchmarks for the hash join, driven in **complete isolation** — no
//! Parquet, no IO, no catalog. Pre-built Arrow `RecordBatch`es stream into the
//! join's two inputs through `values_input`, exactly as scans would fan
//! morsels across the worker pool, and the output drains through a global
//! COUNT so every listed output column is materialized (the gather runs) but
//! nothing downstream adds weight to the profile.
//!
//! # Why these scenarios
//!
//! Each scenario reproduces the *statistical shape* of a join regime real
//! analytical queries put the executor in. For a hash join the properties
//! that decide the perf sample distribution are: the **key shape** (which
//! monomorphized [`JoinKey`] path runs), the **build row count** relative to
//! the last-level cache (whether a probe's directory and arena reads are
//! cache hits or DRAM misses), the **match rate and multiplicity** (how much
//! of probing is bloom rejection versus arena walking versus output
//! gathering), the **gathered column count and types**, and the **join
//! kind** (an outer join adds the matched-flag writes and the unmatched
//! scan, a semi join gathers nothing and stops at the first match). Each
//! scenario pins one combination, so a few seconds of it is a fast, faithful
//! replicator of the matching real join's profile — iterate on the join here
//! instead of on a 40 GB dataset.
//!
//! The data shapes (not any query) are described next to each scenario. We
//! deliberately avoid coupling the bench to any external suite.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench joins
//!
//! # One scenario, looped with no analysis — ideal to attach perf to:
//! cargo bench --bench joins -- "join/packed_pair_big_build" --profile-time 20
//! ```
//!
//! Tunables (env): `PIVOT_BENCH_JOIN_SCALE` (multiplies every scenario's row
//! counts, default 1.0), `PIVOT_BENCH_WORKERS` (default: all cores),
//! `PIVOT_BENCH_BUFFERS` (ring slots of 2 MiB each, default 4096). The sizing
//! notes in `operators.rs` apply here too; under `perf`, raise the
//! locked-memory limit (`ulimit -l unlimited`).
//!
//! VALIDATION — each scenario was perf-profiled side by side with the real
//! query whose join it mirrors (per-query worker-scoped recordings on a
//! 16-core box, comparing the shares of the join-namespace symbols: the
//! probe pipeline, the output gather appends, and the build scatter). The
//! sparse, mid-build, packed-pair, outer, and semi scenarios land within a
//! few points of their real counterparts' splits; the big-build and
//! build-heavy scenarios isolate the scatter-dominated component that real
//! plans mix with other probes. Two notes from that comparison: real plans
//! precompute string predicates on the build side, so string columns reach
//! the gather less often than query text suggests (the string scenario
//! exercises the view-gather path that shows up at a few percent inside
//! several real joins), and per-query profiles merge every same-shaped join
//! instantiation into one symbol, so a query with several joins compares
//! against a blend of scenarios rather than one.

use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{BatchSize, Criterion, Throughput, black_box};

use dispatch::{
    AggregationKind, AggregationSlot, DataFlowDispatcher, Dispatch, JoinKind, JoinSpec, memory_ctx,
    values_input,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Rows per `RecordBatch`, mirroring a scan morsel.
const BATCH_ROWS: usize = 64 * 1024;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Every scenario's canonical row counts multiply by this.
fn scale() -> f64 {
    std::env::var("PIVOT_BENCH_JOIN_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0)
}

fn scaled(rows: usize) -> usize {
    ((rows as f64 * scale()) as usize).max(BATCH_ROWS)
}

fn worker_count() -> usize {
    env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}

fn ring_buffers() -> usize {
    env_usize("PIVOT_BENCH_BUFFERS", 4096)
}

fn batch_sizes(total: usize) -> impl Iterator<Item = usize> {
    (0..total)
        .step_by(BATCH_ROWS)
        .map(move |start| BATCH_ROWS.min(total - start))
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64), as in operators.rs.
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

    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    fn chance(&mut self, p: f64) -> bool {
        self.below(1_000_000) < (p * 1_000_000.0) as usize
    }
}

// ---------------------------------------------------------------------------
// Column generators
// ---------------------------------------------------------------------------

/// Spread an id across the 64-bit space so values look like real keys. Both
/// sides apply the same transform, so equality is preserved.
#[inline]
fn spread(id: usize) -> i64 {
    (id as i64).wrapping_mul(0x100_0001)
}

/// The second lane of a two-column key, derived from the first so a probe row
/// drawing `id` reproduces exactly one stored pair.
#[inline]
fn pair_lane(id: usize, domain: usize) -> i64 {
    (id.wrapping_mul(2_654_435_761) % domain) as i64
}

/// Sequential unique keys `start..start + n`, spread.
fn unique_keys(start: usize, n: usize) -> ArrayRef {
    let v: Int64Array = (start..start + n).map(spread).collect();
    Arc::new(v)
}

/// Keys drawn uniformly from `[0, domain)`, spread: every row matches a build
/// side holding `unique_keys(0, domain)`, with multiplicity `rows / domain`.
fn uniform_keys(rng: &mut Rng, n: usize, domain: usize) -> ArrayRef {
    let v: Int64Array = (0..n).map(|_| spread(rng.below(domain))).collect();
    Arc::new(v)
}

/// Keys that hit `[0, domain)` with probability `hit_rate` and a disjoint
/// range otherwise, spread. The build side never holds the disjoint range.
fn sparse_keys(rng: &mut Rng, n: usize, domain: usize, hit_rate: f64) -> ArrayRef {
    let v: Int64Array = (0..n)
        .map(|_| {
            if rng.chance(hit_rate) {
                spread(rng.below(domain))
            } else {
                spread(domain + rng.below(domain * 64 + 1))
            }
        })
        .collect();
    Arc::new(v)
}

/// An i64 value column (a price, a date, a count).
fn i64_values(rng: &mut Rng, n: usize) -> ArrayRef {
    let v: Int64Array = (0..n).map(|_| rng.below(1 << 40) as i64).collect();
    Arc::new(v)
}

/// A ~25-byte string per row, unique-ish: long enough that most values live
/// outside a view's inline bytes, the shape of a name or type column.
fn string_values(start: usize, n: usize) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    for i in start..start + n {
        b.append_value(format!("value-{:012}-tail{:06}", i, i % 971));
    }
    Arc::new(b.finish())
}

fn schema(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

fn batch(schema: &SchemaRef, cols: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), cols).unwrap()
}

fn i64_field(name: &str) -> Field {
    Field::new(name, DataType::Int64, false)
}

// ---------------------------------------------------------------------------
// Run helper: two prebuilt inputs through one join, drained by a global COUNT.
// ---------------------------------------------------------------------------

struct JoinData {
    probe: Vec<RecordBatch>,
    build: Vec<RecordBatch>,
}

/// Execute `probe JOIN build` and drain through COUNT(*): the join
/// materializes every column its spec lists (the gather runs in full), and
/// the aggregate consumes the output without adding meaningful weight, so
/// the profile stays the join's.
fn run_join(d: &DataFlowDispatcher, data: &JoinData, key_types: &[DataType], spec: &JoinSpec) {
    let probe = values_input(d, data.probe.clone()).record_batches();
    let build = values_input(d, data.build.clone()).record_batches();
    let out = probe
        .join(build, key_types, spec.clone())
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();
    black_box(out);
}

/// Register one scenario; data is built lazily on first use, as in
/// operators.rs, so a filtered run only pays for what it measures.
#[allow(clippy::too_many_arguments)]
fn bench<D>(
    c: &mut Criterion,
    d: &DataFlowDispatcher,
    name: &str,
    probe_rows: usize,
    make_data: D,
    key_types: Vec<DataType>,
    spec: JoinSpec,
) where
    D: FnOnce() -> JoinData,
{
    let make_data = std::cell::RefCell::new(Some(make_data));
    let cache: std::cell::RefCell<Option<JoinData>> = std::cell::RefCell::new(None);
    let mut g = c.benchmark_group("dispatch");
    g.throughput(Throughput::Elements(probe_rows as u64));
    g.sample_size(10);
    g.bench_function(name, |b| {
        if cache.borrow().is_none() {
            let data = make_data.borrow_mut().take().expect("make_data missing")();
            *cache.borrow_mut() = Some(data);
        }
        let cached = cache.borrow();
        let data = cached.as_ref().unwrap();
        b.iter_batched(
            || {
                d.run_on_workers(|| {
                    memory_ctx().zero_dirty_buffers();
                })
            },
            |_| run_join(d, data, &key_types, &spec),
            BatchSize::PerIteration,
        );
    });
    g.finish();
}

/// A single-key spec over column 0 of both sides, emitting the listed
/// output columns as plain i64 fields.
fn single_key_spec(probe_out: Vec<usize>, build_out: Vec<usize>, kind: JoinKind) -> JoinSpec {
    let outer = matches!(kind, JoinKind::BuildOuter);
    JoinSpec {
        build_key_indices: vec![0],
        probe_key_indices: vec![0],
        probe_fields: probe_out
            .iter()
            .map(|i| Field::new(format!("p{i}"), DataType::Int64, outer))
            .collect(),
        build_fields: build_out
            .iter()
            .map(|i| Field::new(format!("b{i}"), DataType::Int64, false))
            .collect(),
        probe_output_indices: probe_out,
        build_output_indices: build_out,
        kind,
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

fn bench_joins(c: &mut Criterion, d: &DataFlowDispatcher) {
    // (a) A cache-resident dimension build probed by a huge fact stream, few
    //     hits: the shape of "fact JOIN heavily-filtered dimension". Almost
    //     all probe time is hashing plus bloom rejection on the directory
    //     word; the arena and gather barely run. Build: 128k unique keys, no
    //     output columns. Probe: 32M rows, 5% hit rate, one value column
    //     carried through.
    {
        let probe_rows = scaled(32_000_000);
        let build_rows = scaled(131_072);
        bench(
            c,
            d,
            "join/tiny_build_sparse_probe",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k")]);
                let mut rng = Rng::new(11);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    sparse_keys(&mut rng, n, build_rows, 0.05),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(&bsch, vec![unique_keys(start, n)]);
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![], JoinKind::Inner),
        );
    }

    // (b) A DRAM-sized build of a few million rows probed by a filtered fact
    //     stream with a moderate hit rate, two build columns gathered: the
    //     shape of "big fact JOIN (already-joined) mid-size relation". The
    //     directory and arena reads miss cache, and the gather is a real
    //     part of the profile.
    {
        let probe_rows = scaled(32_000_000);
        let build_rows = scaled(3_000_000);
        bench(
            c,
            d,
            "join/mid_build_selective_probe",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k"), i64_field("b1"), i64_field("b2")]);
                let mut rng = Rng::new(12);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    sparse_keys(&mut rng, n, build_rows, 0.15),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(
                                    &bsch,
                                    vec![
                                        unique_keys(start, n),
                                        i64_values(&mut rng, n),
                                        i64_values(&mut rng, n),
                                    ],
                                );
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![1, 2], JoinKind::Inner),
        );
    }

    // (c) A two-column key over a build side far larger than cache, every
    //     probe row matching exactly one stored pair, one value gathered:
    //     the shape of "line items JOIN supply table ON (part, supplier)".
    //     This is the packed-pair key path end to end.
    {
        let probe_rows = scaled(8_000_000);
        let build_rows = scaled(16_000_000);
        let lane_domain = scaled(1_000_000);
        bench(
            c,
            d,
            "join/packed_pair_big_build",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k1"), i64_field("k2"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k1"), i64_field("k2"), i64_field("cost")]);
                let mut rng = Rng::new(13);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            let ids: Vec<usize> = (0..n).map(|_| rng.below(build_rows)).collect();
                            let k1: Int64Array = ids.iter().map(|&i| spread(i)).collect();
                            let k2: Int64Array =
                                ids.iter().map(|&i| pair_lane(i, lane_domain)).collect();
                            batch(
                                &psch,
                                vec![Arc::new(k1), Arc::new(k2), i64_values(&mut rng, n)],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let k1: Int64Array = (start..start + n).map(spread).collect();
                                let k2: Int64Array = (start..start + n)
                                    .map(|i| pair_lane(i, lane_domain))
                                    .collect();
                                let b = batch(
                                    &bsch,
                                    vec![Arc::new(k1), Arc::new(k2), i64_values(&mut rng, n)],
                                );
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64, DataType::Int64],
            JoinSpec {
                build_key_indices: vec![0, 1],
                probe_key_indices: vec![0, 1],
                probe_fields: vec![Field::new("v", DataType::Int64, false)],
                build_fields: vec![Field::new("cost", DataType::Int64, false)],
                probe_output_indices: vec![2],
                build_output_indices: vec![2],
                kind: JoinKind::Inner,
            },
        );
    }

    // (d) A single-key build far larger than the probe side, every probe row
    //     matching exactly once: the shape of "filtered stream JOIN full
    //     fact table" that a planner sometimes builds from the big side.
    //     Build-phase cost (partition, scatter) is a visible share here.
    {
        let probe_rows = scaled(8_000_000);
        let build_rows = scaled(24_000_000);
        bench(
            c,
            d,
            "join/big_build_full_match",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k"), i64_field("d")]);
                let mut rng = Rng::new(14);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    uniform_keys(&mut rng, n, build_rows),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(
                                    &bsch,
                                    vec![unique_keys(start, n), i64_values(&mut rng, n)],
                                );
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![1], JoinKind::Inner),
        );
    }

    // (e) A build-side outer join: unique build keys, a probe stream hitting
    //     two thirds of them about a dozen times each, the last third only
    //     reached by the unmatched pass. Adds the matched-flag writes to
    //     every drain and the flag scan at the end.
    {
        let probe_rows = scaled(24_000_000);
        let build_rows = scaled(3_000_000);
        bench(
            c,
            d,
            "join/outer_partial_match",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k")]);
                let mut rng = Rng::new(15);
                let matched_domain = build_rows * 2 / 3;
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    uniform_keys(&mut rng, n, matched_domain),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(&bsch, vec![unique_keys(start, n)]);
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![0], JoinKind::BuildOuter),
        );
    }

    // (f) A semi join against a few thousand keys, hit by a fraction of a
    //     percent of a huge probe stream: the shape of "rows whose key is in
    //     a tiny computed set". No columns gather; nearly all time is the
    //     probe pipeline itself.
    {
        let probe_rows = scaled(32_000_000);
        let build_rows = 8_192;
        bench(
            c,
            d,
            "join/semi_tiny_build",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k")]);
                let mut rng = Rng::new(16);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    sparse_keys(&mut rng, n, build_rows, 0.0004),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: vec![batch(&bsch, vec![unique_keys(0, build_rows)])],
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![], JoinKind::ProbeSemi),
        );
    }

    // (g) A large build probed by almost nothing: the shape of "tiny keyed
    //     stream JOIN full dimension" where the planner builds the big side
    //     because its estimates said otherwise. The profile is the build
    //     phase itself - consume, partition, scatter, publish.
    {
        let probe_rows = BATCH_ROWS;
        let build_rows = scaled(12_000_000);
        bench(
            c,
            d,
            "join/build_heavy_small_probe",
            build_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![i64_field("k"), i64_field("b1")]);
                let mut rng = Rng::new(17);
                JoinData {
                    probe: vec![batch(
                        &psch,
                        vec![
                            uniform_keys(&mut rng, probe_rows, build_rows),
                            i64_values(&mut rng, probe_rows),
                        ],
                    )],
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(
                                    &bsch,
                                    vec![unique_keys(start, n), i64_values(&mut rng, n)],
                                );
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![1], JoinKind::Inner),
        );
    }

    // (h) A full-match join whose build side carries a string column that
    //     every output row gathers: the view-rebasing gather path, the shape
    //     of "measure stream JOIN dimension carrying a type/name string".
    {
        let probe_rows = scaled(2_000_000);
        let build_rows = scaled(4_000_000);
        bench(
            c,
            d,
            "join/string_payload_gather",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![
                    i64_field("k"),
                    Field::new("s", DataType::Utf8View, false),
                ]);
                let mut rng = Rng::new(18);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    uniform_keys(&mut rng, n, build_rows),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let b = batch(
                                    &bsch,
                                    vec![unique_keys(start, n), string_values(start, n)],
                                );
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            JoinSpec {
                build_key_indices: vec![0],
                probe_key_indices: vec![0],
                probe_fields: vec![Field::new("v", DataType::Int64, false)],
                build_fields: vec![Field::new("s", DataType::Utf8View, false)],
                probe_output_indices: vec![1],
                build_output_indices: vec![1],
                kind: JoinKind::Inner,
            },
        );
    }

    // (i) The output gather as the dominant cost: a full-match single-key
    //     join whose build side carries six value columns, every one gathered
    //     for every probe row. The probe pipeline is as cheap as it gets
    //     (unique keys, multiplicity one), so most samples land in the
    //     chunked gather itself, reading a build payload far larger than any
    //     cache through a couple thousand stored batches. This is the canary
    //     for per-row costs in the gather path (an extra dependent load per
    //     row shows up here first); the mixed scenarios above dilute such a
    //     change several-fold. Run it at scale >= 1: a smaller build keeps
    //     the payload and the per-batch metadata cache-resident and hides
    //     exactly what it exists to catch.
    {
        let probe_rows = scaled(16_000_000);
        let build_rows = scaled(16_000_000);
        bench(
            c,
            d,
            "join/wide_payload_gather",
            probe_rows,
            move || {
                let psch = schema(vec![i64_field("k"), i64_field("v")]);
                let bsch = schema(vec![
                    i64_field("k"),
                    i64_field("b1"),
                    i64_field("b2"),
                    i64_field("b3"),
                    i64_field("b4"),
                    i64_field("b5"),
                    i64_field("b6"),
                ]);
                let mut rng = Rng::new(19);
                JoinData {
                    probe: batch_sizes(probe_rows)
                        .map(|n| {
                            batch(
                                &psch,
                                vec![
                                    uniform_keys(&mut rng, n, build_rows),
                                    i64_values(&mut rng, n),
                                ],
                            )
                        })
                        .collect(),
                    build: {
                        let mut start = 0;
                        batch_sizes(build_rows)
                            .map(|n| {
                                let mut cols = vec![unique_keys(start, n)];
                                cols.extend((0..6).map(|_| i64_values(&mut rng, n)));
                                let b = batch(&bsch, cols);
                                start += n;
                                b
                            })
                            .collect()
                    },
                }
            },
            vec![DataType::Int64],
            single_key_spec(vec![1], vec![1, 2, 3, 4, 5, 6], JoinKind::Inner),
        );
    }
}

// ---------------------------------------------------------------------------
// Harness: one shared Dispatch (worker pool) for every scenario.
// ---------------------------------------------------------------------------

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    eprintln!(
        "dispatch join benches: {workers} workers, {buffers} ring buffers, scale {}",
        scale()
    );
    if scale() < 1.0 {
        eprintln!(
            "WARNING: scale < 1 shrinks the build sides into cache; \
             comparisons of gather-path or other cache-sensitive changes \
             are NOT valid at this scale"
        );
    }

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let dispatcher = dispatch.dispatcher().clone();

    let mut c = Criterion::default().configure_from_args();
    bench_joins(&mut c, &dispatcher);
    c.final_summary();

    dispatch.exit();
}
