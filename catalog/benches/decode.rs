//! Microbenchmarks for the Parquet **decompression + decoding** path — the work
//! that dominates "decode-bound" queries. Profiling the real ClickBench-style
//! queries showed that for many of them the dispatch *operator* is a sliver and
//! the wall-time is spent in the scan: snappy-decompressing pages and decoding
//! them into Arrow. Examples measured on a c8g.4xlarge:
//!   * a `LIKE '%x%'` scan over the long URL column — ~73% `snap::decompress`,
//!   * `ORDER BY <string/time> LIMIT 10` — 33–46% `snap::decompress`,
//! with `RleDecoder::read`, the `bytes_view` plain/dict decoders, and
//! `PrimitiveDict` making up most of the rest. None of that is exercised by the
//! `dispatch` operator benches (which run on pre-built in-memory batches), so
//! this bench covers the other half: the column reader.
//!
//! It generates its own SNAPPY + dictionary Parquet file in a tempdir (no
//! dependency on the 14 GB hits dataset), shaped to the measured statistics of
//! the real columns so the encoding the writer picks — and therefore the decode
//! path the reader takes — matches:
//!   * `url`          — long (~88 B avg), high-cardinality strings → too many
//!                      distinct per row group to dictionary-encode, so PLAIN
//!                      byte-array pages (the URL/q20/q33 scan shape).
//!   * `search_phrase`— ~87% empty, the rest ~31 B; low distinct per row group →
//!                      DICTIONARY + RLE (the SearchPhrase/q12/q24 scan shape).
//!   * `event_time`   — `i64` (the timestamp columns behind q24/q26/q42).
//!
//! Then it scans one column at a time with `table_input(..).collect()`, driving
//! fetch → snappy-decompress → decode → Arrow materialize.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench decode
//! cargo bench --bench decode -- "decode/url" --profile-time 20   # attach perf
//! ```
//!
//! Env: `PIVOT_BENCH_ROWS` (default 4M), `PIVOT_BENCH_WORKERS` (default all
//! cores), `PIVOT_BENCH_BUFFERS` (ring/file-cache slots of 2 MiB, default 1024).
//! Iteration 1 is a cold read; the rest decode from the warm file cache (so the
//! steady-state samples are decompress+decode, not disk IO). Under `perf`, raise
//! the locked-memory limit (`ulimit -l unlimited`) for io_uring setup.

use std::fs::File;
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, black_box};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::parquet::{ParquetTable, table_input};
use dispatch::{Dispatch, Projection};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_ROWS: usize = 4_000_000;
/// Rows per row group — many row groups so the scan fans across all workers,
/// like a real multi-row-group file.
const ROW_GROUP_ROWS: usize = 256 * 1024;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn total_rows() -> usize {
    env_usize("PIVOT_BENCH_ROWS", DEFAULT_ROWS)
}
fn worker_count() -> usize {
    env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}
fn ring_buffers() -> usize {
    env_usize("PIVOT_BENCH_BUFFERS", 1024)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64)
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

/// Dictionary of distinct strings: a short shared lead then a per-entry token so
/// they diverge early, with lengths varying around `avg_len` — the realistic
/// shape (see the dispatch operator bench for why this matters for the decoder).
fn string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    let lead: String = prefix.bytes().take(7).map(|b| b as char).collect();
    (0..n_distinct.max(1))
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

// ---------------------------------------------------------------------------
// Synthetic Parquet generation
// ---------------------------------------------------------------------------

fn string_col(rng: &mut Rng, n: usize, dict: &[String]) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    for _ in 0..n {
        b.append_value(&dict[rng.below(dict.len())]);
    }
    Arc::new(b.finish())
}

fn sparse_string_col(rng: &mut Rng, n: usize, dict: &[String], empty_fraction: f64) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    for _ in 0..n {
        if rng.chance(empty_fraction) {
            b.append_value("");
        } else {
            b.append_value(&dict[rng.below(dict.len())]);
        }
    }
    Arc::new(b.finish())
}

fn i64_col(rng: &mut Rng, n: usize, base: i64, span: usize) -> ArrayRef {
    let v: Int64Array = (0..n).map(|_| base + rng.below(span) as i64).collect();
    Arc::new(v)
}

/// Write a SNAPPY + dictionary Parquet file (one column per decode shape) into
/// `dir`, `rows` total across `ROW_GROUP_ROWS`-sized row groups. Returns the
/// schema column order: 0=url, 1=search_phrase, 2=event_time.
fn write_parquet(dir: &TempDir, rows: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("url", DataType::Utf8View, false),
        Field::new("search_phrase", DataType::Utf8View, false),
        Field::new("event_time", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true) // writer still falls back to PLAIN when a
        // row group has too many distinct values (the URL case) — exactly what
        // the real file does.
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("data.parquet")).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();

    // Real-data stats (measured on the hits dataset): URL ≈ rows×18/100 distinct
    // (~5.5 rows/group), ~88 B; SearchPhrase ≈ rows×6/100 distinct, ~87% empty,
    // ~31 B non-empty.
    let url_dict = string_dict((rows * 18 / 100).max(1), "http://example.com/path", 88);
    let phrase_dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
    let mut rng = Rng::new(1);

    let mut written = 0;
    while written < rows {
        let n = ROW_GROUP_ROWS.min(rows - written);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                string_col(&mut rng, n, &url_dict),
                sparse_string_col(&mut rng, n, &phrase_dict, 0.868),
                i64_col(&mut rng, n, 1_700_000_000_000, 86_400_000),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        written += n;
    }
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion, dispatch: &Dispatch, table: &Arc<ParquetTable>, rows: usize) {
    let d = dispatch.dispatcher();
    let mut scenario = |name: &str, proj: Projection| {
        let mut g = c.benchmark_group("decode");
        g.throughput(Throughput::Elements(rows as u64));
        g.bench_function(name, |b| {
            b.iter(|| {
                let out = table_input(d, table, proj.clone(), false)
                    .collect()
                    .unwrap();
                black_box(out);
            });
        });
        g.finish();
    };

    // Long high-cardinality strings → PLAIN byte-array pages + snappy (the URL /
    // `LIKE` / q33 scan: decompress-dominated).
    scenario("decode/url", Projection::columns([0]));
    // Mostly-empty, low-distinct strings → DICTIONARY + RLE + snappy (the
    // SearchPhrase / order-by q12/q24/q25 scan).
    scenario("decode/search_phrase", Projection::columns([1]));
    // `i64` timestamps (the EventTime column behind q24/q26/q42).
    scenario("decode/event_time", Projection::columns([2]));
    // All three columns at once — a fuller scan.
    scenario("decode/all", Projection::all(3));
}

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    let rows = total_rows();
    eprintln!("catalog decode benches: {workers} workers, {buffers} buffers, {rows} rows");

    let dispatch = Dispatch::spin_up(workers, buffers, None);

    // Generate the synthetic parquet once, then load the table (the metadata
    // fetch runs on the coordinator, like the test helper).
    let dir = TempDir::new().unwrap();
    write_parquet(&dir, rows);
    let table = Arc::new(
        ParquetTable::from_directory(dispatch.dispatcher(), dir.path())
            .expect("ParquetTable::from_directory failed"),
    );

    let mut c = Criterion::default().configure_from_args();
    bench_decode(&mut c, &dispatch, &table, rows);
    c.final_summary();

    dispatch.exit();
}
