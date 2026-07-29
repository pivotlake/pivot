//! Microbenchmarks for the Parquet **decompression + decoding** path — the work
//! that dominates "decode-bound" queries. Profiling the real analytical
//! queries showed that for many of them the dispatch *operator* is a sliver and
//! the wall-time is spent in the scan: snappy-decompressing pages and decoding
//! them into Arrow. For example, a `LIKE '%x%'` scan over
//! the long URL column spent ~73% in `snap::decompress`, and an `ORDER BY
//! <string/time> LIMIT 10` spent 33-46%, with `RleDecoder::read`, the
//! `bytes_view` plain/dict decoders, and `PrimitiveDict` making up most of the
//! rest. None of that is exercised by the `dispatch` operator benches (which run
//! on pre-built in-memory batches), so this bench covers the other half: the
//! column reader.
//!
//! It generates its own SNAPPY + dictionary Parquet file in a tempdir (no
//! dependency on a 14 GB dataset), shaped to the measured statistics of
//! the real columns so the encoding the writer picks — and therefore the decode
//! path the reader takes matches. The `url` column is long (~88 B avg),
//! high-cardinality strings, with too many distinct values per row group to
//! dictionary-encode, so it lands in PLAIN byte-array pages (the long-URL scan
//! shape). `search_phrase` is ~87% empty and otherwise ~31 B, with low distinct
//! per row group, so DICTIONARY + RLE (the search-phrase scan shape).
//! `event_time` is `i64` (the timestamp columns).
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
//! Iteration 1 is a cold read; the rest decode from the warm compressed cache (so the
//! steady-state samples are decompress+decode, not disk IO). Under `perf`, raise
//! the locked-memory limit (`ulimit -l unlimited`) for io_uring setup.

use std::fs::File;
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, black_box};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
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

/// An ascending column starting at `start`, stepping by a small random gap:
/// the shape of a key a table is clustered on.
fn sorted_i64_col(rng: &mut Rng, n: usize, start: i64) -> ArrayRef {
    let mut value = start;
    let values: Vec<i64> = (0..n)
        .map(|_| {
            value += 1 + rng.below(7) as i64;
            value
        })
        .collect();
    Arc::new(Int64Array::from(values)) as ArrayRef
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
        Field::new("scattered_key", DataType::Int64, false),
        Field::new("sorted_key", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true) // writer still falls back to PLAIN when a
        // row group has too many distinct values (the URL case) — exactly what
        // the real file does.
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        // The two key columns carry the delta encoding. Dictionary encoding
        // wins the race for a column unless it is turned off per column, and a
        // key column with millions of distinct values is exactly where a
        // dictionary stops paying, which is why these are written delta packed.
        .set_column_dictionary_enabled("scattered_key".into(), false)
        .set_column_encoding("scattered_key".into(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled("sorted_key".into(), false)
        .set_column_encoding("sorted_key".into(), Encoding::DELTA_BINARY_PACKED)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("data.parquet")).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();

    // Real-data stats: URL ≈ rows×18/100 distinct
    // (~5.5 rows/group), ~88 B; search_phrase ≈ rows×6/100 distinct, ~87% empty,
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
                // A foreign key: distinct values spread over a wide range, so
                // every delta needs its full width (the `l_partkey` shape).
                i64_col(&mut rng, n, 1, 20_000_000),
                // A clustered key: ascending with small gaps, so the deltas
                // pack into a handful of bits (the `l_orderkey` shape).
                sorted_i64_col(&mut rng, n, written as i64 * 4),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        written += n;
    }
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// DELTA_BINARY_PACKED page decoding, on its own
// ---------------------------------------------------------------------------

/// Encode `values` as one `DELTA_BINARY_PACKED` page, the way a writer lays it
/// out: a header, then blocks of `miniblocks` miniblocks, each holding
/// `values_per_miniblock` deltas bit-packed at the narrowest width that fits.
fn encode_delta_page(values: &[i64], values_per_miniblock: usize, miniblocks: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let put_uvarint = |out: &mut Vec<u8>, mut v: u64| loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if v == 0 { byte } else { byte | 0x80 });
        if v == 0 {
            break;
        }
    };
    let zigzag = |v: i64| ((v << 1) ^ (v >> 63)) as u64;

    put_uvarint(&mut out, (values_per_miniblock * miniblocks) as u64);
    put_uvarint(&mut out, miniblocks as u64);
    put_uvarint(&mut out, values.len() as u64);
    put_uvarint(&mut out, zigzag(values[0]));

    let deltas: Vec<i64> = values.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
    for block in deltas.chunks(values_per_miniblock * miniblocks) {
        let min_delta = *block.iter().min().unwrap();
        put_uvarint(&mut out, zigzag(min_delta));
        let widths: Vec<u8> = (0..miniblocks)
            .map(|i| {
                let start = (i * values_per_miniblock).min(block.len());
                let end = (start + values_per_miniblock).min(block.len());
                block[start..end]
                    .iter()
                    .map(|d| 64 - d.wrapping_sub(min_delta).leading_zeros() as u8)
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        out.extend_from_slice(&widths);
        for (i, width) in widths.iter().enumerate() {
            let start = (i * values_per_miniblock).min(block.len());
            let end = (start + values_per_miniblock).min(block.len());
            let mut bits = vec![0u8; values_per_miniblock * *width as usize / 8];
            for (j, delta) in block[start..end].iter().enumerate() {
                let packed = delta.wrapping_sub(min_delta) as u64;
                for bit in 0..*width as usize {
                    if packed >> bit & 1 == 1 {
                        let pos = j * *width as usize + bit;
                        bits[pos / 8] |= 1 << (pos % 8);
                    }
                }
            }
            out.extend_from_slice(&bits);
        }
    }
    out
}

/// Decode-only benchmark: no IO, no decompression, no Arrow materialisation
/// beyond the builder the decoder writes into. Two shapes, because the width
/// of the packed deltas is what the decode loop's cost tracks: a scattered key
/// packs at ~25 bits, a clustered one at ~3.
fn bench_delta_pages(c: &mut Criterion) {
    const PAGE_VALUES: usize = 1 << 16;
    let mut rng = Rng::new(7);
    let scattered: Vec<i64> = (0..PAGE_VALUES)
        .map(|_| 1 + rng.below(20_000_000) as i64)
        .collect();
    let mut running = 0i64;
    let sorted: Vec<i64> = (0..PAGE_VALUES)
        .map(|_| {
            running += 1 + rng.below(7) as i64;
            running
        })
        .collect();

    // A string column of the shape a writer sends down this path: values a
    // little either side of the twelve bytes a view can inline, so both the
    // block-referencing and the inlining branch are exercised.
    let strings: Vec<String> = (0..PAGE_VALUES)
        .map(|i| {
            if i.is_multiple_of(4) {
                format!("s{i}")
            } else {
                format!("value number {i:016} with a tail that will not inline")
            }
        })
        .collect();

    dispatch::memory::init_test_free_pool(64);
    let mut g = c.benchmark_group("delta_page");
    g.throughput(Throughput::Elements(PAGE_VALUES as u64));
    for (name, values) in [("scattered", &scattered), ("sorted", &sorted)] {
        let page = vec![bytes::Bytes::from(encode_delta_page(values, 32, 4))];
        g.bench_function(name, |b| {
            let mut allocator = dispatch::memory::SlabAllocator::new(true);
            b.iter(|| {
                black_box(
                    catalog::parquet::decoder_bench_hooks::decode_delta_binary_packed(
                        &page,
                        PAGE_VALUES,
                        &mut allocator,
                    ),
                )
            });
        });
    }
    // The string encoding: delta-packed lengths followed by the bytes.
    let lengths: Vec<i64> = strings.iter().map(|s| s.len() as i64).collect();
    let mut page_bytes = encode_delta_page(&lengths, 32, 4);
    for s in &strings {
        page_bytes.extend_from_slice(s.as_bytes());
    }
    let string_page = vec![bytes::Bytes::from(page_bytes)];
    g.bench_function("strings", |b| {
        let mut allocator = dispatch::memory::SlabAllocator::new(true);
        b.iter(|| {
            black_box(
                catalog::parquet::decoder_bench_hooks::decode_delta_length_byte_array(
                    &string_page,
                    PAGE_VALUES,
                    &mut allocator,
                ),
            )
        });
    });
    g.finish();
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
    // `LIKE` scan: decompress-dominated).
    scenario("decode/url", Projection::columns([0]));
    // Mostly-empty, low-distinct strings → DICTIONARY + RLE + snappy (the
    // search_phrase / order-by scan).
    scenario("decode/search_phrase", Projection::columns([1]));
    // `i64` timestamps (the event_time column).
    scenario("decode/event_time", Projection::columns([2]));
    // DELTA_BINARY_PACKED at full width: scattered keys, so every delta uses
    // all the bits its miniblock allows.
    scenario("decode/delta_scattered", Projection::columns([3]));
    // DELTA_BINARY_PACKED at a narrow width: a clustered key whose deltas fit
    // in a few bits, so the unpacking loop does the most work per byte read.
    scenario("decode/delta_sorted", Projection::columns([4]));
    // Every column at once — a fuller scan.
    scenario("decode/all", Projection::all(5));
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
        ParquetTable::from_directory(dispatch.dispatcher(), dir.path(), &[])
            .expect("ParquetTable::from_directory failed"),
    );

    let mut c = Criterion::default().configure_from_args();
    bench_decode(&mut c, &dispatch, &table, rows);
    bench_delta_pages(&mut c);
    c.final_summary();

    dispatch.exit();
}
