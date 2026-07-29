//! Microbenchmarks for the Parquet **encode** path: turning a column's values
//! into a page body.
//!
//! The counterpart to `decode.rs`. Writing a file wraps this in a dataflow, a
//! snappy compression pass and footer assembly, all of which cost more than the
//! encoding itself, so a change inside an encoder is no more visible from the
//! outside there than a decoder's is from a scan. These drive one encoder on
//! its own, through the same `bench-hooks` entry points the decode bench uses.
//!
//! The shapes are the ones a key column takes: values scattered over a wide
//! range, where each difference still needs most of its width, and values that
//! climb, where they pack into a few bits. Strings carry the byte-array
//! encoding, whose lengths pack the same way before the values are laid end to
//! end.
//!
//! ```sh
//! cargo bench --bench encode
//! ```

use arrow_array::StringViewArray;
use criterion::{Criterion, Throughput, black_box};

const PAGE_VALUES: usize = 1 << 16;

/// A small deterministic generator, so a run is repeatable and the numbers can
/// be compared across changes.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0 >> 33
    }
}

fn bench_encode(c: &mut Criterion) {
    let mut rng = Rng(7);
    let scattered: Vec<i64> = (0..PAGE_VALUES)
        .map(|_| 1 + (rng.next() % 20_000_000) as i64)
        .collect();
    let mut climbing = 0i64;
    let sorted: Vec<i64> = (0..PAGE_VALUES)
        .map(|_| {
            climbing += 1 + (rng.next() % 7) as i64;
            climbing
        })
        .collect();
    let strings: Vec<String> = (0..PAGE_VALUES)
        .map(|i| {
            if i.is_multiple_of(4) {
                format!("s{i}")
            } else {
                format!("value number {i:016} with a tail that will not inline")
            }
        })
        .collect();
    let strings = StringViewArray::from(strings.iter().map(|s| s.as_str()).collect::<Vec<_>>());

    let mut g = c.benchmark_group("delta_encode");
    g.throughput(Throughput::Elements(PAGE_VALUES as u64));
    for (name, values) in [("scattered", &scattered), ("sorted", &sorted)] {
        g.bench_function(name, |b| {
            let mut out = Vec::new();
            b.iter(|| {
                black_box(
                    catalog::parquet::decoder_bench_hooks::encode_delta_binary_packed(
                        values, &mut out,
                    ),
                )
            });
        });
    }
    g.bench_function("strings", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            black_box(
                catalog::parquet::decoder_bench_hooks::encode_delta_length_byte_array(
                    &strings, &mut out,
                ),
            )
        });
    });
    g.finish();
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_encode(&mut c);
    c.final_summary();
}
