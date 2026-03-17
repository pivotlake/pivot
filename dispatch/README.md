# dispatch

A high-throughput, low-latency parallel dataflow execution engine. One worker per core, local-first with work-stealing.

See `src/lib.rs` or our [docs](https://epsio-labs.github.io/pivotdb/dispatch/index.html) for full documentation.

## Running

```sh
# Unit tests
cargo test

# Integration tests
cargo test --test integration
```

## Benchmarking & PGO (Profile-Guided Optimization)

It is highly recommended to run benchmarks with PGO — minor things such as extracting
a function or even just moving code can cause up to ~5% degradations; PGO negates most of these effects,
but not all! Stay vigilant :)

To run benchmarks WITHOUT pgo:
```sh
# Benchmarks (ClickBench) — all queries by default, or pick specific ones
SOURCE_DIRECTORY=/path/to/hits cargo bench --bench clickbench
QUERY=7,33 SOURCE_DIRECTORY=/path/to/hits cargo bench --bench clickbench
```

To run with PGO, first install [just](https://github.com/casey/just) (`cargo install just`).
Next, you'll need to generate profiles, and then run with them.
Note that as long as you don't clear your profiles, pgo can continue using previous ones
even if you make code changes- there will just be more misses and it will fallback to normal optimizations.
```sh
# 1. Generate profiles
# This would run once for all queries- this can take a few minutes
just pgo-gen bench --bench clickbench
# You can choose certain queries to optimize on so gen takes less time
QUERY=33,23 just pgo-gen --bench clickbench

# 2. Run with optimized build- this should hopefully NOT take a few minutes :)
QUERY=33,23 QUERY_TEST_COUNT=3 just pgo-use bench --bench clickbench
```
