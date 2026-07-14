# syntax=docker/dockerfile:1.6
#
# Build context is the workspace root (see .github/workflows/bench.yml).
# Submodules (arrow-rs, duckdb-planner/duckdb-sources) must be checked out
# before docker build — they are referenced by path = "..." in Cargo.toml.

FROM rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake clang libclang-dev pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Bypass the kache rustc wrapper (.cargo/config.toml): kache is not installed
# in the image, and release binaries are always compiled fresh, never cached.
ENV RUSTC_WRAPPER=

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p benchmarks --bin pivot-bench \
 && cp target/release/pivot-bench /usr/local/bin/pivot-bench

FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

WORKDIR /app
COPY --from=builder /usr/local/bin/pivot-bench /usr/local/bin/pivot-bench
COPY benchmarks/clickbench /app/benchmarks/clickbench

ENV RUST_LOG=info
ENTRYPOINT ["pivot-bench"]
