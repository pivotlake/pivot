# syntax=docker/dockerfile:1.6
#
# Image for the pivotdb server (pivot).
#
# Build context is the workspace root (not this directory), because the crate
# has path = "..." deps on the arrow-rs and duckdb submodules. Submodules must
# be checked out before docker build. Build from the repo root with:
#
#   docker build -t pivot -f server/Dockerfile .

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
    cargo build --release -p server --bin pivotdb-server \
 && cp target/release/pivotdb-server /usr/local/bin/pivotdb-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/pivotdb-server /usr/local/bin/pivotdb-server

ENV RUST_LOG=info
# Bind to all interfaces so the pgwire port is reachable from outside the
# container (the binary defaults to 127.0.0.1, which is container-local only).
EXPOSE 5432
ENTRYPOINT ["pivotdb-server"]
CMD ["--bind", "0.0.0.0:5432"]
