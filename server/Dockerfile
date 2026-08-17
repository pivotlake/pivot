# syntax=docker/dockerfile:1.6
#
# Image for the pivotdb server (pivot).
#
# Build context is the workspace root (not this directory), because the crate
# has path = "..." deps on the arrow-rs and duckdb submodules. Submodules must
# be checked out before docker build. Build from the repo root with:
#
#   docker build -t pivot -f server/Dockerfile .
#
# BINARY_SOURCE picks where /usr/local/bin/pivotdb-server comes from:
#   builder (default)  compile it in-image, as above
#   prebuilt           copy the file named by PREBUILT_BIN from the build
#                      context. deploy.yml passes the PGO-optimized release
#                      binary this way, so the image ships the same artifact
#                      the GCS upload does instead of a plain in-image build.
ARG BINARY_SOURCE=builder

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

FROM scratch AS prebuilt
ARG PREBUILT_BIN
# --chmod because the binary may arrive through a CI artifact download, which
# drops the executable bit.
COPY --chmod=755 ${PREBUILT_BIN} /usr/local/bin/pivotdb-server

# Alias resolving to whichever stage carries the binary; the unused stage is
# never built.
FROM ${BINARY_SOURCE} AS binary

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=binary /usr/local/bin/pivotdb-server /usr/local/bin/pivotdb-server

ENV RUST_LOG=info
# Mount a config file and point the server at it. Set `server.bind` to
# 0.0.0.0:5432 there so the pgwire port is reachable from outside the container
# (the default, 127.0.0.1, is container-local only).
EXPOSE 5432
ENTRYPOINT ["pivotdb-server"]
CMD ["--config", "/etc/pivot/pivot.yaml"]
