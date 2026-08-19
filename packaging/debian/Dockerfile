# syntax=docker/dockerfile:1.7

# This is a CI-only build environment. Docker is not included in the package
# and is not required to install or run Pivot. Building directly on GitHub's
# Ubuntu runner could link Pivot against a newer glibc than older supported
# distributions provide. Bullseye uses glibc 2.31, which keeps the binary
# compatible with Debian 11+ and Ubuntu 20.04+ while still building natively on
# amd64 and arm64.
FROM rust:1-slim-bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake clang libclang-dev pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
ENV RUSTC_WRAPPER=

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p cli --bin pivot \
    && PIVOT_BINARY=/src/target/release/pivot packaging/debian/build.sh /packages \
    && packaging/debian/test-package.sh /packages/*.deb

FROM scratch AS package
COPY --from=builder /packages/ /
