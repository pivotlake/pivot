# syntax=docker/dockerfile:1.7

# This is a CI-only build environment. Docker is not included in the package
# and is not required to install or run Pivot. Building directly on GitHub's
# Ubuntu runner could link Pivot against a newer glibc than older supported
# distributions provide. Bullseye uses glibc 2.31, which keeps the binary
# compatible with Debian 11+ and Ubuntu 20.04+ while still building natively on
# amd64 and arm64.
#
# The release binary is PGO-built in two passes around a training run that the
# release workflow drives on the runner host:
#
#   --target instrumented   instrumented `pivot` + the `pivot-bench` client
#   (host)                  packaging/pgo/train.sh -> pgo-profiles/*.profraw
#   --target package        merge, profile-use build, verify, package
#
# The package target refuses to build when `pgo-profiles/` holds no profiles:
# a release binary silently shipped without its profile would only surface as
# a mystery regression.
FROM rust:1-slim-bullseye AS builder

# lld: the instrumented link can outgrow the 128MB aarch64 branch range, which
# GNU ld fails with relocation overflows. llvm-tools: llvm-profdata, matched to
# the rustc that instruments, for the merge and the applied-profile check.
# just runs the shared PGO recipes (pgo.just); Bullseye does not package it,
# and installing from crates.io adds no host the build does not already trust.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake clang libclang-dev lld pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add llvm-tools
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo install just@1.36.0 --locked

WORKDIR /src
COPY . .

# Each build flavor keeps its own target dir: RUSTFLAGS are part of cargo's
# fingerprint, so a shared dir would rebuild the world on every flavor switch.
# `target` dirs are cache mounts, so they are gone by the time a stage is
# exported: binaries are copied out to /out inside the same layer.
# sharing=locked serializes concurrent builds on one machine instead of
# letting two cargo processes extract crates into the same cache at once,
# which corrupts the registry.
#
# The PGO recipes are reached through benchmarks/justfile (the same importer
# the perf flows use), and just runs them from that justfile's directory, so
# every target dir lives under /src/benchmarks/.
FROM builder AS builder-instrumented
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/benchmarks/target-pgogen,sharing=locked \
    --mount=type=cache,target=/src/benchmarks/target-client,sharing=locked \
    just --justfile benchmarks/justfile pgo-gen-build build --release -p bin --bin pivot \
    # The client that drives the training workload. Plain release: it is not
    # shipped and not profiled, and the floor comes from .cargo/config.toml.
    && CARGO_TARGET_DIR=benchmarks/target-client cargo build --release -p benchmarks --bin pivot-bench \
    && install -D -m 0755 \
        "/src/benchmarks/target-pgogen/$(rustc -vV | sed -n 's/^host: //p')/release/pivot" \
        /out/pivot \
    && install -D -m 0755 /src/benchmarks/target-client/release/pivot-bench /out/pivot-bench

# The instrumented server and the client that drives its training workload.
FROM scratch AS instrumented
COPY --from=builder-instrumented /out/ /

# The binary keeps Cargo's version, while the Debian package can carry an
# independently ordered release-candidate version supplied by the release
# workflow (for example 0.2.0~rc.20260824.120000+git.abcdef0).
FROM builder AS builder-release
ARG PIVOT_DEB_VERSION
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/benchmarks/target-pgouse,sharing=locked \
    just --justfile benchmarks/justfile pgo-merge /tmp/merged.profdata /src/pgo-profiles \
    && just --justfile benchmarks/justfile pgo-use-with /tmp/merged.profdata build --release -p bin --bin pivot \
    && install -D -m 0755 \
        "/src/benchmarks/target-pgouse/$(rustc -vV | sed -n 's/^host: //p')/release/pivot" \
        /out/pivot \
    && just --justfile benchmarks/justfile pgo-verify /out/pivot /tmp/merged.profdata \
    && PIVOT_BINARY=/out/pivot PIVOT_DEB_VERSION="$PIVOT_DEB_VERSION" \
        packaging/debian/build.sh /packages \
    && packaging/debian/test-package.sh /packages/*.deb

# The `.deb` plus the unstripped binary it was built from. build.sh strips only
# its copy inside the package, so /pivot keeps its symbols for profiling.
FROM scratch AS package
COPY --from=builder-release /packages/ /
COPY --from=builder-release /out/pivot /pivot
