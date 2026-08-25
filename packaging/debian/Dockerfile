# syntax=docker/dockerfile:1.7

# This is a CI-only build environment. Docker is not included in the package
# and is not required to install or run Pivot. Building directly on GitHub's
# Ubuntu runner could link Pivot against a newer glibc than older supported
# distributions provide. Bullseye uses glibc 2.31, which keeps the binary
# compatible with Debian 11+ and Ubuntu 20.04+ while still building natively on
# amd64 and arm64.
FROM rust:1-slim-bullseye AS builder

# The binary keeps Cargo's version, while the Debian package can carry an
# independently ordered release-candidate version supplied by the release
# workflow (for example 0.2.0~rc.20260824.120000+git.abcdef0).
ARG PIVOT_DEB_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake clang libclang-dev pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# `target` is a cache mount, so it is gone by the time the stage is exported:
# copy the binary out to /out inside the same layer. The deploy workflow ships
# that copy and the package below wraps the very same build.
# sharing=locked serializes concurrent builds on one machine instead of
# letting two cargo processes extract crates into the same cache at once,
# which corrupts the registry.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release -p bin --bin pivot \
    && install -D -m 0755 /src/target/release/pivot /out/pivot \
    && PIVOT_BINARY=/out/pivot PIVOT_DEB_VERSION="$PIVOT_DEB_VERSION" \
        packaging/debian/build.sh /packages \
    && packaging/debian/test-package.sh /packages/*.deb

# The `.deb` plus the unstripped binary it was built from. build.sh strips only
# its copy inside the package, so /pivot keeps its symbols for profiling.
FROM scratch AS package
COPY --from=builder /packages/ /
COPY --from=builder /out/pivot /pivot
