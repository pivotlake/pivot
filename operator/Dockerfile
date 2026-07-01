# Image for the operator + connection router. Build context is this `operator/`
# directory (the crate is self-contained, no workspace path deps), so it builds
# without the heavy arrow-rs / duckdb submodules the pivot server needs:
#
#   docker build -t pivot-operator:dev operator/
#
# Carries all three binaries (pivot-operator, pivot-router, pivot-crdgen); the
# operator Deployment and the router pods it creates both run from this image.
FROM rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --bins \
 && mkdir -p /out \
 && cp target/release/pivot-operator target/release/pivot-router target/release/pivot-crdgen /out/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/pivot-operator /out/pivot-router /out/pivot-crdgen /usr/local/bin/
ENTRYPOINT ["pivot-operator"]
