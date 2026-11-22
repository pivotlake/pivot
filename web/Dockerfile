# syntax=docker/dockerfile:1.7

# Image for the pivot web page: one process, the backend, which serves the
# built page and reads a pivot server for it.
#
# Unlike the server image (bin/image), this one compiles what it ships. The
# page is a Vite bundle and the backend a small Rust crate, a few minutes on a
# hosted runner, and neither is released any other way.
#
# The context is the repository root: the page reads the version it shows from
# the workspace's Cargo.toml, and the backend builds with the Rust that
# rust-toolchain.toml pins.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f web/Dockerfile .
#
# The image has to be told where the pivot server is. Loopback inside a
# container reaches nothing, so unless the container shares the host's
# network, name the server:
#
#   docker run -p 8080:8080 -e PIVOT_HOST=pivot.internal pivotlake/pivot-web
#
# PIVOT_PORT, PIVOT_USER and PIVOT_PASSWORD say the rest of how to log in, and
# PIVOT_COMPACT_BYTES and PIVOT_COMPACT_MIN_FILES tell the page the server's
# compaction settings when they are not the defaults.

# The page is the same files on every platform, so it is built once, on the
# builder's own platform.
FROM --platform=$BUILDPLATFORM node:22-bookworm-slim AS page
WORKDIR /src
COPY Cargo.toml ./
COPY web/package.json web/package-lock.json web/
RUN npm --prefix web ci
COPY web/index.html web/tsconfig.json web/vite.config.ts web/
COPY web/public web/public
COPY web/src web/src
RUN npm --prefix web run build

# The backend is compiled on the builder's own platform for the image's, so
# the other architecture's build is a cross-compile, not an emulated one. The
# crate is pure Rust, so crossing needs only the target's linker and C
# library.
FROM --platform=$BUILDPLATFORM rust:1-bookworm AS backend
ARG TARGETARCH
ARG BUILDARCH
WORKDIR /src
COPY rust-toolchain.toml ./
RUN set -eu; \
    case "$TARGETARCH" in \
      amd64) rust_target=x86_64-unknown-linux-gnu; gnu_target=x86_64-linux-gnu; cross_packages="gcc-x86-64-linux-gnu libc6-dev-amd64-cross" ;; \
      arm64) rust_target=aarch64-unknown-linux-gnu; gnu_target=aarch64-linux-gnu; cross_packages="gcc-aarch64-linux-gnu libc6-dev-arm64-cross" ;; \
      *) echo "no pivot-web build for $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    echo "$rust_target" > /rust-target; \
    # Run where rust-toolchain.toml is, this installs the pinned Rust first.
    rustup target add "$rust_target"; \
    if [ "$TARGETARCH" != "$BUILDARCH" ]; then \
      apt-get update; \
      apt-get install -y --no-install-recommends $cross_packages; \
      rm -rf /var/lib/apt/lists/*; \
      mkdir -p /usr/local/cargo; \
      printf '[target.%s]\nlinker = "%s-gcc"\n' "$rust_target" "$gnu_target" >> /usr/local/cargo/config.toml; \
    fi
COPY web/server web/server
RUN cargo build --release --locked --manifest-path web/server/Cargo.toml --target "$(cat /rust-target)" \
    && cp "web/server/target/$(cat /rust-target)/release/pivot-web" /pivot-web

FROM debian:bookworm-slim
COPY --from=backend /pivot-web /usr/local/bin/pivot-web
COPY --from=page /src/web/dist /usr/share/pivot-web

# Beyond loopback, so a published container port reaches the page, and with
# the page's files, so this one process is all of it.
ENV PIVOT_WEB_LISTEN=0.0.0.0:8080 \
    PIVOT_WEB_STATIC_DIR=/usr/share/pivot-web

# It reads a pivot server and serves files, and writes nothing.
USER nobody
EXPOSE 8080
ENTRYPOINT ["pivot-web"]
