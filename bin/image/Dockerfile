# syntax=docker/dockerfile:1.7

# Runtime image for the pivotdb server (pivot), assembled from prebuilt
# release binaries. Nothing is compiled here: the deploy workflow drops the
# released binary for each platform into this directory and builds a
# multi-arch image from it, so the pushed image can never drift from the
# released server.
#
# Expected context layout (TARGETARCH is amd64 or arm64):
#
#   bin/image/Dockerfile
#   bin/image/amd64/pivot
#   bin/image/arm64/pivot
#
#   docker buildx build --platform linux/amd64,linux/arm64 bin/image
#
# For a local single-arch image, build the binary and drop it in:
#
#   cargo build --release -p bin --bin pivot
#   mkdir -p bin/image/amd64 && cp target/release/pivot bin/image/amd64/
#   docker build -t pivot bin/image

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The binary arrives through an artifact download, which drops permission
# bits, so the executable bit is restored on COPY.
ARG TARGETARCH
COPY --chmod=755 ${TARGETARCH}/pivot /usr/local/bin/pivot

# The image runs out of the box. The config carries only the instance
# settings; the datastores and users live in the metastore file, which the
# server owns and rewrites, so CREATE USER works. Mount a file over
# /etc/pivot/pivot.yaml to reconfigure. A mount over /var/lib/pivot (to keep
# data across containers) replaces this layer's content, so it must bring its
# own metastore.yaml; the server refuses to start without it.
COPY --chmod=644 pivot.yaml /etc/pivot/pivot.yaml
COPY --chmod=644 metastore.yaml /var/lib/pivot/metastore.yaml
RUN mkdir -p /var/lib/pivot/datastores/default

ENV RUST_LOG=info
# 5432 is the pgwire endpoint, 8081 the web dashboard.
EXPOSE 5432 8081
ENTRYPOINT ["pivot", "server"]
CMD ["--config", "/etc/pivot/pivot.yaml", "--metastore-file", "/var/lib/pivot/metastore.yaml"]
