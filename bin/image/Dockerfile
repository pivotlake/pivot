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
# entrypoint generates on the first boot of a fresh /var/lib/pivot and the
# server then owns and rewrites, so CREATE USER works. The generated default
# datastore comes from PIVOT_DATASTORE: a local directory when unset, or an
# object store with, for example,
#
#   docker run -e PIVOT_DATASTORE=s3://bucket/prefix \
#     -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
#     -e AWS_REGION=... pivotlake/pivot
#
# (an AWS_ENDPOINT_URL selects a path-style S3-compatible endpoint like
# MinIO). A gs:// location carries no credentials of its own; the server
# follows the ambient Google credentials chain, so mount a service-account
# key file,
#
#   docker run -e PIVOT_DATASTORE=gs://bucket/prefix \
#     -v /path/key.json:/etc/pivot/gcs-key.json:ro \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/etc/pivot/gcs-key.json pivotlake/pivot
#
# (keep both set on every run), or set nothing on Google compute and let
# workload identity answer. As with the PostgreSQL image's POSTGRES_*
# variables, the generation-time variables apply
# only while the metastore file does not exist yet: a mount over
# /var/lib/pivot that brings its own metastore.yaml is served as-is, and one
# without gets a file generated into it. Mount a file over
# /etc/pivot/pivot.yaml to reconfigure the instance settings.
#
# PIVOT_DISK_CACHE_SIZE (for example 64g) caches remote object-store reads
# on local disk under /var/cache/pivot, sized to the variable. Unlike the
# generation-time variables it applies on every boot, by writing the
# `disk_cache` section of /etc/pivot/pivot.yaml; a mounted config should
# carry its own disk_cache section instead of setting this.
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY --chmod=644 pivot.yaml /etc/pivot/pivot.yaml
RUN mkdir -p /var/lib/pivot/datastores/default

ENV RUST_LOG=info
# 5432 is the pgwire endpoint, 8081 the web dashboard.
EXPOSE 5432 8081
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["pivot", "server", "--config", "/etc/pivot/pivot.yaml", "--metastore-file", "/var/lib/pivot/metastore.yaml"]
