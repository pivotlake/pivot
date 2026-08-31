#!/bin/sh
# Entrypoint for the pivotlake/pivot image.
#
# On the first boot of a fresh /var/lib/pivot (no metastore.yaml yet), the
# server's initial metastore file is generated here from the environment: the
# default datastore at PIVOT_DATASTORE (a local path, or an s3:// or gs://
# URI; a local directory when unset), plus the trusted `pivot` user. An s3://
# location also captures the AWS_* variables as the secret that signs its
# requests. A gs:// location carries no secret: the server follows the
# ambient Google credentials chain, so mount a service-account key file and
# keep GOOGLE_APPLICATION_CREDENTIALS pointing at it on every run, or set
# nothing on Google compute and let workload identity answer.
#
# Once the file exists these variables are ignored: the file is the server's
# own to rewrite (CREATE USER lands there), and a mounted /var/lib/pivot that
# already brings one is served as-is. To point an initialised volume
# somewhere else, edit its metastore.yaml directly.
#
# PIVOT_DISK_CACHE_SIZE (for example 64g) turns on the on-disk cache for
# remote (object store) reads by writing a `disk_cache` section, sized to the
# variable and caching under /var/cache/pivot, into the server section of
# /etc/pivot/pivot.yaml. The write happens on every server boot but is
# idempotent, and a config that already carries its own disk_cache section is
# refused rather than rewritten: set the size there instead.
set -eu

# Flags alone select the server command with exactly those flags, so
# `docker run pivotlake/pivot --config /my/pivot.yaml` keeps working; no
# arguments at all fall back to the image's default command.
if [ $# -eq 0 ]; then
  set -- pivot server \
    --config /etc/pivot/pivot.yaml \
    --metastore-file /var/lib/pivot/metastore.yaml
elif [ "${1#-}" != "$1" ]; then
  set -- pivot server "$@"
fi

METASTORE_FILE=/var/lib/pivot/metastore.yaml
CONFIG_FILE=/etc/pivot/pivot.yaml
DISK_CACHE_DIR=/var/cache/pivot

configure_disk_cache() {
  if grep -q '^  disk_cache:' "$CONFIG_FILE"; then
    # A previous boot of this container already wrote the section (the
    # container's environment cannot have changed since). Anything else is a
    # config bringing its own disk_cache, which the environment must not
    # rewrite behind the operator's back.
    if grep -qF "    dir: $DISK_CACHE_DIR" "$CONFIG_FILE" &&
      grep -qF "    size: \"$PIVOT_DISK_CACHE_SIZE\"" "$CONFIG_FILE"; then
      return 0
    fi
    echo "PIVOT_DISK_CACHE_SIZE is set, but $CONFIG_FILE already configures server.disk_cache; set the size there instead" >&2
    exit 1
  fi
  if ! grep -q '^server:' "$CONFIG_FILE"; then
    echo "PIVOT_DISK_CACHE_SIZE is set, but $CONFIG_FILE has no top-level server section to hold the disk cache" >&2
    exit 1
  fi
  echo "enabling the $PIVOT_DISK_CACHE_SIZE disk cache at $DISK_CACHE_DIR in $CONFIG_FILE" >&2
  awk -v dir="$DISK_CACHE_DIR" -v size="$PIVOT_DISK_CACHE_SIZE" '
    { print }
    /^server:/ {
      print "  disk_cache:"
      print "    dir: " dir
      print "    size: \"" size "\""
    }
  ' "$CONFIG_FILE" >"$CONFIG_FILE.tmp"
  # Replacing the file (rather than editing it in place) means a config
  # bind-mounted over this path fails loudly here, instead of the entrypoint
  # quietly rewriting a file on the operator's host.
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
}

generate_metastore_file() {
  location="${PIVOT_DATASTORE:-/var/lib/pivot/datastores/default}"
  echo "generating $METASTORE_FILE with the default datastore at $location" >&2
  {
    echo "datastores:"
    echo "  default:"
    echo "    kind: pivot"
    echo "    location: \"$location\""
    echo "    default: true"
    case "$location" in
    s3://* | s3a://*)
      : "${AWS_ACCESS_KEY_ID:?is required for an s3:// PIVOT_DATASTORE}"
      : "${AWS_SECRET_ACCESS_KEY:?is required for an s3:// PIVOT_DATASTORE}"
      echo "secrets:"
      echo "  default:"
      echo "    type: s3"
      echo "    scope: \"$location\""
      echo "    region: \"${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}\""
      echo "    access_key_id: \"$AWS_ACCESS_KEY_ID\""
      echo "    secret_access_key: \"$AWS_SECRET_ACCESS_KEY\""
      if [ -n "${AWS_ENDPOINT_URL:-}" ]; then
        echo "    endpoint: \"$AWS_ENDPOINT_URL\""
      fi
      ;;
    esac
    echo "users:"
    echo "  pivot:"
    echo "    auth:"
    echo "      method: trust"
  } >"$METASTORE_FILE"
  # The file may carry the S3 keys, so only the server process should read it.
  chmod 600 "$METASTORE_FILE"
}

if [ "$1" = "pivot" ] && [ "${2:-}" = "server" ]; then
  if [ ! -f "$METASTORE_FILE" ]; then
    generate_metastore_file
  fi
  if [ -n "${PIVOT_DISK_CACHE_SIZE:-}" ]; then
    configure_disk_cache
  fi
fi

exec "$@"
