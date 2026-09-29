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
set -eu

# Flags alone select the server command with exactly those flags, so
# `docker run pivotlake/pivot --config /my/pivot.yaml` keeps working; no
# arguments at all fall back to the image's default command.
if [ $# -eq 0 ]; then
  set -- pivot server --config /etc/pivot/pivot.yaml
elif [ "${1#-}" != "$1" ]; then
  set -- pivot server "$@"
fi

# The config names the metastore file at this path, fixed by the image, so
# the entrypoint knows it without reading the config.
METASTORE_FILE=/var/lib/pivot/metastore.yaml

generate_metastore_file() {
  location="${PIVOT_DATASTORE:-/var/lib/pivot/datastores/default}"
  echo "generating $METASTORE_FILE with the default datastore at $location" >&2
  {
    echo "datastores:"
    echo "  default:"
    echo "    kind: pivotlake"
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
fi

exec "$@"
