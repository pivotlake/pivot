#!/bin/sh
# Entrypoint for the pivotlake/pivot image.
#
# On the first boot of a fresh /var/lib/pivot (no metastore.yaml yet), the
# server's initial metastore file is generated here from the environment: the
# default datastore at PIVOT_DATASTORE (a local path, or an s3:// or gs://
# URI; a local directory when unset), plus the trusted `pivot` user. An s3://
# location also gets a secret naming its region and endpoint. Long-lived keys
# in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are captured in that secret
# and sign the location's requests from then on. Without them, or with a
# session's keys (AWS_SESSION_TOKEN is set, and they expire), the secret
# carries no keys and the server signs as the identity each run gives it: that
# run's AWS_* variables, or the role of the task, pod or instance it runs on.
# A gs:// location carries no secret: the server follows the ambient Google
# credentials chain, so mount a service-account key file and keep
# GOOGLE_APPLICATION_CREDENTIALS pointing at it on every run, or set nothing
# on Google compute and let workload identity answer.
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
      # AWS is asked which region the bucket is in when the environment names
      # none. An S3-compatible service may not answer that, so one gets the
      # conventional default instead.
      region="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
      if [ -z "$region" ] && [ -n "${AWS_ENDPOINT_URL:-}" ]; then
        region=us-east-1
      fi
      echo "secrets:"
      echo "  default:"
      echo "    type: s3"
      echo "    scope: \"$location\""
      if [ -n "$region" ]; then
        echo "    region: \"$region\""
      fi
      if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_SESSION_TOKEN:-}" ]; then
        : "${AWS_SECRET_ACCESS_KEY:?must be set together with AWS_ACCESS_KEY_ID}"
        echo "    access_key_id: \"$AWS_ACCESS_KEY_ID\""
        echo "    secret_access_key: \"$AWS_SECRET_ACCESS_KEY\""
      fi
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
  # The file may carry S3 keys, so only the server process should read it.
  chmod 600 "$METASTORE_FILE"
}

if [ "$1" = "pivot" ] && [ "${2:-}" = "server" ]; then
  if [ ! -f "$METASTORE_FILE" ]; then
    generate_metastore_file
  fi
fi

exec "$@"
