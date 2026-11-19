#!/bin/bash
# usage: create_table.sh <table name> [extra WITH options, e.g. "with_pre_existing_parquets = '/lake/adopt'"]
set -euo pipefail
NAME=$1
EXTRA=${2:-}
OPTIONS="sort_by = 'round_timestamp,env,cluster,bucket_workload,workload,level,instance,container_name'"
if [ -n "$EXTRA" ]; then OPTIONS="$OPTIONS, $EXTRA"; fi
psql -h 127.0.0.1 -p 5432 -U pivot -d lake -v ON_ERROR_STOP=1 <<EOF
CREATE TABLE $NAME (
  guid UBIGINT,
  index_tier VARCHAR,
  round_timestamp TIMESTAMP,
  "timestamp" TIMESTAMP,
  cluster VARCHAR,
  source VARCHAR,
  env_type VARCHAR,
  pod_name VARCHAR,
  pod_uid VARCHAR,
  workload VARCHAR,
  bucket_workload UTINYINT,
  namespace VARCHAR,
  level VARCHAR,
  format VARCHAR,
  container_name VARCHAR,
  env VARCHAR,
  instance VARCHAR,
  is_pii BOOLEAN,
  series_id UBIGINT,
  pattern_id UBIGINT,
  pattern VARCHAR,
  trace_id VARCHAR,
  span_id VARCHAR,
  body VARCHAR,
  flatten_string_keys VARIANT,
  flatten_string_values VARIANT,
  flatten_float_keys VARIANT,
  flatten_float_values VARIANT,
  flatten_bool_keys VARIANT,
  flatten_bool_values VARIANT,
  string_attributes VARIANT,
  float_attributes VARIANT,
  string_tags VARIANT,
  float_tags VARIANT,
  node_name VARCHAR,
  host VARCHAR,
  service VARCHAR,
  instance_id VARCHAR,
  function_name VARCHAR,
  lambda_arn VARCHAR,
  content VARCHAR,
  body_length UBIGINT,
  metadata VARIANT,
  favorite_attributes VARIANT,
  favorite_attributes_json VARIANT,
  session_id VARCHAR,
  instance_uid UBIGINT,
  sensitive_attributes VARIANT,
  sensitive_keys VARIANT
) WITH ($OPTIONS);
EOF
