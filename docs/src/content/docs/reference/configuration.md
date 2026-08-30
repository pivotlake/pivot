---
title: Configuration
description: YAML configuration for the pivotdb server and its datastores.
sidebar:
  order: 6
---

The server reads one YAML file. Unknown fields are rejected, including unknown
fields nested inside a section.

Sizes accept whole bytes or base-1024 `k`, `m`, `g`, and `t` suffixes, such as
`512m` or `32g`. Durations require a `ms`, `s`, `m`, or `h` suffix, such as
`500ms` or `30s`.

### `server`

Every server setting is optional.

| Key | Default | Description |
| --- | --- | --- |
| `server.bind` | `127.0.0.1:5432` | Address for the Postgres wire endpoint. |
| `server.http_bind` | Disabled | Address for the bundled web dashboard and HTTP API. |
| `server.memory` | 80% of total memory | Buffer-pool budget. Overrides `PIVOT_MEMORY_PCT`. |
| `server.workers` | Machine core count | Dispatch worker threads. |
| `server.refresh_interval` | `30s` | How often in-memory catalogs refresh commits made by other processes. |
| `server.disk_cache.dir` | Required when enabled | Persistent local directory for cached remote reads. |
| `server.disk_cache.size` | `64g` | Cached-byte budget. |
| `server.disk_cache.max_objects` | `65536` | Maximum cached objects and open cache file descriptors. |
| `server.tls.cert` | Required when enabled | PEM certificate followed by any intermediate certificates. |
| `server.tls.key` | Required when enabled | PEM private key in PKCS#8, PKCS#1, or SEC1 form. |

Adding `server.tls` makes TLS available but does not require clients to use it.
Both `cert` and `key` are required.

### `metastore.datastores`

Exactly one datastore must set `default: true`.

| Key | Default | Description |
| --- | --- | --- |
| `kind` | Required | Datastore format. `delta` is the only supported value. |
| `location` | Required | Local path, `s3://` URI, or `gs://` URI. |
| `default` | `false` | Makes this the target of unqualified SQL names. Exactly one must be true. |
| `compact` | `true` | Runs background compaction. Enable it in only one process per shared datastore. |
| `compact_bytes` | `64m` | Layout-compaction output target; files strictly below half this size are small-file candidates, and an individual row group may exceed it. |
| `compact_merge_bytes` | 1.3 times `compact_bytes` | Accumulated small-file bytes that immediately trigger a merge. |
| `compact_min_files` | `100` | File count at which the small-file balance fallback may merge. |
| `compact_parallelism` | `3` | Maximum number of disjoint compaction merges rewritten concurrently. Values below one run one merge at a time. Each merge in flight holds its decoded input rows in memory. |
| `vacuum` | `true` | Deletes expired unreferenced files and old log entries. Enable it in only one process per shared datastore. |

### `metastore.secrets`

Secret names are user-defined. `scope` is an optional URI prefix; the most
specific secret covering a datastore location is selected.

| Secret type | Keys |
| --- | --- |
| `s3` | `type: s3`, optional `scope`, required `region`, `access_key_id`, and `secret_access_key`, plus optional `endpoint` for S3-compatible storage. |
| `gcs` | `type: gcs`, optional `scope`, and required `credentials_file`. Without a matching secret, GCS uses ambient Application Default Credentials. |

Two secrets cannot claim the same scope.

### `metastore.users`

| Authentication method | Configuration |
| --- | --- |
| Trust | `auth: { method: trust }`. No password is checked. |
| SCRAM-SHA-256 | `auth: { method: scram-sha-256, verifier: "pivot-scram-sha-256$..." }`. Store the precomputed verifier, not the password. |

The built-in `pivot` user uses trust authentication unless it is configured
explicitly.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `PIVOT_MEMORY_PCT` | `80` | Percentage of total memory used when `server.memory` is omitted. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Unset | GCS credentials file used by the ambient credentials chain when no matching GCS secret exists. |

See `server/config.example.yaml` in the repository for a complete annotated
configuration.
