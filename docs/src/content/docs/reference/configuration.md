---
title: Configuration
description: YAML configuration for the pivotdb server and its datastores.
sidebar:
  order: 6
---

The server reads one YAML file, the one `--config` names. Unknown fields are
rejected, including unknown fields nested inside a section.

The file has a `server` section for the instance, top-level `datastores`,
`secrets` and `users` maps for what it serves, and a `metastore` section naming
the file the server writes to. The three maps are the operator's and the server
never rewrites them; `CREATE USER` lands in the metastore file.

Sizes accept whole bytes or base-1024 `k`, `m`, `g`, and `t` suffixes, such as
`512m` or `32g`. Durations require a `ms`, `s`, `m`, or `h` suffix, such as
`500ms` or `30s`.

### `server`

Every server setting is optional.

| Key | Default | Description |
| --- | --- | --- |
| `server.bind` | `127.0.0.1:5432` | Address for the Postgres wire endpoint. |
| `server.memory` | 80% of total memory minus 4 GiB | Buffer-pool budget. Overrides `PIVOT_MEMORY_PCT`. The 4 GiB is held back for allocations outside the pool. |
| `server.workers` | Machine core count | Dispatch worker threads. |
| `server.refresh_interval` | `30s` | How often in-memory catalogs refresh commits made by other processes. |
| `server.disk_cache.dir` | Required when enabled | Persistent local directory for cached remote reads. |
| `server.disk_cache.size` | `64g` | Cached-byte budget. |
| `server.disk_cache.max_objects` | `65536` | Maximum cached objects and open cache file descriptors. |
| `server.tls.cert` | Required when enabled | PEM certificate followed by any intermediate certificates. |
| `server.tls.key` | Required when enabled | PEM private key in PKCS#8, PKCS#1, or SEC1 form. |

Adding `server.tls` makes TLS available but does not require clients to use it.
Both `cert` and `key` are required.

### `datastores`

Exactly one datastore, in the config or the metastore file, must set
`default: true`.

| Key | Default | Description |
| --- | --- | --- |
| `kind` | Required | Datastore implementation. `pivot` is the only supported value. |
| `location` | Required | Local path, `s3://` URI, or `gs://` URI. |
| `default` | `false` | Makes this the target of unqualified SQL names. Exactly one must be true. |
| `compact` | `true` | Runs background compaction. Enable it in only one process per shared datastore. |
| `compact_bytes` | `64m` | Compaction output target; files strictly below half this size are small-file candidates, and an individual row group may exceed it. |
| `compact_merge_bytes` | 1.3 times `compact_bytes` | Accumulated small-file bytes that immediately trigger a merge. |
| `compact_min_files` | `100` | File count at which the small-file balance fallback may merge. |
| `compact_parallelism` | `1` | Maximum number of disjoint compaction merges rewritten concurrently. Values below one run one merge at a time. Each merge in flight holds its decoded input rows in memory. |
| `vacuum` | `true` | Deletes expired unreferenced files and old log entries. Enable it in only one process per shared datastore. |

### `secrets`

Secret names are user-defined. `scope` is an optional URI prefix; the most
specific secret covering a datastore location is selected.

| Secret type | Keys |
| --- | --- |
| `s3` | `type: s3`, optional `scope` and `region`, required `access_key_id` and `secret_access_key`, plus optional `endpoint` for S3-compatible storage. Without a matching secret, S3 uses anonymous, unsigned requests. A missing region is discovered with an unsigned `HeadBucket` request; specify it for compatible endpoints that do not return `x-amz-bucket-region`. |
| `gcs` | `type: gcs`, optional `scope`, and required `credentials_file`. Without a matching secret, GCS uses ambient Application Default Credentials. |

Two secrets cannot claim the same scope.

### `users`

| Authentication method | Configuration |
| --- | --- |
| Trust | `auth: { method: trust }`. No password is checked. |
| SCRAM-SHA-256 | `auth: { method: scram-sha-256, verifier: "pivot-scram-sha-256$..." }`. Store the precomputed verifier, not the password. |

The built-in `pivot` user uses trust authentication unless it is configured
explicitly.

### `metastore`

The file the server writes to. Optional: without it the server serves the
config's own entries and refuses `CREATE USER`.

| Key | Default | Description |
| --- | --- | --- |
| `metastore.kind` | Required | `file` is the only supported value. |
| `metastore.path` | Required | A YAML file of the same `datastores`, `secrets` and `users` maps as the config, without a `server` or `metastore` section. It must exist, and only the server should be able to read it. |

The config's entries and the file's are served together. A name defined in
both is a startup error, and `CREATE USER` refuses a name the config defines,
because the server never rewrites the config.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `PIVOT_MEMORY_PCT` | `80` | Percentage of total memory the buffer pool takes, minus 4 GiB, when `server.memory` or `pivot open --memory` is omitted. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Unset | GCS credentials file used by the ambient credentials chain when no matching GCS secret exists. |
| `PIVOT_ICEBERG_TOKEN` | Unset | Bearer token `pivot open --kind iceberg` sends on every request to the Iceberg REST catalog. |
| `PIVOT_ICEBERG_CREDENTIAL` | Unset | OAuth2 client credential (`client_id:client_secret`) `pivot open --kind iceberg` exchanges for a token at the catalog's token endpoint. Exclusive with `PIVOT_ICEBERG_TOKEN`. |
| `PIVOT_ICEBERG_OAUTH2_SERVER_URI` | Unset | Token endpoint the credential is exchanged at, when it is not the catalog's own. |
| `PIVOT_ICEBERG_SCOPE` | Unset | OAuth2 scope requested with the credential. |

See `server/config.example.yaml` in the repository for a complete annotated
configuration.
