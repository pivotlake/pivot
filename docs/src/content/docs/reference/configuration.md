---
title: Configuration
description: Every key of the config file, plus the environment it reads.
sidebar:
  order: 6
---

One YAML file configures a whole instance. It has two sections: `server`, the
instance itself, and `metastore`, the data it serves. Every `server` key has a
default, so that section may be left out; `metastore` is required, since a
server with no datastores has nothing to answer a query with.

Unknown keys are rejected rather than ignored, in both sections. A misspelled
setting would otherwise leave the server running on a default nobody asked for.

```sh
pivot server --config pivot.yaml
```

## Units

| Written | Means |
| --- | --- |
| `32g`, `512m`, `64k`, `1t` | Byte counts, base 1024, case-insensitive, each suffix also accepting a trailing `b` |
| `500ms`, `30s`, `5m`, `1h` | Time spans. The unit is mandatory, and zero is not a span |

## server

```yaml
server:
  bind: 0.0.0.0:5432
  http_bind: 127.0.0.1:8081
  memory: 32g
  workers: 16
  refresh_interval: 30s
  disk_cache:
    dir: /var/cache/pivot
    size: 64g
    max_objects: 65536
  tls:
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key
```

| Key | Default | Meaning |
| --- | --- | --- |
| `bind` | `127.0.0.1:5432` | Address the PostgreSQL endpoint listens on. The loopback default keeps an unconfigured server off the network |
| `http_bind` | unset | Also serve the bundled web dashboard here. Unset means no dashboard |
| `memory` | 80% of total RAM | Budget for the buffer pool |
| `workers` | the machine's core count | Dispatch worker threads |
| `refresh_interval` | `30s` | How often each datastore brings its in-memory table set up to date with storage |
| `disk_cache` | unset | Cache remote reads on local disk. Unset means no cache |
| `tls` | unset | Offer SSL to clients that ask for it. Unset means the endpoint is plaintext-only |

`refresh_interval` bounds how stale a query's view of *externally* committed
data can be. This process's own inserts and compactions are visible
immediately.

### server.disk_cache

Caches byte ranges read from an object store, never local files, which are read
from the filesystem directly. The contents survive a restart.

| Key | Default | Meaning |
| --- | --- | --- |
| `dir` | required | Directory the cached ranges live in |
| `size` | `64g` | Byte budget for the cached data |
| `max_objects` | `65536` | Maximum cached objects. The cache holds one open file descriptor per object, so keep this under the process limit |

### server.tls

Both keys are required once the section is written: an endpoint that offers
encryption has to present a certificate, and a certificate is worth nothing
without the key proving the server holds it. Both files are PEM and both are
read at startup, so a missing or mismatched pair stops the server rather than
surfacing on the first client that asks to encrypt.

| Key | Meaning |
| --- | --- |
| `cert` | The certificate to present: the server's own first, then any intermediates |
| `key` | Its private key, in PKCS#8, PKCS#1 or SEC1 |

Writing the section turns SSL on, it does not make it compulsory: a client
asking for plaintext still gets a plaintext session, which is how PostgreSQL's
own `ssl = on` behaves. Channel binding (`SCRAM-SHA-256-PLUS`) is not offered,
so a client passing `channel_binding=require` is turned away, while `prefer`,
libpq's default, negotiates plain SCRAM-SHA-256.

## metastore

The `metastore` section holds three maps: `datastores`, `secrets` and `users`.
The `--metastore-file` flag names a second YAML file with the same three maps
at its top level, whose entries are merged in. A name defined in both files is
an error rather than an override.

### datastores

```yaml
metastore:
  datastores:
    hot:
      kind: delta
      location: /var/lib/pivot/hot
      default: true
    warm:
      kind: delta
      location: s3://analytics/warm/
      compact: true
      compact_bytes: 128m
      compact_merge_bytes: 192m
      compact_min_files: 100
      vacuum: true
    cold:
      kind: delta
      location: gs://analytics/cold/
```

| Key | Default | Meaning |
| --- | --- | --- |
| `kind` | required | The datastore format. `delta` today |
| `location` | required | Where it lives. A plain path is local, `s3://` (also `s3a://`) is S3, `gs://` is Google Cloud Storage, `file://` is local |
| `default` | `false` | Marks the current database: the target of unqualified table names and DDL. Exactly one datastore must set it |
| `compact` | `true` | Run this datastore's background compaction in this process |
| `compact_bytes` | `64m` | Per-table boundary between small files and layout candidates |
| `compact_merge_bytes` | 1.3 times `compact_bytes` | Accumulated small-file bytes that trigger a merge immediately |
| `compact_min_files` | `100` | File count at which sub-target small files may use the balance fallback |
| `vacuum` | `true` | Run this datastore's background vacuum, which deletes unreferenced data files and superseded commit logs past their retention |

Each datastore is attached as a database of its own name, so a query reads any
of them by qualifying the table: `SELECT * FROM warm.main.events`.

Compaction rewrites files, so it must run in only one process per datastore.
Set `compact: false` on the others. The same holds for `vacuum`, which is also
what to turn off on a read-only server.

A local datastore directory may be open in only one pivotdb process at a time.
The process holds `.pivot.lock` in that directory until shutdown and names the
owning PID if another one tries to open it.

### secrets

A datastore carries no credentials of its own. A secret holds the credentials
for the paths its `scope` covers, so a bucket's keys are written once however
many datastores sit in it.

```yaml
metastore:
  secrets:
    analytics:
      type: s3
      scope: s3://analytics/
      region: us-east-1
      access_key_id: AKIA...
      secret_access_key: "..."
      endpoint: http://localhost:9000
    google:
      type: gcs
      credentials_file: /etc/pivot/gcs-key.json
```

| Key | Applies to | Meaning |
| --- | --- | --- |
| `type` | both | `s3` or `gcs`, carrying exactly that backend's fields |
| `scope` | both | The URI prefix this secret covers. Omitted, it covers every location of its type, and there can be only one such secret per type |
| `region` | `s3` | AWS region |
| `access_key_id`, `secret_access_key` | `s3` | The keys, written inline, so protect the file accordingly |
| `endpoint` | `s3` | A path-style S3-compatible endpoint such as MinIO. Omitted, requests go AWS virtual-hosted style |
| `credentials_file` | `gcs` | A Google service-account or authorized-user JSON key file |

A scope is matched whole path segments at a time, so `s3://analytics/warm`
covers `s3://analytics/warm/2024` and not `s3://analytics/warmer`. The most
specific scope covering a location authenticates it, and no two secrets may
claim the same scope, so which one that is never depends on the order they were
written in.

An `s3://` datastore needs a secret covering it, and startup stops without one.
A `gs://` datastore without one falls back to the ambient Application Default
Credentials chain.

### users

```yaml
metastore:
  users:
    pivot:
      auth:
        method: trust
    analytics:
      auth:
        method: scram-sha-256
        verifier: "pivot-scram-sha-256$4096:8fZ1u...$Wm9tYm..."
```

| Method | Fields | Meaning |
| --- | --- | --- |
| `trust` | none | No identity proof: anyone supplying the name connects |
| `scram-sha-256` | `verifier` | SCRAM-SHA-256, in the format `pivot-scram-sha-256$4096:<base64 salt>$<base64 salted password>` |

Fields are enforced per method, so `trust` cannot carry a verifier and SCRAM
cannot omit one.

A user named `pivot` is always served, whatever the section says, so a server
is always reachable. It is trusted unless `pivot` is defined explicitly, which
takes its authentication over entirely.

[`CREATE USER`](/docs/reference/sql-statements/#create-user) adds users at
runtime, writing them to the `--metastore-file`.

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `RUST_LOG` | the server | Log filter, defaulting to `info`. `RUST_LOG=server=debug,dispatch=info` |
| `PIVOT_MEMORY_PCT` | the server | Percentage of total RAM the buffer pool takes when `server.memory` is unset. Defaults to 80 |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | `pivot open` | S3 credentials, when a datastore is opened straight from a URI rather than through a configured secret |
| `AWS_REGION`, `AWS_DEFAULT_REGION` | `pivot open` | S3 region, `us-east-1` when neither is set |
| `AWS_ENDPOINT_URL` | `pivot open` | A path-style S3-compatible endpoint |
| `GOOGLE_APPLICATION_CREDENTIALS` | both | Google service-account key file, the first step of the Application Default Credentials chain |

## Session variables

Set on a connection with [`SET`](/docs/reference/sql-statements/#set-and-reset).

| Variable | Effect |
| --- | --- |
| `pivot_stats` | Attach a timing and IO notice to every following statement |
