# server

A PostgreSQL-wire-compatible server for pivotdb. Any Postgres client (`psql`,
`tokio-postgres`, JDBC, …) can connect and query a set of named datastores.

[`engine`](../engine) owns statement transactions, planning, the plan cache,
dispatch, cancellation, and command classification. The server adds PostgreSQL
wire and HTTP adapters around that transport-neutral core. The local CLI uses
the same engine and converts result batches directly to terminal cells on the
dispatch workers.

See `src/lib.rs` for the library API, or run the public CLI.

## Running

```sh
cargo run --release -p cli --bin pivot -- server --config <FILE>
```

Or after `cargo install --path cli` or installing the Debian package:

```sh
pivot server --config <FILE>
```

One YAML file configures the whole instance. See
[`config.example.yaml`](config.example.yaml) for a commented file to copy. The
optional `--metastore-file` flag merges a second datastore and user file into
that configuration.

### The config file

The file has two sections. `server` is the instance: where it listens and what
it may use. Every setting there has a default, so the section may be left out
entirely. `metastore` is the data to serve, and is required.

```yaml
server:
  bind: 0.0.0.0:5432        # default 127.0.0.1:5432
  memory: 32g               # default: 80% of total RAM (see PIVOT_MEMORY_PCT)
  workers: 16               # default: number of cores
  refresh_interval: 30s     # default 30s
  http_bind: 127.0.0.1:8081 # serve the web dashboard; omitted means no dashboard
  disk_cache:               # cache S3 reads on local disk; omitted means no cache
    dir: /var/cache/pivot   # required once the section is present
    size: 64g               # default 64g
    max_objects: 65536      # default; one open file descriptor per cached object
  tls:                      # offer SSL to clients that ask; omitted means no SSL
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key

metastore:
  datastores: ...
  secrets: ...
  users: ...
```

### Datastores

The disk provider lives in the separate `metastore-disk` crate, which owns the
`metastore` section. At least one datastore is always required, including when
serving one local directory. Exactly one datastore must set `default = true`; it
becomes the current database (the target of unqualified table names). Every
datastore is attached as a database of its own name, so a query reads any other
one by qualifying it: `SELECT * FROM warm.main.tbl`. `kind` is the datastore
format (`delta` today); the storage backend is inferred from `location` (a plain
path is local, an `s3://` URI is S3, a `gs://` URI is Google Cloud Storage).
Compaction is configured per datastore with `compact`. `compact_bytes` sets the
small-file boundary, `compact_merge_bytes` sets the accumulated bytes that
immediately trigger a merge (by default 1.3 times that boundary), and
`compact_min_files` sets when the balance fallback is allowed (100 by default);
compaction is on by default and should run in only one process per datastore
(set `compact: false` on the others):

```yaml
metastore:
  datastores:
    hot:
      kind: delta
      location: /var/lib/pivot/hot    # local path -> local store
      default: true                   # the current database
    warm:
      kind: delta
      location: s3://analytics/warm/  # S3 store
      compact: true                   # this datastore compacts itself
      compact_bytes: 128m
      compact_merge_bytes: 192m
      compact_min_files: 100
    cold:
      kind: delta
      location: gs://analytics/cold/  # Google Cloud Storage store
```

Start the server with:

```sh
pivot server --config pivot.yaml
```

A local datastore directory may be open in only one Pivot process at a time.
Pivot holds `.pivot.lock` in that directory until shutdown and reports the
owning PID if another process tries to open it.

The `server` section's `refresh_interval` sets how often the background refresh
brings the in-memory table set up to date with the store: new Delta versions,
new files' footers, and, for shared remote stores, tables committed by other
processes. It bounds how stale a query's view of externally committed data can
be; this process's own INSERT and compaction publish their commits immediately.

### Secrets

A datastore carries no credentials of its own. Each entry under `secrets` holds
the credentials for the paths its `scope` covers, so a bucket's keys are written
once however many datastores sit in it. `type` names the backend, `s3` or `gcs`,
and carries exactly that backend's fields:

```yaml
metastore:
  secrets:
    analytics:
      type: s3
      scope: s3://analytics/          # this bucket, whatever the prefix
      region: us-east-1
      access_key_id: AKIA...
      secret_access_key: "..."
      # endpoint: http://localhost:9000   # MinIO / S3-compatible
    analytics-archive:
      type: s3
      scope: s3://analytics/archive/  # more specific: wins under archive/
      region: us-east-1
      access_key_id: AKIA...
      secret_access_key: "..."
    google:
      type: gcs                       # no scope: every gs:// location
      credentials_file: /etc/pivot/gcs-key.json
```

A scope is matched whole path segments at a time, so `s3://analytics/warm`
covers `s3://analytics/warm/2024` and not `s3://analytics/warmer`. The most
specific scope covering a location authenticates it, and no two secrets may
claim the same scope, so which secret that is never depends on the order they
were written in. A secret with no `scope` covers every location of its type;
there can be only one such secret per type.

An `s3://` datastore needs a secret covering it: without one there is nothing to
sign its requests with, and startup stops. A `gs://` datastore without one falls
back to the ambient Application Default Credentials chain instead:
`GOOGLE_APPLICATION_CREDENTIALS`, the file
`gcloud auth application-default login` writes, or the workload identity of the
Google compute instance.

An S3 secret's keys are written inline, so protect files containing one
appropriately.

### Users

Each entry under `users` names a user that may connect. Its nested `auth`
contains exactly one authentication method and that method's fields:

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

The SCRAM verifier is precomputed in PivotDB's
`pivot-scram-sha-256$4096:<base64 salt>$<base64 salted password>` format. Trust
performs no identity proof: anyone who supplies `pivot` as the user name is
accepted. Variant-specific fields are enforced, so trust cannot contain a
verifier and SCRAM cannot omit one.

A user named `pivot` is always served, whatever else the `users` section
defines, so a server is always reachable. It is trusted unless `pivot` is
defined explicitly, which takes over its authentication method entirely: give it
a `scram-sha-256` verifier to require a password of it.

Logging is controlled by `RUST_LOG` (defaults to `info`):

```sh
RUST_LOG=server=debug,dispatch=info pivot server --config pivot.yaml
```

## Connecting

The server speaks the PostgreSQL v3 wire protocol. Any Postgres client works, for example `psql`:

```sh
psql -h 127.0.0.1 -p 5432 -U analytics
```

SCRAM users prove their password without sending it over the wire. Trusted users
connect without a password. The built-in configuration therefore connects as:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

### SSL

Give the `server` section a `tls` block and the endpoint offers encryption to
clients that ask for it:

```yaml
server:
  tls:
    cert: /etc/pivot/server.crt   # the server's certificate, then any intermediates
    key: /etc/pivot/server.key    # its private key, in PKCS#8, PKCS#1 or SEC1
```

Both files are PEM and both are read at startup, so a certificate that is
missing, malformed, or paired with the wrong key stops the server rather than
surfacing on the first client that asks to encrypt. Keep the key readable only
by the user the server runs as.

Clients then connect as they would to PostgreSQL:

```sh
psql "host=pivot.example.com port=5432 user=pivot sslmode=verify-full sslrootcert=/etc/ssl/ca.crt"
```

Turning SSL on makes it available, not compulsory, the same way PostgreSQL's own
`ssl = on` does: a client asking for plaintext still gets a plaintext session on
the same port. There is no server-side setting yet that refuses those, so a
deployment that must have every session encrypted should keep the port off any
untrusted network.

Encrypted sessions authenticate exactly as plaintext ones do, with trust or
SCRAM-SHA-256. Channel binding (`SCRAM-SHA-256-PLUS`) is not offered, so a
client passing `channel_binding=require` is turned away; `prefer`, the libpq
default, negotiates plain SCRAM-SHA-256 and connects.

Tables are registered in a datastore with `CREATE TABLE`. A table path is
relative to that datastore's configured location:

```sql
CREATE TABLE hits (url VARCHAR, ts BIGINT)
  WITH (with_pre_existing_parquets = 'hits');

CREATE TABLE warm.main.events (id BIGINT, ts BIGINT)
  WITH (with_pre_existing_parquets = 'events');

SELECT COUNT(*) FROM hits WHERE url <> '';
SELECT * FROM warm.main.events;
```

`with_pre_existing_parquets` takes Parquet files that already sit at that path as the
table's initial data, instead of starting the table empty.

Adopted files are recorded by their absolute path and are never written to,
moved, or deleted: they stay the directory owner's.

The catalog is process-global, so a table created on one connection is visible
to every other connection. A query may join tables from several datastores;
each datastore contributes an independent snapshot to the query transaction.
