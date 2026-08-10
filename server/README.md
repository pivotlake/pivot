# server

A PostgreSQL-wire-compatible server for pivotdb. Any Postgres client (`psql`,
`tokio-postgres`, JDBC, …) can connect and query a set of named datastores.

Glue layer: [`pgwire`](https://crates.io/crates/pgwire) drives the wire protocol,
[`planner`](../planner) turns each SQL string into a Pivot plan (via an embedded
DuckDB) against the composite [`PivotCatalog`](../catalog), and
[`dispatch`](../dispatch) runs the resulting dataflow on its thread-per-core
worker pool. Each query hops to `tokio::task::spawn_blocking` to drive the
(non-`Send`) DuckDB planner; query execution itself happens on the dispatch
workers.

See `src/lib.rs` for the library API, or run the binary directly.

## Running

```sh
cargo run --release -- --config <FILE>
```

Or after `cargo install --path .` / building, the binary is named
`pivotdb-server`:

```sh
pivotdb-server --config <FILE>
```

`--config` is the only flag: one YAML file configures the whole instance. See
[`config.example.yaml`](config.example.yaml) for a commented file to copy.

### The config file

The file has two sections. `server` is the instance: where it listens and what
it may use. Every setting there has a default, so the section may be left out
entirely. `metastore` is the data to serve, and is required.

```yaml
server:
  bind: 0.0.0.0:5432        # default 127.0.0.1:5432
  memory: 32g               # default: 80% of total RAM (see PIVOT_MEMORY_PCT)
  workers: 16               # default: number of cores
  http_bind: 127.0.0.1:8081 # serve the web dashboard; omitted means no dashboard
  disk_cache:               # cache S3 reads on local disk; omitted means no cache
    dir: /var/cache/pivot   # required once the section is present
    size: 64g               # default 64g
    max_objects: 65536      # default; one open file descriptor per cached object

metastore:
  refresh_interval: 30s     # default 30s
  datastores: ...
  users: ...
```

### Datastores

The YAML provider lives in the separate `metastore-yaml` crate, which owns the
`metastore` section. At least one datastore is always required, including when
serving one local directory. Exactly one datastore must set `default = true`; it
becomes the current database (the target of unqualified table names). Every
datastore is attached as a database of its own name, so a query reads any other
one by qualifying it: `SELECT * FROM warm.main.tbl`. `kind` is the datastore
format (`delta` today); the storage backend is inferred from `location` (a plain
path is local, an `s3://` URI is S3, a `gs://` URI is Google Cloud Storage).
Compaction is configured per datastore
with `compact` (and the optional `compact_bytes` / `compact_min_files` tuning);
it is off by default and should run in only one process per datastore:

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
      region: us-east-1
      access_key_id: AKIA...
      secret_access_key: "..."
      # endpoint: http://localhost:9000
    cold:
      kind: delta
      location: gs://analytics/cold/  # Google Cloud Storage store
      # credentials_file: /etc/pivot/gcs-key.json
```

Start the server with:

```sh
pivotdb-server --config pivot.yaml
```

`refresh_interval` sets how often the background refresh brings the in-memory
table set up to date with the store: new Delta versions, new files' footers, and
tables committed by other processes. It bounds how stale a query's view of
externally committed data can be; this process's own INSERT and compaction
publish their commits immediately.

An S3 datastore's `region`, `access_key_id`, and `secret_access_key` are
required and given inline. Protect files containing inline credentials
appropriately.

A GCS datastore names a service-account (or authorized-user) JSON key file with
`credentials_file`. Omit it to resolve credentials from the ambient Application
Default Credentials chain instead: `GOOGLE_APPLICATION_CREDENTIALS`, the file
`gcloud auth application-default login` writes, or the workload identity of the
Google compute instance.

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

A config file with no users (an omitted or empty `users` section) receives one
built-in trusted user named `pivot`. Defining any users replaces that default
with the configured allowlist.

Logging is controlled by `RUST_LOG` (defaults to `info`):

```sh
RUST_LOG=server=debug,dispatch=info pivotdb-server --config pivot.yaml
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
