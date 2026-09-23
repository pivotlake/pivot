# pivot

`pivot open` is an interactive, local Pivot SQL shell. It embeds the planner,
catalog, Pivot datastore, and dispatch workers in one process. It does not
connect to a Pivot server and does not require `psql`.

## Running

```sh
cargo run --release -p bin --bin pivot -- open ./pivot-data
```

The installed binary is named `pivot`:

```sh
cargo install --path bin
pivot open ./pivot-data
```

The CLI uses every available dispatch worker and assigns 80% of physical
memory, minus a 4 GiB reserve for allocations outside the pool, to the dispatch
buffer pool. Override either resource independently:

```sh
pivot open ./pivot-data --memory 8g --workers 4
```

`--memory` accepts base-1024 `k`, `m`, `g`, and `t` suffixes, or a share of
physical memory such as `50%`. Pivot checks the budget against memory currently
available before faulting in the pool, and exits with an error instead of
risking an OOM kill when it does not fit.

## Datastore directory

Every `pivot open` invocation requires the path of one local Pivot datastore.
The directory is created when it does not exist. Tables, schemas, and inserted
data remain in that directory after `\q`, Ctrl+D, and subsequent invocations.

Only one Pivot process may open a local datastore directory at a time. Pivot
holds an exclusive lock on `.pivot.lock` for as long as the datastore is open
and reports the lock owner's PID when another process tries to use it.

## Interaction

SQL may span several lines and executes at a semicolon. Several pasted
statements execute in order and stop at the first error.

Supported shell commands:

```text
\q                 quit
quit;              quit
\timing            toggle statement timing
\timing on|off     set statement timing
\h, \help           show this help
```

Ctrl+C cancels a running statement. At the prompt it clears the current input
without ending the session. Command history exists only in memory and is not
written to disk.

Results use psql's aligned layout and are written directly to the terminal.

Run `VACUUM;` to wait for one cleanup sweep of the current datastore. This works
in `pivot open` and on servers with `vacuum: false`. It removes unreferenced data
files, superseded log files, and dropped-table storage only after their retention
windows have elapsed. Background vacuum remains configured separately.

## Server

The same executable runs a configured server in the foreground: a
PostgreSQL-wire-compatible endpoint any Postgres client (`psql`,
`tokio-postgres`, JDBC, ...) can connect to and query a set of named
datastores through. `src/execution/` owns statement transactions, planning,
the plan cache, dispatch, cancellation, and command classification; the
server adds PostgreSQL wire and HTTP adapters around that transport-neutral
core, and the shell converts result batches directly to terminal cells on
the dispatch workers. See `src/server/mod.rs` for the library API.

The Debian package installs `pivot server` as a systemd service under a
dedicated non-login account. See
[`../packaging/debian/README.md`](../packaging/debian/README.md) for the
package layout and lifecycle contract.

### Running

```sh
cargo run --release -p bin --bin pivot -- server --config <FILE>
```

Or after `cargo install --path bin` or installing the Debian package:

```sh
pivot server --config <FILE>
```

One YAML file configures the whole instance. See
[`config.example.yaml`](config.example.yaml) for a commented file to copy.

#### The config file

The top level is the instance: the resources it runs on. Every setting there
has a default. `server` is the endpoint, which only
serving has; it too may be left out entirely. `datastores`, `secrets` and
`users` are what the instance serves; they are the operator's, and the server
never rewrites the file. `metastore` names the file the server does write to,
where `CREATE USER` lands.

```yaml
memory: 32g                 # default 80%: that share of total RAM minus 4 GiB
workers: 16                 # default: number of cores
datastore_refresh_interval: 30s  # default 30s
log: info                   # default; or debug, or per-target directives
disk_cache:                 # cache S3 reads on local disk; omitted means no cache
  dir: /var/cache/pivot     # required once the section is present
  size: 64g                 # default 64g
  max_objects: 65536        # default; one open file descriptor per cached object

server:
  bind: 0.0.0.0:5432        # default 127.0.0.1:5432
  tls:                      # offer SSL to clients that ask; omitted means no SSL
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key

datastores: ...
secrets: ...
users: ...

metastore:
  kind: file
  path: /var/lib/pivot/metastore.yaml
```

#### The metastore file

The file `metastore` names holds the same `datastores`, `secrets` and `users`
maps as the config, without a `server` or `metastore` section. The server owns
it: `CREATE USER` rewrites it, so the file must be writable by the server and
readable by nobody else, since a secret may land in it. It must exist when the
server starts; a path that is not there is an error, not an empty store. A name
defined in both files stops startup rather than one definition silently
winning, and `CREATE USER` refuses a name the config defines, since the config
is never rewritten.

Leave the `metastore` section out to run on the config's entries alone. The
server then has nowhere to write, and `CREATE USER` says so.

#### Datastores

The disk provider lives in the separate `metastore-disk` crate, which owns the
`datastores`, `secrets` and `users` maps. At least one datastore is always
required, including when serving one local directory. Exactly one datastore must set `default = true`; it
becomes the current database (the target of unqualified table names). Every
datastore is attached as a database of its own name, so a query reads any other
one by qualifying it: `SELECT * FROM warm.main.tbl`. `kind` is the datastore
implementation (`pivot` today); the storage backend is inferred from `location` (a plain
path is local, an `s3://` URI is S3, a `gs://` URI is Google Cloud Storage).
Compaction is configured per datastore with `compact`. `compact_bytes` targets
the compaction output size; files strictly below half that size are small-file
candidates, while an individual row group may exceed the target.
`compact_merge_bytes` sets the accumulated small-file bytes that immediately
trigger a merge (by default 1.3 times `compact_bytes`), and
`compact_min_files` sets when the balance fallback is allowed (100 by default).
`compact_parallelism` limits concurrent, disjoint merge rewrites (1 by default);
each merge in flight holds its decoded input rows in memory. Compaction is on by
default and should run in only one process per datastore (set `compact: false`
on the others):

```yaml
datastores:
  hot:
    kind: pivot
    location: /var/lib/pivot/datastores/hot   # local path -> local store
    default: true                             # the current database
  warm:
    kind: pivot
    location: s3://analytics/warm/            # S3 store
    compact: true                             # this datastore compacts itself
    compact_bytes: 128m
    compact_merge_bytes: 192m
    compact_min_files: 100
    compact_parallelism: 3
  cold:
    kind: pivot
    location: gs://analytics/cold/            # Google Cloud Storage store
```

Start the server with:

```sh
pivot server --config pivot.yaml
```

A local datastore directory may be open in only one Pivot process at a time.
Pivot holds `.pivot.lock` in that directory until shutdown and reports the
owning PID if another process tries to open it.

`datastore_refresh_interval` sets how often each datastore brings itself up
to date with the store: new Delta versions, new files' footers, and, for shared
remote stores, tables committed by other processes. It bounds how stale a query's view of externally committed data can
be; this process's own INSERT and compaction publish their commits immediately.

#### Secrets

A datastore carries no credentials of its own. Each entry under `secrets` holds
the credentials for the paths its `scope` covers, so a bucket's keys are written
once however many datastores sit in it. `type` names the backend, `s3` or `gcs`,
and carries exactly that backend's fields:

```yaml
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

An `s3://` datastore without a covering secret uses anonymous, unsigned
requests. This supports public S3 locations; define a secret when the location
requires authentication. A `gs://` datastore without one falls back to the
ambient Application Default Credentials chain instead:
`GOOGLE_APPLICATION_CREDENTIALS`, the file
`gcloud auth application-default login` writes, or the workload identity of the
Google compute instance.

An S3 secret's keys are written inline, so protect files containing one
appropriately.

#### Users

Each entry under `users` names a user that may connect. Its nested `auth`
contains exactly one authentication method and that method's fields:

```yaml
users:
  pivot:
    auth:
      method: password
      password: Password1337

  reader:
    auth:
      method: trust

  analytics:
    auth:
      method: scram-sha-256
      verifier: "pivot-scram-sha-256$4096:8fZ1u...$Wm9tYm..."
```

A `password` is written as is. The server derives a SCRAM-SHA-256 verifier
from it when it reads the file, so the password is never sent over the wire
and never stored: a metastore file the server rewrites holds the verifier
in its place. The `scram-sha-256` method takes that precomputed verifier
directly, in PivotDB's
`pivot-scram-sha-256$4096:<base64 salt>$<base64 salted password>` format;
it is how `CREATE USER` stores a user. Both methods authenticate a client the
same way. Trust performs no identity proof: anyone who supplies `reader` as
the user name is accepted. Variant-specific fields are enforced, so trust
cannot contain a verifier, SCRAM cannot omit one, and `password` takes the
password alone.

A user named `pivot` is always served, whatever else the `users` map
defines, so a server is always reachable. It is trusted unless `pivot` is
defined explicitly, which takes over its authentication method entirely: give it
a `password` or a `scram-sha-256` verifier to require a password of it.

Logging is controlled by the config's `log`. A level applies to everything;
`tracing` filter directives narrow it per target:

```yaml
log: debug
```

```yaml
log: info,dispatch=debug
```

The server writes its log to stdout, where journald and `docker logs` read it.

Every documented setting is in the file; see
[`config.example.yaml`](config.example.yaml). The switches Pivot's own
experiments and profiling runs flip stay environment variables the engine
reads itself, and are not part of the configuration.

### Connecting

The server speaks the PostgreSQL v3 wire protocol. Any Postgres client works, for example `psql`:

```sh
psql -h 127.0.0.1 -p 5432 -U analytics
```

Every statement commits individually. `BEGIN`, `COMMIT`, and `ROLLBACK` are
accepted with their usual tags so drivers that wrap statements in a
transaction by default can work, but they group nothing: statements between
`BEGIN` and `ROLLBACK` are already committed and stay.

SCRAM users prove their password without sending it over the wire. Trusted users
connect without a password. The built-in configuration therefore connects as:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

#### SSL

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
