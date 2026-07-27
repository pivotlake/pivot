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
cargo run --release -- [OPTIONS]
```

Or after `cargo install --path .` / building, the binary is named
`pivotdb-server`:

```sh
pivotdb-server --metastore <FILE> [OPTIONS]
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--bind` | `127.0.0.1:5432` | TCP socket the server binds to. |
| `--workers` | number of cores | Number of dispatch worker threads. |
| `--metastore` | required | TOML file defining the named datastores. |
| `--memory` | 80% of RAM | Buffer-pool budget such as `32g` or `512m`. |

### Metastore configuration

The TOML provider lives in the separate `metastore-toml` crate. A metastore is
always required, including when serving one local datastore. A datastore named
`default` is required because DuckDB uses it for unqualified table names. Each
other datastore is attached as a database of the same name:

```toml
[datastore.default]
kind = "local"
location = "/var/lib/pivot/hot"

[datastore.warm]
kind = "s3"
location = "s3://analytics/warm/"
region = "us-east-1"
access_key_id = "AKIA..."
secret_access_key = "..."
# session_token = "..."
# endpoint = "http://localhost:9000"
```

Start the server with:

```sh
pivotdb-server --metastore config.toml
```

For S3 credentials supplied through `AWS_*` environment variables, use
`source = "env"` instead of inline keys. Protect files containing inline
credentials appropriately.

Logging is controlled by `RUST_LOG` (defaults to `info`):

```sh
RUST_LOG=server=debug,dispatch=info pivotdb-server --metastore config.toml --bind 0.0.0.0:5432
```

## Connecting

The server speaks the PostgreSQL v3 wire protocol with no auth. Any Postgres client works — for example `psql`:

```sh
psql -h 127.0.0.1 -p 5432 -U anything
```

Tables are registered in a datastore with `CREATE TABLE`. A table path is
relative to that datastore's configured location:

```sql
CREATE TABLE hits (url VARCHAR, ts BIGINT)
  WITH (path = 'hits');

CREATE TABLE warm.main.events (id BIGINT, ts BIGINT)
  WITH (path = 'events');

SELECT COUNT(*) FROM hits WHERE url <> '';
SELECT * FROM warm.main.events;
```

The catalog is process-global, so a table created on one connection is visible
to every other connection. A query may join tables from several datastores;
each datastore contributes an independent snapshot to the query transaction.
