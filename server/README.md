# server

A PostgreSQL-wire-compatible server for pivotdb. Any Postgres client (`psql`, `tokio-postgres`, JDBC, …) can connect and run SQL against parquet directories.

Glue layer: [`pgwire`](https://crates.io/crates/pgwire) drives the wire protocol, [`planner`](../planner) turns each SQL string into a Pivot plan (via an embedded DuckDB) against a [`ParquetCatalog`](../catalog), and [`dispatch`](../dispatch) runs the resulting dataflow on its thread-per-core worker pool. Each query hops to `tokio::task::spawn_blocking` to drive the (non-`Send`) DuckDB planner — the planner is cached in a thread-local on each blocking-pool thread and reused across queries; query execution itself happens on the dispatch workers.

See `src/lib.rs` for the library API, or run the binary directly.

## Running

```sh
cargo run --release -- [OPTIONS]
```

Or after `cargo install --path .` / building, the binary is named `pivot`:

```sh
pivot [OPTIONS]
```

### Options

| Flag         | Default            | Description                                                              |
| ------------ | ------------------ | ------------------------------------------------------------------------ |
| `--bind`     | `127.0.0.1:5432`   | TCP socket the server binds to.                                          |
| `--workers`  | number of cores    | Number of dispatch worker threads. One thread per core is recommended.   |

Logging is controlled by `RUST_LOG` (defaults to `info`):

```sh
RUST_LOG=server=debug,dispatch=info pivot --bind 0.0.0.0:5432
```

## Connecting

The server speaks the PostgreSQL v3 wire protocol with no auth. Any Postgres client works — for example `psql`:

```sh
psql -h 127.0.0.1 -p 5432 -U anything
```

Tables are not pre-registered. Each session creates them on demand:

```sql
CREATE TABLE hits (url VARCHAR, ts BIGINT)
  WITH (path = '/path/to/parquet/dir');

SELECT COUNT(*) FROM hits WHERE url <> '';
```

The catalog is process-global, so a table created on one connection is visible to every other connection.
