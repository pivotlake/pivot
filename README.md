<p align="center">
  <img src="web/frontend/public/pivot.png" alt="Pivot" width="420">
</p>

# Pivot

Pivot is an experimental analytical database for querying and writing Delta
tables on local storage or S3. It speaks the PostgreSQL wire protocol, so it
works with `psql` and existing PostgreSQL clients.

> [!WARNING]
> Pivot is early-stage software. SQL coverage is incomplete, and configuration
> and storage compatibility may change. It is not ready for production use.

## Quick start

Pivot currently builds on Linux. You need stable Rust, Git, CMake, Clang,
libclang, pkg-config, a C/C++ build toolchain, and `psql`. On Debian or Ubuntu,
install the native dependencies with:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake clang libclang-dev pkg-config git postgresql-client
```

Clone the repository and build the server:

```sh
git clone --recurse-submodules https://github.com/Epsio-Labs/pivotdb.git
cd pivotdb
RUSTC_WRAPPER= cargo build --release -p server --bin pivotdb-server
```

`RUSTC_WRAPPER=` bypasses the optional `kache` compiler cache used by Pivot
developers. You can omit it if `kache` is installed.

Create a minimal `pivot.yaml` in the repository root:

```yaml
metastore:
  datastores:
    local:
      kind: delta
      location: /tmp/pivot-data
      default: true
```

Start Pivot:

```sh
./target/release/pivotdb-server --config pivot.yaml
```

In another terminal, connect with `psql`:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

The default local configuration trusts the `pivot` user and does not require a
password. Create a table, insert a few rows, and query them:

```sql
CREATE TABLE events (
  id BIGINT,
  category VARCHAR
);

INSERT INTO events VALUES
  (1, 'build'),
  (2, 'query'),
  (3, 'query');

SELECT category, COUNT(*)
FROM events
GROUP BY category
ORDER BY category;
```

Pivot stores this table under `/tmp/pivot-data/events`. Tables and rows remain
available after the server restarts.

## Working with existing data

`CREATE TABLE` can register Parquet files that already exist under a datastore.
The path is relative to that datastore's configured location:

```sql
CREATE TABLE page_views (
  user_id BIGINT,
  url VARCHAR,
  viewed_at TIMESTAMP
) WITH (path = 'page_views');

SELECT url, COUNT(*)
FROM page_views
GROUP BY url
ORDER BY COUNT(*) DESC
LIMIT 10;
```

The same model works with an S3-backed datastore. See the commented
[`server/config.example.yaml`](server/config.example.yaml) for local and S3
configuration, authentication, memory and worker limits, caching, and
compaction.

## What works today

- PostgreSQL v3 wire protocol
- Local filesystem and S3-backed Delta datastores
- `CREATE TABLE`, `INSERT`, and analytical `SELECT` queries
- Filters, projections, joins, grouping, aggregates, ordering, and limits
- Multiple datastores in one query
- An optional web dashboard and SQL console

SQL support is a growing subset of the DuckDB dialect. Unsupported statements,
types, and expressions return an error.

## More documentation

- [`server/README.md`](server/README.md): server configuration, users, and
  connecting clients
- [`web/README.md`](web/README.md): building and running the web dashboard
- [`dispatch/README.md`](dispatch/README.md): execution engine internals

To run the repository checks, use:

```sh
RUSTC_WRAPPER= ./ci.sh
```
