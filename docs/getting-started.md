# Getting started

This guide builds Pivot from source, starts a local datastore, and runs a query
through `psql`.

## Requirements

Pivot currently builds on Linux. You need:

- Stable Rust
- Git
- CMake and a C/C++ build toolchain
- Clang, libclang, and pkg-config
- A PostgreSQL client such as `psql`

On Debian or Ubuntu:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake clang libclang-dev pkg-config git postgresql-client
```

## Build the server

Clone the repository with its submodules and build the release binary:

```sh
git clone --recurse-submodules https://github.com/Epsio-Labs/pivotdb.git
cd pivotdb
RUSTC_WRAPPER= cargo build --release -p server --bin pivotdb-server
```

The empty `RUSTC_WRAPPER` overrides the optional `kache` compiler cache used by
Pivot developers. Omit it if `kache` is installed.

## Configure a local datastore

Create `pivot.yaml` in the repository root:

```yaml
metastore:
  datastores:
    local:
      kind: delta
      location: /tmp/pivot-data
      default: true
```

The server defaults to `127.0.0.1:5432`. With no `users` section, it also adds
a trusted user named `pivot` for local development.

## Start and connect

Start Pivot:

```sh
./target/release/pivotdb-server --config pivot.yaml
```

In another terminal:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

Create a table and add some rows:

```sql
CREATE TABLE events (
  id BIGINT,
  category VARCHAR
);

INSERT INTO events VALUES
  (1, 'build'),
  (2, 'query'),
  (3, 'query');

SELECT category, COUNT(*) AS rows
FROM events
GROUP BY category
ORDER BY category;
```

The table is stored under `/tmp/pivot-data/events` and is loaded again when the
server restarts.

## Optional dashboard

The web dashboard is embedded into the server binary. Build the frontend before
rebuilding the server:

```sh
npm --prefix web/frontend ci
npm --prefix web/frontend run build
RUSTC_WRAPPER= cargo build --release -p server --bin pivotdb-server
```

Then add an HTTP address to `pivot.yaml`:

```yaml
server:
  http_bind: 127.0.0.1:8081
```

Restart Pivot and open <http://127.0.0.1:8081>.

!!! warning

    The dashboard has no authentication layer and can execute SQL. Keep its
    HTTP endpoint on a trusted network.

Next, read [configuration](configuration.md) for storage and server settings or
the [SQL guide](sql.md) for the current query surface.
