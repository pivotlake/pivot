---
title: Quickstart
description: Open a datastore, create a table, run a query.
---

## Build

```sh
cargo build --release -p cli
```

That produces `pivot`, which both opens a datastore locally and runs the
server.

## Open a datastore

The shortest path to a query needs no configuration at all: point `pivot open`
at a directory, which is created if it does not exist, and it opens there.

```sh
./target/release/pivot open ./analytics
```

An object-store URI works the same way, with credentials taken from the
environment:

```sh
./target/release/pivot open s3://analytics/warm
```

## Create a table

```sql
CREATE TABLE events (id BIGINT, name VARCHAR, ts TIMESTAMP);
INSERT INTO events VALUES (1, 'signup', TIMESTAMP '2026-08-20 10:00:00');
```

A table can also adopt Parquet files that already sit in the datastore's
storage, instead of starting empty:

```sql
CREATE TABLE hits (url VARCHAR, ts TIMESTAMP)
  WITH (with_pre_existing_parquets = 'hits', sort_by = 'ts');
```

## Query

```sql
SELECT name, count(*)
FROM events
WHERE ts >= now() - INTERVAL '1 day'
GROUP BY name
ORDER BY 2 DESC
LIMIT 10;
```

## Run a server

For clients other than the local shell, run the server. It takes one YAML file
describing the instance:

```yaml
# pivot.yaml
metastore:
  datastores:
    analytics:
      kind: delta
      location: /var/lib/pivot/analytics
      default: true
```

```sh
./target/release/pivot server --config pivot.yaml
```

It listens on `127.0.0.1:5432` and speaks the PostgreSQL wire protocol, so any
PostgreSQL client connects:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## Next

- [Reference](/docs/reference/) for the statements, types, functions and
  configuration keys in full.
- [Configuration](/docs/reference/configuration/) for object-store credentials,
  the disk cache, TLS, and users.
