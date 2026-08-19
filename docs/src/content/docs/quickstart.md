---
title: Quickstart
description: Start a server, create a table, run a query.
---

## Build and run

```sh
cargo build --release -p server
./target/release/pivotdb-server --config pivot.yaml
```

The server listens on the Postgres wire protocol, so any Postgres client
works:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## Load data

```sql
CREATE TABLE events (id BIGINT, name TEXT, ts TIMESTAMPTZ);
COPY events FROM '/path/to/events.parquet';
```

## Query

```sql
SELECT name, count(*)
FROM events
WHERE ts >= now() - interval '1 day'
GROUP BY name
ORDER BY 2 DESC
LIMIT 10;
```
