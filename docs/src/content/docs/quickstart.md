---
title: Quickstart
description: Start a server, create a table, run a query.
---

## Install on Debian or Ubuntu

```sh
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://packages.pivotlake.io/keys/pivotlake-archive-key.asc |
  sudo gpg --dearmor --yes -o /usr/share/keyrings/pivotlake-archive-keyring.gpg

ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/pivotlake-archive-keyring.gpg arch=${ARCH}] https://packages.pivotlake.io/deb stable main" |
  sudo tee /etc/apt/sources.list.d/pivotlake.list
sudo apt-get update
sudo apt-get install -y pivot
```

Use `testing` instead of `stable` in the repository line to receive release
candidates. Then inspect the packaged service:

```sh
sudo systemctl status pivot
```

## Build from source

```sh
cargo build --release -p bin --bin pivot
./target/release/pivot server --config pivot.yaml
```

The server listens on the Postgres wire protocol, so any Postgres client
works:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## Load data

```sql
CREATE TABLE events (id BIGINT, name TEXT, ts TIMESTAMP) WITH (
  with_pre_existing_parquets = '/path/to/events'
);
```

The directory can contain one or more parquet files with the declared schema.

## Query

```sql
SELECT name, count(*)
FROM events
WHERE ts >= now() - interval '1 day'
GROUP BY name
ORDER BY 2 DESC
LIMIT 10;
```
