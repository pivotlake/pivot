---
title: Data warehousing
description: Serve warehouse-style SQL directly from open tables in object storage.
sidebar:
  order: 5
---

Pivot can provide a SQL serving layer over analytical data in object storage.
Applications and BI tools connect through the Postgres wire protocol, while
the underlying tables remain in Delta Lake and Parquet.

### How Pivot fits

1. Register local, S3, or GCS-backed Pivot datastores in the server
   configuration.
2. Choose one default datastore and expose others as named databases.
3. Query tables with ordinary SQL, qualifying names when a query crosses
   datastores.
4. Connect application backends, scripts, or BI tools with a Postgres client.

Pivot can create empty tables, adopt existing Parquet files, and append rows
with SQL or Arrow IPC. Partitioning, sorting, statistics-based pruning, and
compaction keep the physical table layout useful as data accumulates.

### An open serving layer

The storage layer is not owned by the query server. Spark, DuckDB, Trino,
Snowflake, and other Delta-aware tools can participate in the surrounding data
platform without exporting data from Pivot.

This makes Pivot suitable for architectures that want a lightweight serving
engine without giving up open formats or direct object-store access.

### Current fit

Pivot is not yet a complete replacement for a production data warehouse. SQL
coverage is intentionally limited, multi-statement transactions are not
implemented, and upgrades may include breaking changes. Evaluate the
[supported SQL](/docs/reference/sql-statements/) and
[database limits](/docs/database/limits/) against the workload first.
