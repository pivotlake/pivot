---
title: Real-time analytics
description: Ingest events and query fresh analytical data over the Postgres wire protocol.
sidebar:
  order: 3
---

Real-time analytical applications need recent events to become queryable
quickly without moving the full dataset into a separate serving system. Pivot
combines writes and low-latency analytical queries over Delta Lake tables.

### How Pivot fits

1. Applications append events with `INSERT`, `INSERT ... SELECT`, or
   `COPY ... FROM STDIN` using Arrow IPC.
2. Pivot commits those rows to open Parquet and Delta Lake metadata.
3. Applications, dashboards, and Postgres clients query the updated tables
   through the same server.
4. Background compaction combines small files to keep later scans efficient.

Changes committed by the running Pivot process are published to its catalog
immediately. Changes written by another engine become visible on the server's
next refresh; the default refresh interval is 30 seconds and is configurable.

### Query serving

Pivot splits surviving Parquet row groups across its dispatch workers and
prunes files using statistics. This is a good fit for concurrent filters,
aggregations, and grouped time-window queries over append-heavy data.

Clients connect through the Postgres wire protocol, so applications can use
existing Postgres drivers. Wire compatibility does not mean every PostgreSQL
feature is supported; consult the [SQL reference](/docs/reference/sql-statements/).

### Current fit

Pivot does not currently include a managed streaming connector. Use an
application or ingestion service to send rows through the supported SQL and
Arrow interfaces. Transactions are accepted for driver compatibility but do
not group multiple statements atomically.
