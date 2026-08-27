---
title: Observability
description: Query logs, metrics, and traces stored as open lake data.
sidebar:
  order: 4
---

Observability workloads combine continuous writes with selective queries over
large time-oriented datasets. Pivot can serve telemetry stored in Delta Lake
without moving it into a proprietary database format.

### How Pivot fits

1. An existing collector or pipeline writes logs, metrics, or traces into
   Delta Lake tables.
2. Tables are partitioned and sorted around common filters such as time,
   service, or tenant.
3. Pivot prunes irrelevant files and row groups before reading their data.
4. Dashboards and investigation tools query Pivot through the Postgres wire
   protocol.

Structured columns work well for common dimensions. The `VARIANT` type can
hold less regular attributes while still allowing fields to be addressed in
queries.

### Operations

Compaction merges small Parquet files created by frequent ingestion. The
read-only `system` tables expose table sizes, live files, column storage, and
the server's memory blocks, making the physical layout visible through SQL.

See [table options](/docs/reference/table-options/) for partitioning and
sorting, and [system tables](/docs/reference/system-tables/) for operational
metadata.

### Current fit

Pivot is the query layer in this pattern, not a complete telemetry pipeline.
Collection, retention policy, alerting, and visualization remain the
responsibility of the surrounding observability stack.
