---
title: Use cases
description: Where Pivot fits in agentic, real-time, observability, and warehouse workloads.
sidebar:
  order: 1
---

Pivot brings a shared SQL engine to open data in object storage. The same
Delta Lake tables can be queried locally, served to applications over the
Postgres wire protocol, and accessed by other engines without copying them
into a proprietary storage format.

### Agentic analytics

Give agents a local SQL environment over the source of truth. An agent can
open an S3-backed datastore with `pivot open`, inspect its schema, and run
queries without depending on a separate database service.

[Explore agentic analytics](/docs/use-cases/agentic-analytics/)

### Real-time analytics

Append events through SQL or Arrow IPC and query the resulting Delta Lake
tables from the same running server. Pivot is designed for low-latency queries
and concurrent application traffic over the Postgres wire protocol.

[Explore real-time analytics](/docs/use-cases/real-time-analytics/)

### Observability

Keep logs, metrics, and traces in open lake tables, then filter and aggregate
them with SQL. Partitioning, sorting, pruning, and compaction help organize
time-oriented telemetry datasets.

[Explore observability](/docs/use-cases/observability/)

### Data warehousing

Serve analytical tables from object storage to Postgres-compatible clients and
BI tools. Pivot can query multiple configured datastores while the underlying
Delta Lake data remains available to the rest of the data platform.

[Explore data warehousing](/docs/use-cases/data-warehousing/)

Pivot is in early development and is best suited to evaluation, experiments,
and local analysis today. Review the [supported SQL surface](/docs/reference/sql-statements/)
before choosing it for a workload.
