---
title: Roadmap
description: What is being worked on now and what comes after.
---

Where the engine is heading. Items are ordered by when they are likely to land,
not by size, and nothing here carries a date. A line moves down the page when
something ahead of it turns out to be harder than it looked.

## In progress

- **Compaction that keeps up with ingest.** The selector decides which small
  files to rewrite; the remaining work is making the write pipeline hold its
  throughput while queries run against the same tables.
- **Deployment.** The Debian package ships the release binary. Next is a
  container image and a documented upgrade path for a running cluster.
- **System tables.** `datastores` and `columns` describe what a table holds.
  More of the catalog becomes queryable from SQL rather than from logs.

## Next

- **Wider type coverage.** Today a column is integer, floating point, text,
  timestamp or variant. `DECIMAL`, `DATE` and `TIME` are the next three, with
  nested types after them.
- **More of the SQL surface.** The planner rejects some join shapes and some
  aggregate expressions outright. Closing those gaps is steady work rather than
  one change, and each one that lands removes an error message.
- **Window functions.** Not implemented. They are the most common reason a
  working query has to be rewritten before it runs here.

## Later

- **Elastic clusters.** Nodes join and leave while queries are in flight,
  without a restart and without a coordinator holding the table state.
- **Writes from more directions.** Any engine that speaks Delta Lake can
  already write the tables. The goal is for that to stay true as the write path
  gets faster, rather than the fast path becoming a private one.
- **Query result caching.** Repeated dashboard queries against unchanged
  snapshots should not re-read the same row groups.

## Recently shipped

- Open a datastore directly at an object store URI from the CLI and the shell.
- Rewritten compaction selector and write pipeline.
- Predicate and variant extract pushdown into scans, including extracts that
  are provably absent being resolved at plan time.
- Manifest commits through compare-and-swap remotely and a file lock locally.
- A refusal to start when the configured buffer pool exceeds free memory.

Something missing that you need? Open an issue describing the query or the
workload rather than the feature, since the shape of the workload usually
changes what gets built.
