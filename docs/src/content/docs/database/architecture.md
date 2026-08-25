---
title: Architecture
description: How a query becomes parallel work over parquet files.
sidebar:
  order: 1
---

A query arrives on the Postgres wire, is planned, and is then dispatched as
batches of work over the files that survive pruning.

```
SQL ──> planner ──> dispatch pool ──> datastore ──> parquet in object storage
                         │
                         └──> engine operators (scan, filter, group, sort)
```

### Planning

The planner resolves the statement against the catalog and pushes predicates
down to the scan so that row groups can be pruned on their statistics before
any bytes are read.

### Dispatch

Dispatch splits the surviving row groups across worker threads. A single row
group can be split across workers when its decode is expensive enough to be
worth sharing.
