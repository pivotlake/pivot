---
title: Roadmap
description: What is available in Pivot today and what the project is working toward.
---

Pivot is evolving quickly. This roadmap communicates direction, not release
dates or compatibility commitments. Scope and sequencing may change as the
project develops.

### Available

#### Delta Lake tables

Pivot can read and write Delta Lake tables backed by Parquet. It supports
appending data with `INSERT` and `COPY`, adopting existing Parquet files, and
compacting small or overlapping files. See [Architecture](/docs/database/architecture/)
and [SQL statements](/docs/reference/sql-statements/).

#### Local and remote object storage

Datastores can live on a local filesystem or supported object stores. Queries
operate on open table data rather than requiring a proprietary copy. See
[Configuration](/docs/reference/configuration/).

### In progress

#### Iceberg tables

Work is underway to let Pivot query existing tables through an Apache Iceberg
REST catalog. The initial scope is read support; write behavior and the final
configuration surface are not yet committed.

#### Shared PostgreSQL metastore

A PostgreSQL-backed metastore is being developed so multiple Pivot servers can
share the same datastore registry and user directory. This provides a shared
control plane and load-balanced query serving; it does not make one query run
across several Pivot servers. See
[Sharing a metastore across a cluster](/docs/database/architecture/#sharing-a-metastore-across-a-cluster).

### Planned

#### `DELETE` support

Pivot plans to support `DELETE FROM` for removing matching rows from Delta Lake
tables. The exact SQL surface and physical deletion strategy are still being
designed, so they may change before the feature becomes available.

### How to read this roadmap

- **Available** features are documented and usable in the current release.
- **In progress** features have active implementation work but may still change.
- **Planned** features are intended directions without a committed release date.

For the exact surface supported today, use the [Reference](/docs/reference/).

