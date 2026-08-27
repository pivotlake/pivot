---
title: Reference overview
description: Entry points for pivotdb SQL, functions, types, metadata, and configuration.
sidebar:
  hidden: true
  order: 0
---

These pages describe the names, forms, and defaults implemented by pivotdb.
Postgres wire compatibility lets Postgres clients connect, but the supported
SQL surface is the one documented here.

| Area | What it covers |
| --- | --- |
| [SQL statements](/docs/reference/sql-statements/) | Queries, data ingestion, DDL, maintenance, and session statements. |
| [Functions](/docs/reference/functions/) | Aggregate, scalar, and table-valued functions. |
| [Data types](/docs/reference/data-types/) | Stored column types and query-only result types. |
| [Table options](/docs/reference/table-options/) | Partitioning, sorting, and adopting existing parquet files. |
| [System tables](/docs/reference/system-tables/) | Catalog, file, and memory metadata available through SQL. |
| [Configuration](/docs/reference/configuration/) | Server, datastore, credentials, authentication, and environment settings. |

SQL may parse successfully before pivotdb determines that a particular plan,
type, expression, or statement variant is not implemented. Keep to the forms
listed in this section when portability to pivotdb matters.
