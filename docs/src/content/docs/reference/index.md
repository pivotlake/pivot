---
title: Reference
description: The exact SQL surface, configuration keys, and interfaces pivotdb serves.
sidebar:
  order: 1
---

This section states what pivotdb accepts, one page per surface. It is written
to be looked things up in rather than read through.

| Page | Covers |
| --- | --- |
| [SQL statements](/docs/reference/sql-statements/) | Every statement, with its synopsis and its unsupported clauses |
| [Data types](/docs/reference/data-types/) | Column types, their storage, and how they cast |
| [Functions](/docs/reference/functions/) | Operators, scalar functions, aggregates, table functions |
| [System tables](/docs/reference/system-tables/) | The `system` datastore: catalog and memory introspection |
| [Configuration](/docs/reference/configuration/) | The config file, environment variables, session variables |
| [Command line](/docs/reference/cli/) | `pivot open`, `pivot server`, shell meta-commands |
| [HTTP API](/docs/reference/http-api/) | The endpoints behind the bundled dashboard |

## How the SQL surface is defined

A statement is parsed and bound by a DuckDB-compatible parser and binder, and
the bound plan is then compiled into pivotdb's own dataflow engine. The dialect
a query is written in is therefore DuckDB's, and the surface that runs is the
part of it the engine compiles.

That surface is an allowlist. A plan holding an operator, expression, function
or type pivotdb does not implement is rejected while it is being built, and the
error names the part that stopped it:

```
Unsupported scalar function: upper
Unsupported join type: OUTER
TRY_CAST is not supported; use CAST, which fails on unconvertible values
```

Nothing outside the surface is silently rewritten, approximated, or run at
reduced fidelity, so a statement that answers is a statement that ran as
written. Each page below marks what is out of the surface today under a
**Not supported** heading.

## Version

This reference describes pivotdb 0.1.0.
