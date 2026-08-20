---
title: HTTP API
description: The endpoints the bundled dashboard is built on.
sidebar:
  order: 8
---

Setting [`server.http_bind`](/docs/reference/configuration/#server) serves the
bundled web dashboard, and with it the JSON endpoints below. Because they run
inside the server, they read live state the PostgreSQL wire does not expose:
the catalog, and the process's own CPU and memory.

The API has no authentication and is not versioned. It is read-only except for
`/api/query`, and it is meant for a trusted network rather than the public
internet.

Any path that is not one of these serves the dashboard itself, falling back to
its `index.html` so client-side routes resolve.

## GET /api/health

```json
{ "status": "ok" }
```

## GET /api/overview

The default datastore and the process: its tables with their columns, file
counts, sizes, partition and sort keys and row counts, alongside the machine's
CPU, memory and disk use and the server process's own share of it. This is what
the dashboard's landing page is drawn from.

## POST /api/query

Runs one SQL statement on the same planner and dispatch pool as every other
query, in-process, and returns the rows as JSON. This is the only endpoint that
is not read-only, and it accepts the same statements the wire protocol does,
apart from `COPY ... FROM STDIN`, which needs the protocol's copy-in channel.

```sh
curl -s http://127.0.0.1:8081/api/query \
  -H 'content-type: application/json' \
  -d '{"sql":"SELECT name FROM system.tables ORDER BY name"}'
```

```json
{
  "columns": [{ "name": "name", "col_type": "Utf8View" }],
  "rows": [["events"], ["orders"]],
  "row_count": 2,
  "elapsed_ms": 3.4
}
```

Values are rendered as strings, and a `NULL` is `null`. A statement that fails
still answers `200`, with the message in an `error` field and no rows.

## GET /api/tables/{table}/files

One page of a table's data files, from the default datastore. The table is
named without a schema, and resolves in `main`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `offset` | `0` | Files to skip |
| `limit` | `50` | Files to return, capped at 500 |

```json
{
  "items": [{ "path": "part-00000-....parquet", "size": 134217728 }],
  "total": 412
}
```

`total` is the whole file count, not the page's, so a client can page through
it. A table or datastore that does not exist answers with an empty page rather
than an error.

## GET /api/datastores/{datastore}/tables/{table}/files

The same page, from a named datastore rather than the default one.

## Reading the same data in SQL

Everything these endpoints report about the catalog is also queryable, with
more freedom, through the [system tables](/docs/reference/system-tables/).
