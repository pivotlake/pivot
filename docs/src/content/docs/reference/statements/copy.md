---
title: "COPY"
description: Stream Arrow IPC rows from a client into a table.
sidebar:
  order: 3
---

`COPY ... FROM STDIN` ingests an Arrow IPC stream over the PostgreSQL wire
protocol.

## Example

For an existing `events` table with `id` and `region` columns, a client starts
its copy session with:

```sql
COPY events (id, region) FROM STDIN WITH (FORMAT arrow);
```

The client then sends Arrow IPC bytes through the wire protocol's copy
interface and finishes the stream. The SQL statement opens the copy session;
the row data is sent separately.

## Syntax

```sql
COPY table_name [(column [, ...])]
FROM STDIN WITH (FORMAT arrow);
```

## Columns and format

An explicit column list chooses the target columns. Columns omitted from the
list are filled with `NULL`.

`FORMAT arrow` is required for the supported form. Arrow IPC is the only
supported copy format.

## Loading a local file in the shell

In [`pivot open`](/docs/reference/cli/#loading-a-file), `\copy` runs the same
`COPY` with rows read from a local file:

```text
\copy events (id, region) FROM 'events.arrow' WITH (FORMAT arrow)
```

## Limitations

CSV and plain-text copy input are not supported. This form receives data from
a client; it does not read a named server-side file.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/) - create the target table.
- [INSERT](/docs/reference/statements/insert/) - append SQL values or query results.
- [Adopt existing Parquet files](/docs/reference/statements/create-table/#adopt-existing-parquet-files) - register files already in storage.
