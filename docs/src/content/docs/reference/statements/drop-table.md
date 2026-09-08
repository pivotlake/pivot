---
title: "DROP TABLE"
description: Remove a table from the catalog.
sidebar:
  order: 7
---

`DROP TABLE` removes a table from the catalog. It does not immediately delete
the table's data files.

## Example

```sql
DROP TABLE IF EXISTS staging_events;
```

## Syntax

```sql
DROP TABLE [IF EXISTS] [[datastore.]schema.]table;
```

## Parameters

Qualify the table name with a schema and, optionally, a datastore. An
unqualified name uses the default datastore and schema.

`IF EXISTS` permits the statement to succeed when the table is absent.

## Additional example

For a table in the `analytics` schema:

```sql
DROP TABLE analytics.events;
```

## Limitations

`CASCADE` is not supported. Removing the catalog entry is distinct from
reclaiming storage; see [datastore maintenance](/docs/reference/server/datastores/#maintenance).

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/)
- [System tables](/docs/reference/system-tables/)
