---
title: "DROP TABLE"
description: Remove a table from the catalog.
sidebar:
  order: 7
---

`DROP TABLE` removes a table from the catalog immediately. With `vacuum: true`,
a running server deletes its stored files on the next hourly
[vacuum sweep](/docs/reference/server/datastores/#maintenance) after retention
expires (four hours by default). The standalone shell does not run vacuum.

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

`CASCADE` is not supported.

Parquet files adopted from outside the table's storage directory with
[`with_pre_existing_parquets`](/docs/reference/statements/create-table/#adopt-existing-parquet-files)
are not deleted.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/)
- [System tables](/docs/reference/system-tables/)
