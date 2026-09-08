---
title: "CREATE SCHEMA"
description: Create a namespace for tables in a datastore.
sidebar:
  order: 5
---

`CREATE SCHEMA` adds a namespace for tables to a datastore.

## Example

```sql
CREATE SCHEMA analytics;
CREATE TABLE analytics.events (id BIGINT, region VARCHAR);
```

## Syntax

```sql
CREATE SCHEMA [IF NOT EXISTS] [datastore.]schema;
```

## Parameters

`schema` is the new namespace. Qualify it with a configured datastore name to
create it there; otherwise Pivot uses the default datastore.

`IF NOT EXISTS` leaves an existing schema in place.

## Additional example

Given a datastore named `lake`:

```sql
CREATE SCHEMA IF NOT EXISTS lake.analytics;
```

## Limitations

`CREATE OR REPLACE SCHEMA` is not supported.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/)
- [Datastores and storage credentials](/docs/reference/server/datastores/)
