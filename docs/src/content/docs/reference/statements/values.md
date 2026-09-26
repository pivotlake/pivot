---
title: "VALUES"
description: Construct rows from SQL expressions.
sidebar:
  order: 11
---

`VALUES` constructs rows directly from expressions. It can be a standalone
query or supply input rows to another statement.

## Example

```sql
VALUES (1, 'eu'), (2, 'us');
```

This returns two rows, each containing an integer and a string.

## Syntax

```sql
VALUES (expression [, ...]) [, ...];
```

## Use as query input

Give a derived table and its columns names to reference the values in a query:

```sql
SELECT region, count(*) AS events
FROM (VALUES (1, 'eu'), (2, 'us'), (3, 'eu')) AS input(id, region)
GROUP BY region
ORDER BY region;
```

The result is `eu: 2` and `us: 1`.

## Insert values

For an existing table:

```sql
INSERT INTO events (id, region) VALUES (1, 'eu'), (2, 'us');
```

## Related

- [INSERT](/docs/reference/statements/insert/)
- [SELECT](/docs/reference/statements/select/)
- [Data types](/docs/reference/data-types/)
