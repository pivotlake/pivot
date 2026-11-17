---
title: "DELETE"
description: Remove rows by dropping whole partitions of an Iceberg table.
sidebar:
  order: 4
---

`DELETE` removes rows from an Iceberg table by dropping whole partitions. It
writes no data file: the files of every partition the conditions cover leave
the table in one snapshot. Each statement commits individually.

## Example

```sql
CREATE TABLE events (ts TIMESTAMP, id BIGINT) PARTITIONED BY (day(ts));

-- Drop every day before 15 November.
DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 00:00:00';
```

The statement answers with the number of rows it removed.

## Syntax

```sql
DELETE FROM table_name [WHERE condition];
```

Without a `WHERE` clause, every row is deleted.

## Partition bounds

A `DELETE` succeeds only when its conditions cover whole partitions. A
partition that holds rows and is only partly covered makes the statement fail
with an error that names the partition, and nothing is deleted.

For a table partitioned by `day(ts)`:

| Statement | Result |
| --- | --- |
| `DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 00:00:00'` | Drops the days before the 15th. |
| `DELETE FROM events WHERE ts < CURRENT_DATE` | Drops the days before today. |
| `DELETE FROM events WHERE ts >= TIMESTAMP '2026-11-14 00:00:00' AND ts < TIMESTAMP '2026-11-15 00:00:00'` | Drops the 14th. |
| `DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 10:00:00'` | Fails when the 15th holds rows: the condition cuts that day. |
| `DELETE FROM events WHERE id = 7` | Fails: `id` is not what the table is partitioned by. |

A table partitioned by a column drops the values the conditions name:

```sql
CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region);

DELETE FROM visits WHERE region IN ('eu', 'us');
```

## Conditions

A condition compares a column with a constant (`=`, `<>`, `<`, `<=`, `>`,
`>=`, `BETWEEN`, `IN`, `NOT IN`) or tests it with `IS NULL` or `IS NOT NULL`.
Conditions combine with `AND` and `OR`. Constant expressions such as
`CURRENT_DATE - 7` are evaluated before the conditions are checked.

Rows whose partition column is `NULL` are deleted only by `IS NULL` or by a
`DELETE` without conditions, as in any SQL comparison.

## Storage

The dropped files stay in storage until
[vacuum](/docs/reference/server/datastores/#vacuum) removes them, once the
snapshot that dropped them is older than the table's retention window. In an
Iceberg catalog datastore, the catalog's own maintenance removes them.

## Limitations

- `DELETE` runs on Iceberg tables. A datastore in the Delta Lake table format
  refuses it.
- A table that is not partitioned accepts only a `DELETE` without conditions.
- A condition on a column the table partitions by `bucket` or `truncate`, or
  on a floating-point partition column, rarely covers whole partitions.
- A bound at or before 1970-01-01 can be refused when the partition it falls
  in holds rows.
- `USING`, subqueries, and `RETURNING` are not supported.
- `BEGIN` and `ROLLBACK` do not group or undo deletes; see
  [transaction behavior](/docs/reference/statements/transactions/).

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/) - choose what a table is partitioned by.
- [INSERT](/docs/reference/statements/insert/) - append rows.
- [DROP TABLE](/docs/reference/statements/drop-table/) - remove a whole table.
