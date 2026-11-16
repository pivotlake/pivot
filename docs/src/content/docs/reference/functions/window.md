---
title: "Window functions"
description: Number, rank, compare and add up rows within partitions of a query's result.
sidebar:
  order: 8
---

A window function computes a value for each row from the other rows of the
same partition, without collapsing the rows as an aggregate does. The `OVER`
clause defines the partitions and the order of rows within them:

```sql
function(...) OVER ([PARTITION BY expression [, ...]] [ORDER BY expression [ASC | DESC] [, ...]])
```

Without `PARTITION BY`, the whole input is one partition. Without `ORDER BY`,
the order of rows within a partition is unspecified.

| Function | Result |
| --- | --- |
| [row_number](#row_number) | Position of the row within its partition. |
| [rank](#rank) | Position of the row's tie group, with gaps after ties. |
| [dense_rank](#dense_rank) | Position of the row's tie group, without gaps. |
| [percent_rank](#percent_rank) | The rank scaled to run from 0 to 1. |
| [lag](#lag-and-lead) | A value from an earlier row of the partition. |
| [lead](#lag-and-lead) | A value from a later row of the partition. |
| [sum](#sum) | The total of a value over the rows up to the current one. |

`row_number`, `rank` and `dense_rank` return a `BIGINT` that starts at 1 in
every partition. Rows that are equal on every `ORDER BY` key form a tie
group.

## row_number

```sql
row_number() OVER (...)
```

Numbers the rows of each partition 1, 2, 3 and so on, in `ORDER BY` order.
Rows of a tie group are numbered in an unspecified order.

```sql
SELECT team, score,
       row_number() OVER (PARTITION BY team ORDER BY score DESC) AS position
FROM scores;
```

## rank

```sql
rank() OVER (...)
```

Rows of a tie group share the row number of the group's first row, so the
next group's rank skips ahead: 1, 1, 3.

## dense_rank

```sql
dense_rank() OVER (...)
```

Rows of a tie group share a rank and the next group gets the next integer:
1, 1, 2.

## percent_rank

```sql
percent_rank() OVER (...)
```

Returns `(rank - 1) / (partition rows - 1)` as a `DOUBLE`: 0 for the first tie
group and 1 for a row that ranks last on its own. A partition of one row has 0.

## lag and lead

```sql
lag(expression [, offset]) OVER (...)
lead(expression [, offset]) OVER (...)
```

`lag` returns `expression` as it is in the row `offset` rows before the
current row in its partition, and `lead` in the row `offset` rows after it.
The result has the type of `expression` and is `NULL` when the partition has
no such row. `offset` is a constant that is not negative and defaults to 1.

The change in score from each team's previous row:

```sql
SELECT team, score,
       score - lag(score) OVER (PARTITION BY team ORDER BY played_at) AS change
FROM scores;
```

## sum

```sql
sum(expression) OVER (... [ROWS UNBOUNDED PRECEDING])
```

Adds up `expression` from the partition's first row. `NULL` values are
skipped, and the result is `NULL` until the partition has a value. The result
type is that of the aggregate
[sum](/docs/reference/functions/aggregates/).

Which rows are added depends on the frame:

| `OVER` clause | Rows added |
| --- | --- |
| `ORDER BY` without a frame | Up to the current row and the rows that tie with it, so tied rows share one total. |
| `ORDER BY ... ROWS UNBOUNDED PRECEDING` | Up to the current row. |
| No `ORDER BY` | The whole partition. |

```sql
SELECT team, played_at, score,
       sum(score) OVER (PARTITION BY team ORDER BY played_at) AS total_so_far
FROM scores;
```

## Keeping the first rows of each partition

`QUALIFY` filters on the result of a window function. Keep the three highest
scores of each team:

```sql
SELECT team, score
FROM scores
QUALIFY row_number() OVER (PARTITION BY team ORDER BY score DESC) <= 3;
```

A `<=`, `<` or `= 1` comparison between `row_number`, `rank` or `dense_rank` and a constant
is applied while the rows are being read, so only the rows that can pass it
are kept in memory. The same holds when the comparison is written as a `WHERE`
on a subquery that selects the function.

## Compatibility

`lag` and `lead` do not take a default value or `IGNORE NULLS`. `sum` is the
only aggregate that takes an `OVER` clause, without `DISTINCT` or `FILTER`,
and only with a frame that starts at the partition's first row and ends at
the current row. Other window functions, such as `first_value` and `ntile`,
are not supported. Window functions cannot be computed over a query that
selects a `VARIANT` column.
