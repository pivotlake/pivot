---
title: "Aggregate functions"
description: Reduce rows to counts, sums, averages, and other aggregate values.
sidebar:
  order: 1
---

Aggregate functions combine input rows into one result, or one result per
`GROUP BY` key. See [SELECT](/docs/reference/statements/select/#grouping-and-aggregates)
for grouping and `HAVING`.

| Function | Result |
| --- | --- |
| [avg](#avg) | Arithmetic mean. |
| [count](#count) | Row count or count of values. |
| [first](#first) | One input value. |
| [min and max](#min-and-max) | Smallest or largest value. |
| [sum](#sum) | Sum of numeric values. |

## avg

```sql
avg(number)
```

Accepts a numeric expression and returns its arithmetic mean.

```sql
SELECT avg(range) AS mean FROM range(1, 4);
-- mean: 2.0
```

## count

```sql
count(*)
count(expression)
count(DISTINCT expression)
```

Returns a `BIGINT` count. `count(*)` counts input rows; `count(expression)`
counts non-NULL values. `DISTINCT` counts distinct non-NULL values.

```sql
SELECT count(*) AS rows, count(DISTINCT region) AS regions
FROM (VALUES ('eu'), ('eu'), ('us')) AS input(region);
-- rows: 3, regions: 2
```

`DISTINCT` is not supported by the other aggregates.

## first

```sql
first(expression)
```

Returns one input value. Which value is returned is unspecified without an
ordering guarantee, so do not use it to choose the earliest or latest row.

```sql
SELECT first(range) AS value FROM range(1, 2);
-- value: 1 (the input has just one row)
```

## min and max

```sql
min(expression)
max(expression)
```

Return the minimum or maximum input value. The result follows the expression's
type: cast before aggregating when a different comparison is intended.

```sql
SELECT min(range) AS smallest, max(range) AS largest FROM range(1, 4);
-- smallest: 1, largest: 3
```

## sum

```sql
sum(number)
```

Returns the sum of a numeric expression. Integer sums use a `HUGEINT` result
to avoid narrow integer overflow; see [numeric types](/docs/reference/data-types/#numeric-types).

```sql
SELECT sum(range) AS total FROM range(1, 4);
-- total: 6
```
