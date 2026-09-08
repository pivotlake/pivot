---
title: "Arithmetic"
description: Add, subtract, multiply, divide, and adjust dates and timestamps.
sidebar:
  order: 4
---

Pivot supports binary arithmetic on numeric expressions and limited interval
arithmetic on dates and timestamps.

| Operator | Result |
| --- | --- |
| [Addition](#addition) | Sum of two values. |
| [Subtraction](#subtraction) | Difference between two values. |
| [Multiplication](#multiplication) | Product of two values. |
| [Division](#division) | Floating-point quotient. |

## Addition

```sql
left_value + right_value
```

Adds two numeric values.

```sql
SELECT range + 2 AS result FROM range(1, 2);
-- result: 3
```

## Subtraction

```sql
left_value - right_value
```

Subtracts the right numeric value from the left.

```sql
SELECT range - 2 AS result FROM range(5, 6);
-- result: 3
```

## Multiplication

```sql
left_value * right_value
```

Multiplies two numeric values.

```sql
SELECT range * 3 AS result FROM range(2, 3);
-- result: 6
```

## Division

```sql
left_value / right_value
```

Returns a floating-point quotient. Dividing two `REAL` values returns `REAL`;
other numeric pairs use `DOUBLE`.

```sql
SELECT range / 2 AS result FROM range(5, 6);
-- result: 2.5
```

## Date and time arithmetic

A `DATE` or `TIMESTAMP` can be adjusted by a constant `INTERVAL`.

```sql
SELECT ts + INTERVAL '1 hour' AS later
FROM (VALUES (TIMESTAMP '2026-09-09 12:00:00')) AS input(ts);
-- later: 2026-09-09 13:00:00
```

Month and year offsets are not supported. A `DATE` accepts only whole-day
offsets.

## Related

- [Numeric types](/docs/reference/data-types/#numeric-types)
- [Dates and times](/docs/reference/functions/datetime/)
