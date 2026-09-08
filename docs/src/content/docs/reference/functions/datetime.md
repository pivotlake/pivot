---
title: "Dates & times"
description: Construct timestamps, extract fields, and truncate time values.
sidebar:
  order: 3
---

Pivot's `TIMESTAMP` is timezone-free and uses microsecond precision. Normalize
timezone-sensitive input before loading it.

| Function | Result |
| --- | --- |
| [now](#now) | Timestamp fixed for the statement. |
| [date_trunc](#date_trunc) | Timestamp truncated to a supported unit. |
| [extract](#extract) | A date or timestamp field. |
| [make_date](#make_date) | Date from days since the epoch. |
| [make_timestamp](#make_timestamp) | Timestamp from microseconds since the epoch. |

## now

```sql
now()
```

Takes no arguments. Returns a timezone-free `TIMESTAMP` fixed for the duration
of the statement.

```sql
SELECT now() AS statement_time;
```

The result depends on when the statement starts.

## date_trunc

```sql
date_trunc(part, timestamp)
```

Returns a timestamp truncated to the specified unit. `part` must be a constant:
`second`, `minute`, `hour`, or `day`.

```sql
SELECT date_trunc('hour', ts) AS hour
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- hour: 2026-09-09 12:00:00
```

## extract

```sql
extract(part FROM value)
```

Extracts a field from a date or timestamp and returns its numeric value.

```sql
SELECT extract(year FROM ts) AS year
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- year: 2026
```

Supported parts are `epoch`, `second`, `millisecond`, `microsecond`, `minute`,
`hour`, `day`, `month`, `quarter`, `year`, `decade`, `century`, `millennium`,
`dayofweek` (`dow`), `isodow`, `dayofyear` (`doy`), and `week` (`weekofyear`).

## make_date

```sql
make_date(days)
```

Interprets an integer as signed days since 1970-01-01 and returns a `DATE`.

```sql
SELECT make_date(days) AS date
FROM (VALUES (1)) AS input(days);
-- date: 1970-01-02
```

## make_timestamp

```sql
make_timestamp(microseconds)
```

Interprets an integer as microseconds since 1970-01-01 and returns a
`TIMESTAMP`.

```sql
SELECT make_timestamp(microseconds) AS ts
FROM (VALUES (1000000::BIGINT)) AS input(microseconds);
-- ts: 1970-01-01 00:00:01
```

## Related

- [Date and time arithmetic](/docs/reference/functions/arithmetic/#date-and-time-arithmetic)
- [Date and time types](/docs/reference/data-types/#dates-times-and-intervals)
