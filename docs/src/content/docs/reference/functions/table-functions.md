---
title: "Table functions"
description: Read Parquet files with read_parquet, or generate rows of integers or timestamps with range and generate_series.
sidebar:
  order: 6
---

Table functions produce rows and can appear in a query's `FROM` clause.

| Function | Returns |
| --- | --- |
| [read_parquet](#read_parquet) | The rows of one or more Parquet files. |
| [range](#range) | Integers or timestamps up to `stop`, excluding it. |
| [generate_series](#generate_series) | Integers or timestamps up to `stop`, including it. |

## read_parquet

```sql
read_parquet(path)
```

Reads Parquet files from local disk, Amazon S3 (`s3://`), or Google Cloud
Storage (`gs://`). The result has the files' own column names and types.

```sql
SELECT count(*)
FROM read_parquet('s3://pivotlake-examples/nyc-taxi/yellow_tripdata_2024-01.parquet');
```

`path` names a single file or a pattern. A `*` matches any run of characters
within one path segment, in the file name or in a directory name:

```sql
-- Every Parquet file directly under events/
SELECT * FROM read_parquet('s3://my-bucket/events/*.parquet');

-- part-*.parquet inside every hello/ directory one level below events/
SELECT * FROM read_parquet('gs://my-bucket/events/*/hello/part-*.parquet');
```

A `*` never crosses a `/`, so `events/*.parquet` does not read files in
subdirectories of `events/`.

Pivot authenticates to buckets with the CLI's
[object-store credentials](/docs/reference/cli/#object-store-credentials) or,
on a server, with an `s3` or `gcs`
[storage secret](/docs/reference/server/datastores/#storage-credentials). On a
server, a local path is read from the server's own disk, not the client's.

## range

```sql
range(stop)
range(start, stop[, step])
```

Returns a `BIGINT` column named `range`. The one-argument form starts at `0`;
the default step is `1`. The stop value is excluded.

```sql
SELECT * FROM range(1, 4);
-- range: 1, 2, 3
```

Use a negative step to count down:

```sql
SELECT * FROM range(3, 0, -1);
-- range: 3, 2, 1
```

`start` and `stop` can also both be `TIMESTAMP` or both
`TIMESTAMP WITH TIME ZONE`, with an `INTERVAL` `step`. The column then has the
type of the bounds:

```sql
SELECT * FROM range(TIMESTAMP '2026-09-09 12:00:00',
                    TIMESTAMP '2026-09-09 12:30:00',
                    INTERVAL '10 minutes');
-- range: 2026-09-09 12:00:00, 2026-09-09 12:10:00, 2026-09-09 12:20:00
```

Each timestamp is the one before it plus `step`. A step in months moves to the
same day of the next month, or to that month's last day when it is shorter, so
a monthly series from January 31 continues with February 28 and March 28.

A call with a `NULL` argument returns no rows.

## generate_series

```sql
generate_series(stop)
generate_series(start, stop[, step])
```

Returns a `BIGINT` column named `generate_series`. Like `range`, the
one-argument form starts at `0` and the default step is `1`, but the stop value
is included when the sequence reaches it.

```sql
SELECT * FROM generate_series(1, 3);
-- generate_series: 1, 2, 3
```

Timestamp bounds with an `INTERVAL` step work as they do for `range`:

```sql
SELECT * FROM generate_series(TIMESTAMP '2026-09-09 12:00:00',
                              TIMESTAMP '2026-09-09 12:20:00',
                              INTERVAL '10 minutes');
-- generate_series: 2026-09-09 12:00:00, 2026-09-09 12:10:00, 2026-09-09 12:20:00
```

As with `range`, a call with a `NULL` argument returns no rows.

## Arguments from columns

The arguments of `range` and `generate_series` can be columns of an earlier
item of the same `FROM` clause. The function then returns one series for each
row of that item, beside the row's own columns:

```sql
SELECT spans.id, steps.step
FROM spans, generate_series(spans.first, spans.last) AS steps(step);
```

A row whose series is empty produces no rows. `LEFT JOIN LATERAL` keeps it,
with a `NULL` step:

```sql
SELECT spans.id, steps.step
FROM spans
LEFT JOIN LATERAL generate_series(spans.first, spans.last) AS steps(step) ON true;
```

This fills the gaps of a time-bucketed aggregate: generate every bucket
between the first and the last one that has rows, and join the counts to them.

```sql
WITH counts AS (
  SELECT time_bucket(INTERVAL '10 seconds', "timestamp") AS bucket, count(*) AS count
  FROM events
  GROUP BY bucket
)
SELECT buckets.bucket, counts.count
FROM (SELECT min(bucket) AS first_bucket, max(bucket) AS last_bucket FROM counts) AS observed,
     generate_series(observed.first_bucket, observed.last_bucket, INTERVAL '10 seconds')
       AS buckets(bucket)
LEFT JOIN counts ON counts.bucket = buckets.bucket
ORDER BY buckets.bucket;
```

## Limitations

- `step` cannot be zero. The months, days, and time of an `INTERVAL` step must
  all point in the same direction: `INTERVAL '1 month -1 day'` fails.
- Timestamp bounds cannot be `infinity` or `-infinity`.
- A column that an argument of `range` or `generate_series` reads must be an
  integer, decimal, date, timestamp, or text column.
- When such a column is `NULL`, the call must return no rows. It does whenever
  the `NULL` reaches an argument. An argument that replaces the `NULL`, such as
  `CASE WHEN last IS NULL THEN 10 ELSE last END`, fails the query on that row.
- `read_parquet` takes exactly one `VARCHAR` path. The path must name a file or
  a file pattern, not a directory: `s3://my-bucket/events/` fails, and
  `s3://my-bucket/events/*.parquet` works.
- A bucket name cannot contain `*`.
- A path that matches no files fails.
- Every matched file must have the same column names and types, and each must
  contain at least one row group.
- A file with a column of an unsupported
  [data type](/docs/reference/data-types/) fails.

## Related

- [SELECT](/docs/reference/statements/select/)
- [INSERT from a query](/docs/reference/statements/insert/#insert-from-a-query)
- [Datastores & storage credentials](/docs/reference/server/datastores/)
