---
title: Querying
sidebar:
  order: 3
---

To query, you can use normal Postgresql SELECT query syntax on tables in the datastore (Iceberg or Pivot), or raw parquets. 

For example, if we have the trips table, and want to find January's busiest days and their average tip:

```sql
SELECT date_trunc('day', pickup_at) AS day,
       count(*) AS trips,
       CAST(avg(tip_amount) AS DECIMAL(10, 2)) AS avg_tip
FROM trips
WHERE pickup_at >= '2024-01-01' AND pickup_at < '2024-02-01'
GROUP BY day
ORDER BY trips DESC
LIMIT 5;
```

```text
         day         | trips  | avg_tip
---------------------+--------+---------
 2024-01-27 00:00:00 | 110515 |    3.10
 2024-01-17 00:00:00 | 110365 |    3.35
 2024-01-18 00:00:00 | 110358 |    3.36
 2024-01-25 00:00:00 | 110318 |    3.48
 2024-01-20 00:00:00 | 108768 |    2.92
(5 rows)
```

Or see how tipping changes with the number of passengers:

```sql
SELECT passenger_count,
       count(*) AS trips,
       CAST(avg(tip_amount / NULLIF(fare_amount, 0)) * 100 AS DECIMAL(10, 1)) AS tip_percent
FROM trips
WHERE passenger_count BETWEEN 1 AND 6
GROUP BY passenger_count
ORDER BY passenger_count;
```

```text
 passenger_count |  trips  | tip_percent
-----------------+---------+-------------
               1 | 2271171 |        23.7
               2 |  416662 |        20.7
               3 |   93344 |        21.0
               4 |   53029 |        18.0
               5 |   34516 |        21.4
               6 |   23040 |        21.5
(6 rows)
```

To query raw parquets, you can run:
```sql
SELECT * FROM read_parquet('s3://this/is/*/where/my/parquet/is*.parquet');
```

## Next steps

- [SQL reference](/docs/reference/sql-statements/): the statements and
  functions Pivot supports.
- [CLI reference](/docs/reference/cli/): every `pivot open` option and shell
  command.
- [Configuration file](/docs/reference/configuration/) and
  [Datastores & storage credentials](/docs/reference/server/datastores/): set
  up a server for your own data.

