---
title: Creating Tables
sidebar:
  order: 2
---

Creating tables and ingesting data is only available in pivot datastores (in contrast to connecting to pre-existing 
Iceberg datastores).

As an example, create a table of NYC taxi trips, then fill it three ways: with
SQL, from Parquet in object storage, and from raw Arrow IPC data with `\copy`.

```sql
CREATE TABLE trips (
  vendor_id INTEGER,
  pickup_at TIMESTAMP,
  dropoff_at TIMESTAMP,
  passenger_count BIGINT,
  trip_distance DOUBLE,
  pickup_location_id INTEGER,
  dropoff_location_id INTEGER,
  fare_amount DOUBLE,
  tip_amount DOUBLE,
  total_amount DOUBLE
) WITH (sort_by = 'pickup_at');
```

The `sort_by` key organizes our table physically by the specified column. If a query normally only references a subset 
of the `sort_by` key, this will allow pivot to skip any parquets that don't have any rows within the range of the query.
See [Soft ordering data](/docs/database/why-pivot-is-fast/#soft-ordering-data).

To insert data:

```sql
-- Insert rows with SQL
INSERT INTO trips VALUES
  (2, '2024-01-01 00:57:55', '2024-01-01 01:17:43', 1, 1.72, 186, 79, 17.7, 0.0, 22.7),
  (1, '2024-01-01 00:03:00', '2024-01-01 00:09:36', 1, 1.8, 140, 236, 10.0, 3.75, 18.75);

-- Load January 2024 (about 3 million trips) from Parquet in a public bucket
INSERT INTO trips
SELECT VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
       trip_distance, PULocationID, DOLocationID, fare_amount, tip_amount,
       total_amount
FROM read_parquet('s3://pivotlake-examples/nyc-taxi/yellow_tripdata_2024-01.parquet');

-- Ingest raw Arrow IPC data from a local file (works in psql too). Download it first:
-- curl -O https://pivotlake-examples.s3.amazonaws.com/nyc-taxi/yellow_tripdata_2024-02_sample.arrow
\copy trips FROM 'yellow_tripdata_2024-02_sample.arrow' WITH (FORMAT arrow)
```

