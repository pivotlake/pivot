-- ClickBench's upstream DuckDB setup. `binary_as_string=True` decodes the
-- parquet string columns (stored as BLOB) as VARCHAR so `URL LIKE ...` binds,
-- and `make_date(EventDate)` turns EventDate (days since epoch) into a real
-- DATE.
--
-- EventTime is left as its raw packed-seconds integer, matching ClickBench's
-- upstream setup and pivot. The one query that needs it as a timestamp, q42,
-- uses the DuckDB-only `toDateTime` macro in q42-duckdb.sql.
CREATE VIEW hits AS
SELECT *
    REPLACE (make_date(EventDate) AS EventDate)
FROM read_parquet('{source}', binary_as_string=True);
CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000);
