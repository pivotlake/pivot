CREATE VIEW tpch_flat AS
SELECT *
FROM read_parquet('{source}');
