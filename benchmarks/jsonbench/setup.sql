-- JSONBench's schema: one column holding a whole Bluesky event as a JSON
-- document. Upstream DuckDB declares it `j JSON`; pivot declares it `VARIANT`,
-- which is the Parquet type the write path stores and which pivot's planner
-- exposes to DuckDB as its native VARIANT, so `j->'commit'->'collection'` binds.
--
-- The table is loaded by `INSERT`, not by reading Parquet in place: the runner
-- reads the newline-delimited JSON under `--source` and sends it as batched
-- `INSERT ... VALUES ('{...}')` (see `load.conf`), so the documents land through
-- the real write path. Casting each document string to `VARIANT` parses it into
-- a variant the write path then shreds.
CREATE TABLE bluesky (
    j VARIANT
);
