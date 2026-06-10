-- DuckDB override for q18: the DuckDB view keeps EventTime as its raw
-- packed-seconds integer (see run-duckdb.sh), so extract() needs the inline
-- toDateTime conversion — applied per row, exactly as ClickBench's upstream
-- q18 does against this storage.
SELECT UserID, extract(minute FROM toDateTime(EventTime)) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;
