-- DuckDB override for q39: the shared q39.sql wraps EventDate in make_date()
-- (and EventTime in make_timestamp() where used) to convert pivot's raw integer
-- representation. The DuckDB view already exposes EventDate as a real DATE, so
-- the wrapper does not bind there; compare it directly (EventTime, when needed,
-- goes through the view's toDateTime macro), exactly like q42-duckdb.sql.
SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
