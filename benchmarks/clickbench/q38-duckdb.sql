-- DuckDB override for q38: the shared q38.sql wraps EventDate in make_date()
-- (and EventTime in make_timestamp() where used) to convert pivot's raw integer
-- representation. The DuckDB view already exposes EventDate as a real DATE, so
-- the wrapper does not bind there; compare it directly (EventTime, when needed,
-- goes through the view's toDateTime macro), exactly like q42-duckdb.sql.
SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
