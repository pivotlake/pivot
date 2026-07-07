-- DuckDB override for q40: the shared q40.sql wraps EventDate in make_date()
-- (and EventTime in make_timestamp() where used) to convert pivot's raw integer
-- representation. The DuckDB view already exposes EventDate as a real DATE, so
-- the wrapper does not bind there; compare it directly (EventTime, when needed,
-- goes through the view's toDateTime macro), exactly like q42-duckdb.sql.
SELECT URLHash, EventDate AS EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;
