-- DuckDB override for q40: EventDate is a group key that reaches the output,
-- and the DuckDB view converts it to a real DATE while pivot keeps the stored
-- day count. Convert back to day counts so both engines emit the same rows.
SELECT URLHash, EventDate - DATE '1970-01-01' AS EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;
