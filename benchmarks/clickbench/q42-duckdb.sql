-- DuckDB override for q42: the shared q42.sql does DATE_TRUNC('minute', EventTime)
-- directly, which needs EventTime as a timestamp. The DuckDB view keeps EventTime
-- as its raw packed-seconds integer (so q24/q26's ORDER BY EventTime sort the
-- integer, not a per-row conversion), so here we convert it inline via the
-- toDateTime macro — only on the rows that survive the filter — exactly as
-- ClickBench's upstream q42 does.
SELECT DATE_TRUNC('minute', toDateTime(EventTime)) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', toDateTime(EventTime)) ORDER BY DATE_TRUNC('minute', toDateTime(EventTime)) LIMIT 10 OFFSET 1000;
