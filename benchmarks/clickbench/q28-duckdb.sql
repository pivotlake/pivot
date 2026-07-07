-- DuckDB override for q28: the shared q28.sql calls REGEXP_JIT_REPLACE, pivot's
-- JIT-compiled variant of regexp_replace. DuckDB only has the standard
-- regexp_replace, which is semantically identical here.
SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;
