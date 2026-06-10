-- DuckDB override for q06: the DuckDB view converts EventDate to a real DATE
-- (make_date), so its MIN/MAX render as dates while pivot keeps the column's
-- stored day count and prints integers. Convert back to day counts here so
-- both engines emit the same rows.
SELECT MIN(EventDate) - DATE '1970-01-01', MAX(EventDate) - DATE '1970-01-01' FROM hits;
