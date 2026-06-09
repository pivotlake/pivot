SELECT
    s_nationkey,
    c_regionkey,
    SUM(l_disc_price_cents) AS revenue,
    COUNT(*) AS line_count
FROM tpch_flat
WHERE c_region_name = 'ASIA'
  AND o_orderdate >= '1994-01-01'
  AND o_orderdate < '1995-01-01'
GROUP BY s_nationkey, c_regionkey
ORDER BY revenue DESC
LIMIT 10;
