SELECT
    l_orderkey,
    o_shippriority,
    SUM(l_disc_price_cents) AS revenue
FROM tpch_flat
WHERE c_mktsegment = 'BUILDING'
  AND o_orderdate < '1995-03-15'
  AND l_shipdate > '1995-03-15'
GROUP BY l_orderkey, o_shippriority
ORDER BY revenue DESC
LIMIT 10;
