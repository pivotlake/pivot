SELECT
    s_nation AS n_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM tpch_flat
WHERE c_nation = s_nation
  AND s_region = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1995-01-01'
GROUP BY s_nation
ORDER BY revenue DESC, s_nation;
