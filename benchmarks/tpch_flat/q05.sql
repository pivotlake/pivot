SELECT s_nation AS n_name,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem_flat
WHERE s_region = 'ASIA'
  AND c_nation = s_nation
  AND o_orderdate >= DATE '1994-01-01'
  AND o_orderdate < DATE '1995-01-01'
GROUP BY s_nation
ORDER BY revenue DESC;
