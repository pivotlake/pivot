SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
       SUM(CASE WHEN s_nation = 'BRAZIL' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
           / SUM(l_extendedprice * (1 - l_discount)) AS mkt_share
FROM lineitem_flat
WHERE c_region = 'AMERICA'
  AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
  AND p_type = 'ECONOMY ANODIZED STEEL'
GROUP BY EXTRACT(YEAR FROM o_orderdate)
ORDER BY o_year;
