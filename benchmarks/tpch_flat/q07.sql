SELECT s_nation AS supp_nation,
       c_nation AS cust_nation,
       EXTRACT(YEAR FROM l_shipdate) AS l_year,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem_flat
WHERE ((s_nation = 'FRANCE' AND c_nation = 'GERMANY')
    OR (s_nation = 'GERMANY' AND c_nation = 'FRANCE'))
  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY s_nation, c_nation, EXTRACT(YEAR FROM l_shipdate)
ORDER BY s_nation, c_nation, l_year;
