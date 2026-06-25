SELECT s_nation AS nation,
       EXTRACT(YEAR FROM o_orderdate) AS o_year,
       SUM(l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity) AS sum_profit
FROM lineitem_flat
WHERE p_name LIKE '%green%'
GROUP BY s_nation, EXTRACT(YEAR FROM o_orderdate)
ORDER BY s_nation, o_year DESC;
