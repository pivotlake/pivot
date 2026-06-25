SELECT c_custkey,
       c_name,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       c_acctbal,
       c_nation,
       c_address,
       c_phone
FROM lineitem_flat
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R'
GROUP BY c_custkey, c_name, c_acctbal, c_phone, c_nation, c_address
ORDER BY revenue DESC, c_custkey
LIMIT 20;
