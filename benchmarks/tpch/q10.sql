SELECT
    o_custkey,
    c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal,
    c_nation,
    c_address,
    c_phone,
    c_comment
FROM tpch_flat
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R'
GROUP BY o_custkey, c_name, c_acctbal, c_phone, c_nation, c_address, c_comment
ORDER BY revenue DESC, o_custkey
LIMIT 20;
