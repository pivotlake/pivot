-- Grouped by o_custkey alone: it functionally determines every other customer
-- column, so MIN() returns each one's single exact value. This keeps the group
-- key a single integer (a multi-key group that mixes a float column is not
-- supported) while returning every standard column.
SELECT
    o_custkey,
    MIN(c_name) AS c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    MIN(c_acctbal) AS c_acctbal,
    MIN(c_nation) AS c_nation,
    MIN(c_address) AS c_address,
    MIN(c_phone) AS c_phone,
    MIN(c_comment) AS c_comment
FROM tpch_flat
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R'
GROUP BY o_custkey
ORDER BY revenue DESC, o_custkey
LIMIT 20;
