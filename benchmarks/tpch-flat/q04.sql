SELECT
    o_orderpriority,
    COUNT(*) AS c
FROM tpch_flat
WHERE o_orderdate >= '1993-07-01'
  AND o_orderdate < '1993-10-01'
  AND l_commitdate < l_receiptdate
GROUP BY o_orderpriority
ORDER BY c DESC
LIMIT 10;
