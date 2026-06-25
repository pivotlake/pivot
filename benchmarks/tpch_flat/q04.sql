SELECT o_orderpriority,
       COUNT(DISTINCT l_orderkey) AS order_count
FROM lineitem_flat
WHERE o_orderdate >= DATE '1993-07-01'
  AND o_orderdate < DATE '1993-10-01'
  AND l_commitdate < l_receiptdate
GROUP BY o_orderpriority
ORDER BY o_orderpriority;
