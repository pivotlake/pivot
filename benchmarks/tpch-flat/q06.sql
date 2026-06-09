SELECT
    SUM(l_discount_amount_cents) AS revenue
FROM tpch_flat
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount_pct >= 5
  AND l_discount_pct <= 7
  AND l_quantity < 24;
