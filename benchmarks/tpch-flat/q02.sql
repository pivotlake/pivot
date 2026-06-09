SELECT
    p_size,
    s_regionkey,
    COUNT(*) AS line_count,
    SUM(ps_supplycost_cents) AS sum_supply_cost
FROM tpch_flat
WHERE s_region_name = 'EUROPE'
  AND contains(p_type, 'BRASS')
GROUP BY p_size, s_regionkey
ORDER BY sum_supply_cost ASC
LIMIT 10;
