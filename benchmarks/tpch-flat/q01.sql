SELECT
    l_returnflag_code,
    l_linestatus_code,
    COUNT(*) AS line_count,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice_cents) AS sum_base_price,
    SUM(l_disc_price_cents) AS sum_disc_price,
    SUM(l_charge_cents) AS sum_charge
FROM tpch_flat
WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag_code, l_linestatus_code
ORDER BY l_returnflag_code, l_linestatus_code
LIMIT 10;
