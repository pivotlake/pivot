SELECT count(*), sum(quantity_so_far), sum(price_change)
FROM (
    SELECT
        sum(l_quantity) OVER (
            PARTITION BY l_suppkey ORDER BY l_shipdate, l_orderkey, l_linenumber
        ) AS quantity_so_far,
        l_extendedprice - lag(l_extendedprice) OVER (
            PARTITION BY l_suppkey ORDER BY l_shipdate, l_orderkey, l_linenumber
        ) AS price_change
    FROM lineitem
) per_supplier;
