SELECT count(*), sum(order_revenue_share)
FROM (
    SELECT
        l_orderkey,
        l_linenumber,
        l_extendedprice * (1 - l_discount)
            / NULLIF(SUM(l_extendedprice * (1 - l_discount)) OVER (
                PARTITION BY l_orderkey
            ), 0) AS order_revenue_share
    FROM lineitem
) shares;
