SELECT count(*), sum(revenue_rank)
FROM (
    SELECT rank() OVER (
        PARTITION BY l_suppkey ORDER BY l_extendedprice * (1 - l_discount) DESC
    ) AS revenue_rank
    FROM lineitem
) ranked;
