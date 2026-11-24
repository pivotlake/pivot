SELECT count(*), sum(l_extendedprice)
FROM (
    SELECT l_partkey, l_extendedprice
    FROM lineitem
    QUALIFY row_number() OVER (PARTITION BY l_partkey ORDER BY l_extendedprice DESC) <= 3
) top_lines;
