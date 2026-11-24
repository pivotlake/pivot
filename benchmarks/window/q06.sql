SELECT count(*), sum(line_number)
FROM (
    SELECT row_number() OVER (
        PARTITION BY l_suppkey ORDER BY NULLIF(l_linenumber, 1), l_shipdate
    ) AS line_number
    FROM lineitem
) numbered;
