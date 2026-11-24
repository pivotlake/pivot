SELECT count(*), sum(line_number)
FROM (
    SELECT row_number() OVER (
        PARTITION BY l_suppkey ORDER BY l_shipinstruct, l_shipdate
    ) AS line_number
    FROM lineitem
) numbered;
