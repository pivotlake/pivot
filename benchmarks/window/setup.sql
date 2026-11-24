CREATE TABLE lineitem (
    l_orderkey BIGINT,
    l_partkey BIGINT,
    l_suppkey BIGINT,
    l_linenumber INTEGER,
    l_quantity DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount DECIMAL(15,2),
    l_tax DECIMAL(15,2),
    l_returnflag VARCHAR,
    l_linestatus VARCHAR,
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct VARCHAR,
    l_shipmode VARCHAR,
    l_comment VARCHAR
) WITH (with_pre_existing_parquets = '{source}/lineitem');
