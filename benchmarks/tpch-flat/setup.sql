-- TPC-H "flat" schema: the 8 normalized TPC-H tables denormalized into a single
-- lineitem-grain table (one row per lineitem, its order/customer/part/supplier/
-- partsupp and both nation+region names folded in). Built by prep-tpch-flat.sh.
--
-- This lets pivot run the lineitem-centric TPC-H queries as pure single-table
-- scan+aggregate, with no join executor. Queries that live at a different grain
-- (part/supplier-only) or need anti-joins (customers with no orders) are not
-- expressible here and are intentionally absent from this suite.
--
-- Money and quantity columns are DOUBLE (the flat build casts TPC-H's DECIMAL to
-- DOUBLE) so pivot's floating aggregates apply directly. Dates are real DATE.
-- Duplicate join keys are dropped (l_orderkey == o_orderkey, l_partkey ==
-- p_partkey, l_suppkey == s_suppkey), so each key appears once.
--
-- The runner substitutes `{source}` with the value of `--source`.
CREATE TABLE tpch_flat (
    -- lineitem
    l_orderkey BIGINT,
    l_partkey BIGINT,
    l_suppkey BIGINT,
    l_linenumber INTEGER,
    l_quantity DOUBLE,
    l_extendedprice DOUBLE,
    l_discount DOUBLE,
    l_tax DOUBLE,
    l_returnflag VARCHAR,
    l_linestatus VARCHAR,
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct VARCHAR,
    l_shipmode VARCHAR,
    l_comment VARCHAR,
    -- orders (l_orderkey is the order key)
    o_custkey BIGINT,
    o_orderstatus VARCHAR,
    o_totalprice DOUBLE,
    o_orderdate DATE,
    o_orderpriority VARCHAR,
    o_clerk VARCHAR,
    o_shippriority INTEGER,
    o_comment VARCHAR,
    -- customer (o_custkey is the customer key)
    c_name VARCHAR,
    c_address VARCHAR,
    c_nationkey BIGINT,
    c_phone VARCHAR,
    c_acctbal DOUBLE,
    c_mktsegment VARCHAR,
    c_comment VARCHAR,
    -- part (l_partkey is the part key)
    p_name VARCHAR,
    p_mfgr VARCHAR,
    p_brand VARCHAR,
    p_type VARCHAR,
    p_size INTEGER,
    p_container VARCHAR,
    p_retailprice DOUBLE,
    p_comment VARCHAR,
    -- supplier (l_suppkey is the supplier key)
    s_name VARCHAR,
    s_address VARCHAR,
    s_nationkey BIGINT,
    s_phone VARCHAR,
    s_acctbal DOUBLE,
    s_comment VARCHAR,
    -- partsupp (keyed by l_partkey + l_suppkey)
    ps_availqty INTEGER,
    ps_supplycost DOUBLE,
    ps_comment VARCHAR,
    -- geography, resolved to names on both sides
    c_nation VARCHAR,
    c_region VARCHAR,
    s_nation VARCHAR,
    s_region VARCHAR
) WITH (adopt_parquets_at = '{source}');
