-- TPC-H "flat" schema: the eight TPC-H tables denormalised into one
-- lineitem-grain table (one row per lineitem, carrying its order / customer /
-- supplier / part / partsupp attributes, plus the supplier's and the
-- customer's nation+region). The suite's queries are the TPC-H questions that
-- can be answered by a single scan of this table - no joins, no subqueries.
--
-- Built by prep-tpch-flat-data.sh. Column order matches that script's COPY so
-- the parquet binds positionally. Monetary / quantity columns are DOUBLE (see
-- the prep script for why). The runner substitutes `{source}` with --source.
CREATE TABLE lineitem_flat (
    l_orderkey BIGINT,
    l_partkey INTEGER,
    l_suppkey INTEGER,
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
    o_orderstatus VARCHAR,
    o_totalprice DOUBLE,
    o_orderdate DATE,
    o_orderpriority VARCHAR,
    o_clerk VARCHAR,
    o_shippriority INTEGER,
    c_custkey INTEGER,
    c_name VARCHAR,
    c_address VARCHAR,
    c_phone VARCHAR,
    c_acctbal DOUBLE,
    c_mktsegment VARCHAR,
    c_nation VARCHAR,
    c_region VARCHAR,
    s_suppkey INTEGER,
    s_name VARCHAR,
    s_address VARCHAR,
    s_phone VARCHAR,
    s_acctbal DOUBLE,
    s_nation VARCHAR,
    s_region VARCHAR,
    p_name VARCHAR,
    p_mfgr VARCHAR,
    p_brand VARCHAR,
    p_type VARCHAR,
    p_size INTEGER,
    p_container VARCHAR,
    p_retailprice DOUBLE,
    ps_supplycost DOUBLE
) WITH (path = '{source}');
