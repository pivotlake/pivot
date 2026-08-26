-- The 5 Star Schema Benchmark tables: the lineorder fact table and its four
-- dimensions, as prep-ssb-data.sh writes them (ssb-dbgen .tbl converted to
-- parquet). BIGINT keys, INTEGER measures (SSB money values are integers),
-- INTEGER yyyymmdd date keys, VARCHAR text. Each table is a directory of
-- parquet files ({source}/<table>/*.parquet).
--
-- The runner substitutes `{source}` with the value of `--source`.
CREATE TABLE lineorder (
    lo_orderkey BIGINT,
    lo_linenumber INTEGER,
    lo_custkey BIGINT,
    lo_partkey BIGINT,
    lo_suppkey BIGINT,
    lo_orderdate INTEGER,
    lo_orderpriority VARCHAR,
    lo_shippriority VARCHAR,
    lo_quantity INTEGER,
    lo_extendedprice INTEGER,
    lo_ordtotalprice INTEGER,
    lo_discount INTEGER,
    lo_revenue INTEGER,
    lo_supplycost INTEGER,
    lo_tax INTEGER,
    lo_commitdate INTEGER,
    lo_shipmode VARCHAR
) WITH (with_pre_existing_parquets = '{source}/lineorder');

CREATE TABLE customer (
    c_custkey BIGINT,
    c_name VARCHAR,
    c_address VARCHAR,
    c_city VARCHAR,
    c_nation VARCHAR,
    c_region VARCHAR,
    c_phone VARCHAR,
    c_mktsegment VARCHAR
) WITH (with_pre_existing_parquets = '{source}/customer');

CREATE TABLE supplier (
    s_suppkey BIGINT,
    s_name VARCHAR,
    s_address VARCHAR,
    s_city VARCHAR,
    s_nation VARCHAR,
    s_region VARCHAR,
    s_phone VARCHAR
) WITH (with_pre_existing_parquets = '{source}/supplier');

CREATE TABLE part (
    p_partkey BIGINT,
    p_name VARCHAR,
    p_mfgr VARCHAR,
    p_category VARCHAR,
    p_brand1 VARCHAR,
    p_color VARCHAR,
    p_type VARCHAR,
    p_size INTEGER,
    p_container VARCHAR
) WITH (with_pre_existing_parquets = '{source}/part');

CREATE TABLE date (
    d_datekey INTEGER,
    d_date VARCHAR,
    d_dayofweek VARCHAR,
    d_month VARCHAR,
    d_year INTEGER,
    d_yearmonthnum INTEGER,
    d_yearmonth VARCHAR,
    d_daynuminweek INTEGER,
    d_daynuminmonth INTEGER,
    d_daynuminyear INTEGER,
    d_monthnuminyear INTEGER,
    d_weeknuminyear INTEGER,
    d_sellingseason VARCHAR,
    d_lastdayinweekfl INTEGER,
    d_lastdayinmonthfl INTEGER,
    d_holidayfl INTEGER,
    d_weekdayfl INTEGER
) WITH (with_pre_existing_parquets = '{source}/date');
