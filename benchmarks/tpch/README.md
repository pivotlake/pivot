# tpch

The real, normalized TPC-H: the 8 base tables as parquet directories and the
official query texts, joins included. (The single-table variant that predates
pivot's join executor lives in `../tpch-flat`.)

## Data

The datasets are pre-generated (tpchgen-cli) and hosted at `s3://epsio-tpch/`:

| Dataset | Path | Size |
|---|---|---|
| SF10 | `s3://epsio-tpch/sf10/` | ~3.9 GB |
| SF100, 7 MiB row groups | `s3://epsio-tpch/sf100/` | ~41.5 GB |
| SF100, 128 MiB row groups | `s3://epsio-tpch/sf100-large-row-groups/` | ~35.8 GB |

```sh
./prep-tpch-data.sh --dataset sf100        # → ~/tpch-sf100
cd .. && cargo run --release -- --suite tpch --source ~/tpch-sf100 --iterations 3
```

Layout: one directory per table (`<root>/lineitem/lineitem.N.parquet`, ...).
Types are tpchgen's parquet output: BIGINT keys, DECIMAL(15,2) money columns,
real DATEs.

## Queries

Official TPC-H query texts (default substitution parameters), added one by one
as the engine grows the features each needs.

| Query | TPC-H | Shape |
|-------|-------|-------|
| q01 | Pricing Summary | no join: group by returnflag/linestatus, 8 aggregates |
| q03 | Shipping Priority | customer/orders/lineitem, top 10 by revenue |
| q04 | Order Priority Checking | a correlated EXISTS decorrelated into a SEMI delim join |
| q05 | Local Supplier Volume | 6 tables, one join on two keys (suppkey and nationkey), revenue per nation |
| q06 | Forecasting Revenue | no join: one row from a filtered scan |
| q07 | Volume Shipping | 6 tables incl. nation twice, an OR over both nations riding the join as a residual condition |
| q08 | National Market Share | 7 joins, `extract(year ...)` groups, share of two sums |
| q09 | Product Type Profit | 6 tables, partsupp joined on two keys (suppkey and partkey), profit per nation per year |
| q11 | Important Stock | partsupp/supplier/nation computed once as a CTE read twice, HAVING above a scalar subquery via a `>` range join |
| q12 | Shipping Modes | orders/lineitem, CASE priority buckets per shipmode |
| q13 | Customer Distribution | customer/orders outer join, orders per customer, then a histogram of those counts |
| q14 | Promotion Effect | lineitem/part, promo share of revenue |
| q15 | Top Supplier | a revenue CTE read twice, equality join against its own MAX via a scalar subquery |
| q17 | Small-Quantity-Order Revenue | a correlated scalar average decorrelated into a LEFT delim join |
| q18 | Large Volume Customer | orders semi-joined against the order keys whose quantities sum above 300, then the top 100 by price |
| q19 | Discounted Revenue | lineitem/part, a three-disjunct OR over both sides riding the join as a residual condition |

The rest of the 22 need engine features that are not in yet: ANTI delim joins
(q21), mark joins (q16), and the `suffix` / `substring` scalar functions
(q02, q16, q22). LEFT and SEMI delim joins (the shapes DuckDB decorrelates
q04 and q17's correlated subqueries into) are supported: the outer side runs
once and is read both by the join and by a distinct on the correlation
columns, whose output feeds the subquery side's delim scans.

q11's HAVING threshold is the spec's `FRACTION = 0.0001 / SF`, written out for
SF100 as `0.000001`; adjust it (and regenerate the oracle) for another scale.
q11 also carries one deviation from the official text: `ps_partkey` as a
second sort key, because at SF100 several part keys sum to the same `value`
and the byte-exact oracle comparison needs a deterministic row order. q15
would emit several rows if suppliers tied on the maximum revenue; none do in
the reference dataset.

Outer joins are supported on both sides: a RIGHT join preserves the side the
hash table is built from, a LEFT join the side that probes it. DuckDB flips a
written LEFT JOIN into RIGHT whenever the preserved relation is the smaller one
(as in q13) and keeps it LEFT when it is the larger; both shapes run.

Semi joins are supported the other way round: the rows kept are the probe
side's, which is what DuckDB hands over as SEMI (that join type keeps its left
child's rows, and the left child is the probe). q18's `IN` subquery arrives in
that shape, since the aggregate it selects from is the cheaper side to build the
hash table from at every scale factor. Its mirror image RIGHT_SEMI, which
DuckDB's build-probe-side optimizer produces when the left child is the cheaper
one instead, is not supported; q20 needs it (its delim join itself is the
supported LEFT shape).

q10 plans and runs, but its `c_comment` group key comes back corrupted at SF100
(fragments of other rows, with the length prefix of a neighbouring field showing
up inside the string), so it is held back until that is fixed. The corruption
needs scale: the same query is correct at SF0.05, and it appears whether or not
the LIMIT reaches the group operator as a Top-K.

q03 orders by a summed revenue and cuts with a LIMIT, so a tie in that sum would
make its row order (and therefore the exact-string comparison) arbitrary. No tie
occurs in the reference datasets. q18 cuts with a LIMIT too, and at SF100 it has
more qualifying orders than the LIMIT keeps; the prices either side of that
boundary differ, so its row order is not ambiguous either.

Oracles (`qNN.tsv`) are produced by DuckDB over the same parquet files
(`run-duckdb.sh --write-expected`), in pivot's wire format, and are specific to
the dataset's scale factor. The committed ones are for SF100; regenerate them
when you run another scale factor:

```sh
./run-duckdb.sh --source ~/tpch-sf10 --query 1,3,6,8,12,14 --write-expected
```

`q01.tsv` is pivot's own rendering of those numbers rather than DuckDB's text:
the two engines sum the `avg` columns in a different order, so their last digit
differs (`38236.11698430489` against `38236.1169843049`) even though the values
agree. Every other committed oracle is DuckDB's output byte for byte.
