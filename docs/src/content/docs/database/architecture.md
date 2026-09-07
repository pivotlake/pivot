---
title: Architecture
description: How Pivot separates metadata, storage, planning, and execution.
sidebar:
  order: 1
---

## System architecture

Pivot separates query execution, data access, and server metadata into distinct layers. This allows the same execution engine to operate with different data sources (such as Iceberg and Delta Lake) and deployment models (such as a local shell or a server cluster).

<figure class="arch-figure">
<svg viewBox="0 0 920 492" role="img" aria-labelledby="map-title map-desc">
<title id="map-title">Pivot architecture</title>
<desc id="map-desc">Inside Pivot, the catalog holds a snapshot per query, the planner binds and prunes against it, and the dispatch pool runs the scan on worker threads. The metastore supplies configuration and identity to Pivot; the datastore is the transactional data source Pivot scans and commits to, layered as a Pivot manifest, a Delta log, and Parquet files.</desc>
<defs>
<marker id="map-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<g transform="translate(0 -156)">
<rect x="40" y="180" width="840" height="200" rx="3" class="arch-panel" />
<text x="64" y="210" class="arch-title">Pivot</text>
<rect x="64" y="224" width="240" height="104" rx="2" class="arch-inner" />
<rect x="65" y="225" width="238" height="23" class="arch-strip" />
<line x1="65" y1="248" x2="303" y2="248" class="arch-rule" />
<text x="76" y="241" class="arch-label">Catalog</text>
<text x="292" y="241" text-anchor="end" class="arch-tiny">in memory</text>
<text x="76" y="270" class="arch-tiny">datastores by name</text>
<text x="76" y="288" class="arch-tiny">snapshot per query</text>
<rect x="340" y="224" width="240" height="104" rx="2" class="arch-inner" />
<rect x="341" y="225" width="238" height="23" class="arch-strip" />
<line x1="341" y1="248" x2="579" y2="248" class="arch-rule" />
<text x="352" y="241" class="arch-label">Planner</text>
<text x="568" y="241" text-anchor="end" class="arch-tiny">per query</text>
<text x="352" y="270" class="arch-tiny">bind tables from snapshot</text>
<text x="352" y="288" class="arch-tiny">prune + push down</text>
<rect x="616" y="224" width="240" height="104" rx="2" class="arch-inner" />
<rect x="617" y="225" width="238" height="23" class="arch-strip" />
<line x1="617" y1="248" x2="855" y2="248" class="arch-rule" />
<text x="628" y="241" class="arch-label">Dispatch pool</text>
<text x="844" y="241" text-anchor="end" class="arch-tiny">worker threads</text>
<rect x="628" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="650" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="672" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="694" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="716" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="738" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="760" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="782" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="804" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="826" y="262" width="16" height="16" rx="1" class="arch-cell" />
<rect x="628" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="650" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="672" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="694" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="716" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="738" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="760" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="782" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="804" y="284" width="16" height="16" rx="1" class="arch-cell" />
<rect x="826" y="284" width="16" height="16" rx="1" class="arch-cell" />
<text x="628" y="316" class="arch-tiny">parallel scan + compute</text>
<line x1="242" y1="452" x2="242" y2="384" stroke-dasharray="4 5" class="arch-line" marker-end="url(#map-head)" />
<text x="256" y="422" class="arch-muted">configuration + identity</text>
<line x1="678" y1="452" x2="678" y2="384" class="arch-line" marker-start="url(#map-head)" marker-end="url(#map-head)" />
<text x="692" y="422" class="arch-muted">scan + commit</text>
<rect x="40" y="452" width="404" height="172" rx="3" class="arch-panel" />
<text x="64" y="482" class="arch-title">Metastore</text>
<text x="420" y="482" text-anchor="end" class="arch-muted">control plane</text>
<text x="64" y="502" class="arch-muted">YAML · PostgreSQL upcoming</text>
<rect x="64" y="514" width="356" height="26" rx="2" class="arch-inner" />
<text x="76" y="531" class="arch-tiny">datastores</text>
<text x="408" y="531" text-anchor="end" class="arch-tiny">locations + default</text>
<rect x="64" y="546" width="356" height="26" rx="2" class="arch-inner" />
<text x="76" y="563" class="arch-tiny">users</text>
<text x="408" y="563" text-anchor="end" class="arch-tiny">trust · scram-sha-256</text>
<rect x="64" y="578" width="356" height="26" rx="2" class="arch-inner" />
<text x="76" y="595" class="arch-tiny">credentials</text>
<text x="408" y="595" text-anchor="end" class="arch-tiny">object storage keys</text>
<rect x="476" y="452" width="404" height="172" rx="3" class="arch-panel" />
<text x="500" y="482" class="arch-title">Datastore</text>
<text x="856" y="482" text-anchor="end" class="arch-muted">data plane</text>
<text x="500" y="502" class="arch-muted">delta · file:// · s3:// · gs://</text>
<rect x="500" y="514" width="356" height="26" rx="2" class="arch-inner" />
<text x="512" y="531" class="arch-tiny">pivot manifest</text>
<text x="844" y="531" text-anchor="end" class="arch-tiny">schemas + tables</text>
<rect x="500" y="546" width="356" height="26" rx="2" class="arch-inner" />
<text x="512" y="563" class="arch-tiny">delta log</text>
<text x="844" y="563" text-anchor="end" class="arch-tiny">versions + active files</text>
<rect x="500" y="578" width="356" height="26" rx="2" class="arch-inner" />
<text x="512" y="595" class="arch-tiny">parquet files</text>
<text x="844" y="595" text-anchor="end" class="arch-tiny">columns + statistics</text>
</g>
</svg>
</figure>

These responsibilities are divided across five core components:
- [Dispatch execution pool](#dispatch-execution-pool) - Executes physical plans across a pool of workers responsible for computation and I/O.
- [Planner](#planner) - Translates SQL queries into optimized physical execution plans.
- [Catalog](#catalog) - Connects the planner to named datastores and maintains the datastore snapshots used by each query.
- [Datastore](#datastore) - Manages schemas, tables, table versions, and the underlying data files.
- [Metastore](#metastore) - Manages server-level configuration, including users, authentication, and datastore connections.

### Dispatch execution pool

The dispatch execution engine is responsible for receiving a query’s physical execution plan and reliably executing it across a set of dispatch workers: a collection of workers, one per CPU core, that perform the computation, networking, and I/O required to execute a query.

<figure class="arch-figure">
<svg viewBox="0 0 920 352" role="img" aria-labelledby="dispatch-pool-title dispatch-pool-desc">
<title id="dispatch-pool-title">A physical execution plan runs across the dispatch worker pool</title>
<desc id="dispatch-pool-desc">The query's physical execution plan is dispatched to a pool with one worker per CPU core. Workers 1, 2, and N each perform computation, networking, and other I/O.</desc>
<defs>
<marker id="dispatch-pool-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="320" y="16" width="280" height="64" rx="3" class="arch-panel" />
<text x="460" y="43" text-anchor="middle" class="arch-title">Physical execution plan</text>
<text x="460" y="65" text-anchor="middle" class="arch-muted">operators + data flow</text>
<rect x="40" y="116" width="840" height="220" rx="3" class="arch-panel" />
<text x="64" y="147" class="arch-title">Dispatch pool</text>
<text x="856" y="147" text-anchor="end" class="arch-muted">one worker per CPU core</text>
<path d="M460,80 V177 M187,177 H733" class="arch-line" />
<path d="M187,177 V199" class="arch-line" marker-end="url(#dispatch-pool-arrow)" />
<path d="M460,177 V199" class="arch-line" marker-end="url(#dispatch-pool-arrow)" />
<path d="M733,177 V199" class="arch-line" marker-end="url(#dispatch-pool-arrow)" />
<rect x="64" y="202" width="246" height="110" rx="2" class="arch-inner" />
<text x="80" y="227" class="arch-title">Worker 1</text>
<text x="294" y="227" text-anchor="end" class="arch-muted">CPU core 1</text>
<line x1="64" y1="240" x2="310" y2="240" class="arch-rule" />
<text x="80" y="266" class="arch-label">Computation</text>
<text x="80" y="291" class="arch-label">Networking + other I/O</text>
<rect x="337" y="202" width="246" height="110" rx="2" class="arch-inner" />
<text x="353" y="227" class="arch-title">Worker 2</text>
<text x="567" y="227" text-anchor="end" class="arch-muted">CPU core 2</text>
<line x1="337" y1="240" x2="583" y2="240" class="arch-rule" />
<text x="353" y="266" class="arch-label">Computation</text>
<text x="353" y="291" class="arch-label">Networking + other I/O</text>
<text x="596.5" y="264" text-anchor="middle" class="arch-label">…</text>
<rect x="610" y="202" width="246" height="110" rx="2" class="arch-inner" />
<text x="626" y="227" class="arch-title">Worker N</text>
<text x="840" y="227" text-anchor="end" class="arch-muted">CPU core N</text>
<line x1="610" y1="240" x2="856" y2="240" class="arch-rule" />
<text x="626" y="266" class="arch-label">Computation</text>
<text x="626" y="291" class="arch-label">Networking + other I/O</text>
</svg>
</figure>

Once a query has been "dispatched" to the pool, each CPU worker is in charge of executing its share of the physical plan and scheduling the operations within it so that the query runs as efficiently as possible. For example, if a worker has just emitted an output that is hot in the CPU's cache, that worker will prefer to run the next operator in line over another unrelated operator that works on data which is not cache-resident. Conversely, when multiple queries are running at the same time, workers split the available CPU resources between all running queries in a "smart" way, striking a balance between cache efficiency and making sure that one query doesn't starve the other.

Beyond CPU work, the dispatch worker also handles IO requests and prioritizes between them. For example, a "Materialize" operator that enriches existing data with additional columns, takes precedence over the input operator feeding it, to prevent a scenario where the channel between the two fills up and takes all available memory.

<figure class="arch-figure" style="max-width: 620px;">
<svg viewBox="0 0 620 104" role="img" aria-labelledby="materialize-priority-title materialize-priority-desc">
<title id="materialize-priority-title">Prioritizing materialization drains buffered input</title>
<desc id="materialize-priority-desc">Input reads produce record batches that wait in a channel for materialization to fetch additional columns. Prioritizing materialization drains the channel and releases memory held by waiting batches.</desc>
<defs>
<marker id="materialize-priority-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="16" y="16" width="132" height="72" rx="3" class="arch-panel" />
<text x="82" y="44" text-anchor="middle" class="arch-title">Input</text>
<text x="82" y="69" text-anchor="middle" class="arch-muted">reads new data</text>
<path d="M151,52 H211" class="arch-line" marker-end="url(#materialize-priority-arrow)" />
<rect x="216" y="16" width="160" height="72" rx="3" class="arch-inner" />
<text x="296" y="41" text-anchor="middle" class="arch-label">Waiting batches</text>
<rect x="258" y="55" width="20" height="16" rx="1" class="arch-cell" />
<rect x="286" y="55" width="20" height="16" rx="1" class="arch-cell" />
<rect x="314" y="55" width="20" height="16" rx="1" class="arch-cell" />
<path d="M379,52 H439" class="arch-line" marker-end="url(#materialize-priority-arrow)" />
<rect x="444" y="16" width="160" height="72" rx="3" class="arch-panel" />
<text x="524" y="44" text-anchor="middle" class="arch-title">Materialize</text>
<text x="524" y="69" text-anchor="middle" class="arch-muted">fetch columns</text>
</svg>
</figure>

<span id="dispatch"></span>

Dispatch splits the surviving row groups across worker threads. A single row
group can be split across workers when its decode is expensive enough to be
worth sharing.

### Planner

Pivot's planner uses a fork of DuckDB to parse SQL, resolve table and column references, and optimize the logical plan. Pivot then translates that plan into its own execution operators.

A query passes through four stages:

1. Parsing - DuckDB's PostgreSQL-derived parser checks the SQL syntax and builds an abstract syntax tree (AST) representing the statement's expressions and clauses, such as projections, filters, joins, and ordering.
2. Binding and logical planning - the binder resolves table and column references against the query's catalog snapshot, resolves aliases and function calls, and checks expression types. It produces a logical plan describing the operations needed to answer the query.
3. Logical optimization - the optimizer rewrites the plan to reduce unnecessary work. This includes evaluating constant expressions in advance, pushing filters closer to table scans, removing unused columns, choosing join order using available row-count estimates, and more.
4. Physical translation and compilation - Pivot converts the optimized logical plan into its own operators and expressions, then applies additional refinements, such as pushing eligible limits into grouped aggregations. It selects execution implementations for scans, joins, and aggregations and connects the operators into a parallel dataflow ready to run on the dispatch worker pool.

<figure class="arch-figure">
<svg viewBox="0 0 920 290" role="img" aria-labelledby="planner-flow-title planner-flow-desc">
<title id="planner-flow-title">From SQL to a Pivot execution plan</title>
<desc id="planner-flow-desc">SQL passes through four stages in the query planner: parsing into an AST, binding and logical planning using the catalog snapshot, logical optimization, and physical translation and compilation into a dataflow for the dispatch pool. The dashed arrow supplies catalog metadata to binding.</desc>
<defs>
<marker id="planner-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="132" y="24" width="748" height="164" rx="3" class="arch-panel" />
<text x="154" y="53" class="arch-title">Query planner</text>
<rect x="40" y="96" width="72" height="52" rx="3" class="arch-panel" />
<text x="76" y="127" text-anchor="middle" class="arch-title">SQL</text>
<path d="M114,122 H150" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<rect x="154" y="76" width="160" height="88" rx="2" class="arch-inner" />
<text x="234" y="103" text-anchor="middle" class="arch-label">1. Parse</text>
<text x="234" y="137" text-anchor="middle" class="arch-muted">AST</text>
<path d="M316,122 H332" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<rect x="336" y="76" width="160" height="88" rx="2" class="arch-inner" />
<text x="416" y="103" text-anchor="middle" class="arch-label">2. Bind + plan</text>
<text x="416" y="137" text-anchor="middle" class="arch-muted">Logical plan</text>
<path d="M498,122 H514" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<rect x="518" y="76" width="160" height="88" rx="2" class="arch-inner" />
<text x="598" y="103" text-anchor="middle" class="arch-label">3. Optimize</text>
<text x="598" y="137" text-anchor="middle" class="arch-muted">Optimized plan</text>
<path d="M680,122 H696" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<rect x="700" y="76" width="160" height="88" rx="2" class="arch-inner" />
<text x="780" y="99" text-anchor="middle" class="arch-label">4. Translate</text>
<text x="780" y="119" text-anchor="middle" class="arch-label">and compile</text>
<text x="780" y="145" text-anchor="middle" class="arch-muted">Dataflow</text>
<rect x="336" y="220" width="160" height="52" rx="3" class="arch-panel" />
<text x="416" y="243" text-anchor="middle" class="arch-label">Catalog snapshot</text>
<text x="416" y="261" text-anchor="middle" class="arch-tiny">schemas + tables</text>
<path d="M416,220 V168" stroke-dasharray="4 5" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<path d="M780,164 V216" class="arch-line" marker-end="url(#planner-flow-arrow)" />
<rect x="700" y="220" width="160" height="52" rx="3" class="arch-panel" />
<text x="780" y="251" text-anchor="middle" class="arch-title">Dispatch pool</text>
</svg>
</figure>

<span id="planning"></span>

The planner resolves a statement against the query's catalog snapshot and
pushes projections and predicates into the scan. Partition values, file
statistics, row-group statistics, and dictionaries can eliminate work before
any column data is decoded.

### Catalog

The catalog is Pivot's in-memory bridge between the [planner](#planner) and the
[datastores](#datastore).
It keeps track of the datastores available on a server and routes table lookups
to the right one. Through this interface, the planner discovers schemas and
tables, resolves column names and types, and obtains metadata such as row-count
estimates without needing to understand how each datastore persists its data.

Each query opens a catalog transaction. The first time it accesses a datastore,
the transaction captures that datastore's snapshot and reuses it for the rest
of the query. Planning, scans, and late materialization therefore agree on the
same table versions and file sets, even if a background refresh discovers newer
data while the query is running. Snapshots are taken independently for each
datastore, not as one atomic snapshot across all datastores.

The catalog also exposes a read-only "virtual" [system tables datastore](/docs/reference/system-tables/), so users can query metadata about the current running pivot instance.

### Datastore

A datastore is a collection of schemas and tables exposed to Pivot through a common interface. Each configured datastore implementation handles table discovery, metadata, snapshots, and supported read and write operations for its underlying storage or table format.

Pivot currently supports/exposes the following datastores:
* Pivot — A collection of Delta Lake tables.
* Iceberg — Tables stored using the Apache Iceberg table format.
* System — Internal tables exposing information about the running Pivot instance.

A server can expose multiple datastores, with tables addressed as `datastore.schema.table`.

For example, a query can join orders in a Pivot datastore named `analytics` with customer
details in an Iceberg datastore named `lake`:

```sql
SELECT
    orders.order_id,
    customers.customer_name,
    orders.total_amount
FROM analytics.sales.orders AS orders
JOIN lake.crm.customers AS customers
    ON orders.customer_id = customers.customer_id;
```

Here, `analytics` and `lake` are datastore names, while `sales` and `crm` are
schemas within those datastores. Pivot reads each table through its
datastore and executes the join in the same query engine.

### Metastore

The metastore is a collective of configurations and secrets that declare the metadata a pivot instance needs for it to run.

These include:
- Datastores — their names, implementations, locations, and settings, including which datastore is the default for unqualified table names.
- Secrets — credentials for accessing object storage, scoped to the locations they apply to. Multiple datastores can use the same secret.
- Users — which users are authored to access pivot, and how do they authenticate.

Thanks to the separation of datastores and metastore, different instances / deployments of pivot might access different datastores in different permission models and with different configurations:

<figure class="arch-figure">
<svg viewBox="0 0 920 514" role="img" aria-labelledby="deployments-title deployments-desc">
<title id="deployments-title">Two Pivot instances with different metastores share one datastore</title>
<desc id="deployments-desc">Object storage holds two datastores: analytics, a Pivot datastore, and lake, an Iceberg datastore. A Pivot shell started with pivot open on the analytics location uses an ephemeral metastore and reads and writes only analytics. A Pivot server started with pivot server and a metastore from pivot.yaml reads and writes analytics, reads lake, and serves SQL clients over the Postgres wire, where the analyst user logs in with a password.</desc>
<defs>
<marker id="deployments-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="16" width="840" height="156" rx="3" class="arch-panel" />
<text x="64" y="46" class="arch-title">Object storage</text>
<text x="856" y="46" text-anchor="end" class="arch-muted">S3 · GCS</text>
<text x="64" y="66" class="arch-muted">datastores</text>
<rect x="64" y="80" width="392" height="76" rx="3" class="arch-inner" />
<rect x="65" y="81" width="390" height="24" class="arch-strip" />
<line x1="65" y1="105" x2="455" y2="105" class="arch-rule" />
<text x="76" y="98" class="arch-label">analytics</text>
<text x="444" y="98" text-anchor="end" class="arch-tiny">Pivot · Delta Lake format</text>
<text x="76" y="124" class="arch-tiny">s3://company-data/pivot/</text>
<text x="76" y="142" class="arch-tiny">pivot manifest + delta log + parquet</text>
<rect x="480" y="80" width="376" height="76" rx="3" class="arch-inner" />
<rect x="481" y="81" width="374" height="24" class="arch-strip" />
<line x1="481" y1="105" x2="855" y2="105" class="arch-rule" />
<text x="492" y="98" class="arch-label">lake</text>
<text x="844" y="98" text-anchor="end" class="arch-tiny">Iceberg</text>
<text x="492" y="124" class="arch-tiny">s3://lakehouse/warehouse/</text>
<text x="492" y="142" class="arch-tiny">metadata + manifests + parquet</text>
<line x1="182" y1="258" x2="182" y2="158" class="arch-line" marker-end="url(#deployments-arrow)" />
<text x="196" y="210" class="arch-muted">read + write</text>
<path d="M524,258 V222 M380,222 H668" class="arch-line" />
<path d="M380,222 V160" class="arch-line" marker-end="url(#deployments-arrow)" />
<text x="394" y="210" class="arch-muted">read + write</text>
<path d="M668,222 V160" class="arch-line" marker-end="url(#deployments-arrow)" />
<text x="682" y="210" class="arch-muted">read</text>
<rect x="40" y="260" width="310" height="92" rx="3" class="arch-panel" />
<text x="64" y="290" class="arch-title">Pivot shell</text>
<rect x="226" y="271" width="100" height="26" rx="2" class="arch-inner" />
<rect x="235" y="277" width="14" height="14" rx="1" class="arch-cell" />
<text x="255" y="289" class="arch-tiny">metastore</text>
<rect x="64" y="306" width="266" height="26" rx="2" class="arch-inner" />
<text x="76" y="323" class="arch-prompt">$</text>
<text x="88" y="323" class="arch-cmd">pivot open s3://company-data/pivot/</text>
<rect x="380" y="260" width="500" height="92" rx="3" class="arch-panel" />
<text x="404" y="290" class="arch-title">Pivot server</text>
<rect x="756" y="271" width="100" height="26" rx="2" class="arch-inner" />
<rect x="765" y="277" width="14" height="14" rx="1" class="arch-cell" />
<text x="785" y="289" class="arch-tiny">metastore</text>
<rect x="404" y="306" width="452" height="26" rx="2" class="arch-inner" />
<text x="416" y="323" class="arch-prompt">$</text>
<text x="428" y="323" class="arch-cmd">pivot server --config pivot.yaml</text>
<line x1="524" y1="408" x2="524" y2="354" class="arch-line" marker-end="url(#deployments-arrow)" />
<text x="538" y="384" class="arch-muted">Postgres wire</text>
<rect x="380" y="408" width="500" height="90" rx="3" class="arch-panel" />
<text x="404" y="432" class="arch-title">SQL clients</text>
<text x="856" y="432" text-anchor="end" class="arch-muted">backend · psql · BI tools</text>
<rect x="404" y="442" width="452" height="44" rx="2" class="arch-inner" />
<text x="416" y="459" class="arch-prompt">$</text>
<text x="428" y="459" class="arch-cmd">psql -h pivot.internal -U analyst</text>
<text x="416" y="477" class="arch-cmd">Password for user analyst: ********</text>
</svg>
</figure>

In this example, both instances open the same Pivot datastore. The server loads its metastore from a YAML file: two datastores, an analyst user who logs in with a password, and an S3 secret scoped to the datastore location. The shell, started with `pivot open`, builds an ephemeral metastore from its command line: a single datastore, credentials read from the environment, and no user login. Only the server also reads the Iceberg datastore.

##### Sharing a metastore across a cluster of instances

> **Upcoming:** PostgreSQL-backed metastores are under development and are not
> available in the current release. The configuration may change before the
> feature is merged.

A PostgreSQL metastore lets several Pivot servers use the same datastore
registry and user directory. The analytical data still lives in the configured
datastore, typically S3, not in PostgreSQL. Each Pivot server executes queries on
its own worker pool, so this creates a load-balanced cluster rather than one
distributed query spanning several servers.

Every server in the cluster points at the same metastore database:

```yaml title="node-a.yaml"
server:
  bind: 0.0.0.0:5432

metastore:
  kind: postgres
  url: postgres://pivot:secret@pg.internal:5432/pivot_metastore
  compact: true
```

The other nodes use the same `metastore.url`, choose their own `server.bind`,
and leave `compact` disabled. At most one server should enable compaction for a
shared datastore.

On first connection, Pivot creates the `pivot_metastore` schema and its tables.
The cluster will begin serving after the metastore contains exactly one default
datastore. For example:

```sql
INSERT INTO pivot_metastore.datastores
  (name, kind, location, is_default, compact,
   region, access_key_id, secret_access_key)
VALUES
  ('analytics', 'delta', 's3://company-data/pivot/', true, true,
   'us-east-1', 'PIVOT_ACCESS_KEY', 'PIVOT_SECRET_KEY');

INSERT INTO pivot_metastore.users
  (name, auth_method, scram_verifier)
VALUES
  ('analyst', 'scram-sha-256', 'pivot-scram-sha-256$4096:...$...');
```

Datastore definitions are loaded when a Pivot server starts, so changing one
requires restarting the nodes. User records are checked on each login, allowing
new users and credential rotations to apply across the cluster without a Pivot
restart. The metastore URL and stored credentials should be accessible only to
the Pivot servers and their operators.

## Deployment Architecture
