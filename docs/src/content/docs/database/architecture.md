---
title: Architecture
description: How Pivot separates metadata, storage, planning, and execution.
sidebar:
  order: 1
---

Pivot is split into multiple components that operate and communicate with each other -- together forming the complete architecture of the system.

The first and most core component of the engine is the Dispatch execution pool:


## Dispatch execution pool
<architecture>

THe dispatch execution engine is responsible to receive a physical execution plan of a query and execute it reliably on a set of "dispatch workers": a collective of workers (one worker per core) that perform the computational and netwokring operations required to complete the execution of a query.

After "dispatching" a query to the dispatch pool, each CPU worker is in charge of executing the physical plan, and scheling different operations in the plan so the query operates in the most efficiant way. For example, if worker A has just outputted a recordbatch that is hot in ran, the worker will prefer to execute the next operator in line B vs sanother operator "c" that works on data that is not hot in cache. To the contrary, if there are multiple queries running in the same time, the worker might split the available CPU resources the two queries in a "smart" way, trying to strike a balance between CPU cache efficiency while not letting one query "starve" the other.

In addition to running the CPU work, the dispatch worker also handles IO requests and attemps to prioritize IO etween them (a IO request from a "Materialize" operator that can free up memory quickly might be prioritized over an IO request to fetch new data that will cause memory conumtion to only go up).

Pivot separates the metadata required to operate a server from the data and table metadata required to execute queries. This allows a local CLI session to access the same data and tables as a cluster serving clients, while maintaining separate authentication and server configurations.

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
<text x="856" y="210" text-anchor="end" class="arch-muted">one process · one worker pool</text>
<rect x="64" y="224" width="240" height="104" rx="2" class="arch-inner" />
<rect x="65" y="225" width="238" height="23" class="arch-strip" />
<line x1="65" y1="248" x2="303" y2="248" class="arch-rule" />
<text x="76" y="241" class="arch-label">Catalog</text>
<text x="292" y="241" text-anchor="end" class="arch-tiny">in memory</text>
<text x="76" y="270" class="arch-tiny">datastores by name</text>
<text x="76" y="288" class="arch-tiny">snapshot per query</text>
<line x1="306" y1="276" x2="337" y2="276" class="arch-line" marker-end="url(#map-head)" />
<rect x="340" y="224" width="240" height="104" rx="2" class="arch-inner" />
<rect x="341" y="225" width="238" height="23" class="arch-strip" />
<line x1="341" y1="248" x2="579" y2="248" class="arch-rule" />
<text x="352" y="241" class="arch-label">Planner</text>
<text x="568" y="241" text-anchor="end" class="arch-tiny">per query</text>
<text x="352" y="270" class="arch-tiny">bind tables from snapshot</text>
<text x="352" y="288" class="arch-tiny">prune + push down</text>
<line x1="582" y1="276" x2="613" y2="276" class="arch-line" marker-end="url(#map-head)" />
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

### Metastore

The metastore is the server's configuration and identity layer. It tells Pivot:

- Which named datastores to open and which one is the default
- Where each datastore lives and which credentials can access it
- Which users may connect and how they authenticate

The metastore does **not** sit in the scan path and does not hold every table or
Parquet file. Once the server has opened its datastores, queries resolve table
metadata through the datastore snapshots held by the in-memory catalog.

The standard deployment uses the YAML-backed disk metastore. It works well for
one server: datastore definitions and users live in the server configuration
and optional metastore file.

### Datastore

A datastore is one named, transactional data source. It owns schemas, tables,
table versions, and the files that make up those tables. A server can expose
several datastores, and SQL can address them as
`datastore.schema.table`. Unqualified names use the configured default
datastore.

The Pivot datastore keeps durable state alongside the data:

- A Pivot manifest records the schemas and tables in the datastore.
- Each table's Delta log records its schema, partitioning, versions, and active
  Parquet files.
- Parquet footers provide row-group metadata and statistics used during scans.

Each query opens a consistent snapshot of every datastore it touches. A query
that uses only one datastore does not snapshot the others. Transactions across
different datastores are not atomic as a single unit.

A local datastore is owned by one Pivot process at a time. A remote datastore
on shared object storage can be opened by multiple servers; periodic refreshes
make commits from one server visible to the others. Background compaction and
vacuum should run on only one server for a shared datastore.

### Planning

The planner resolves a statement against the query's catalog snapshot and
pushes projections and predicates into the scan. Partition values, file
statistics, row-group statistics, and dictionaries can eliminate work before
any column data is decoded.

### Dispatch

Dispatch splits the surviving row groups across worker threads. A single row
group can be split across workers when its decode is expensive enough to be
worth sharing.

### Sharing a metastore across a cluster

> **Upcoming:** PostgreSQL-backed metastores are under development and are not
> available in the current release. The configuration may change before the
> feature is merged.

A PostgreSQL metastore lets several Pivot servers use the same datastore
registry and user directory. The analytical data still lives in the configured
datastore—typically S3—not in PostgreSQL. Each Pivot server executes queries on
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
