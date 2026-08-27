---
title: Architecture
description: How Pivot separates metadata, storage, planning, and execution.
sidebar:
  order: 1
---

Pivot separates the metadata needed to operate a server from the data and table
metadata needed to run queries. The distinction makes the same execution engine
usable for a local database, object storage, or multiple servers sharing one
storage layer.

<figure class="arch-figure">
<svg viewBox="0 0 920 532" role="img" aria-labelledby="detail-arch-title detail-arch-desc">
<title id="detail-arch-title">Pivot server architecture</title>
<desc id="detail-arch-desc">SQL clients connect to a Pivot server. The server resolves queries through its catalog and planner, then executes them on the dispatch pool against a datastore. A separate metastore supplies datastore definitions, users, and credentials when the server starts.</desc>
<defs>
<marker id="detail-arch-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>

<rect x="40" y="24" width="190" height="108" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">SQL clients</text>
<rect x="64" y="72" width="42" height="32" rx="2" class="arch-inner" />
<text x="85" y="92" text-anchor="middle" class="arch-tiny">psql</text>
<rect x="116" y="72" width="38" height="32" rx="2" class="arch-inner" />
<text x="135" y="92" text-anchor="middle" class="arch-tiny">BI</text>
<rect x="164" y="72" width="42" height="32" rx="2" class="arch-inner" />
<text x="185" y="92" text-anchor="middle" class="arch-tiny">apps</text>

<line x1="232" y1="78" x2="278" y2="78" class="arch-line" marker-end="url(#detail-arch-head)" />
<text x="255" y="66" text-anchor="middle" class="arch-tiny">SQL</text>

<rect x="280" y="24" width="600" height="238" rx="3" class="arch-panel" />
<text x="304" y="54" class="arch-title">Pivot server</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">one process · one worker pool</text>

<rect x="304" y="70" width="552" height="38" rx="2" class="arch-inner" />
<text x="320" y="94" class="arch-label">Postgres wire</text>
<text x="840" y="94" text-anchor="end" class="arch-tiny">authentication · sessions · results</text>
<line x1="580" y1="110" x2="580" y2="130" class="arch-line" marker-end="url(#detail-arch-head)" />

<rect x="304" y="132" width="160" height="94" rx="2" class="arch-inner" />
<rect x="305" y="133" width="158" height="24" class="arch-strip" />
<line x1="305" y1="157" x2="463" y2="157" class="arch-rule" />
<text x="320" y="150" class="arch-label">Catalog</text>
<text x="320" y="180" class="arch-tiny">named datastores</text>
<text x="320" y="198" class="arch-tiny">query snapshots</text>
<text x="320" y="216" class="arch-tiny">table bindings</text>

<line x1="466" y1="179" x2="486" y2="179" class="arch-line" marker-end="url(#detail-arch-head)" />

<rect x="488" y="132" width="160" height="94" rx="2" class="arch-inner" />
<rect x="489" y="133" width="158" height="24" class="arch-strip" />
<line x1="489" y1="157" x2="647" y2="157" class="arch-rule" />
<text x="504" y="150" class="arch-label">Planner</text>
<text x="504" y="180" class="arch-tiny">resolve + optimize</text>
<text x="504" y="198" class="arch-tiny">filter pushdown</text>
<text x="504" y="216" class="arch-tiny">projection pruning</text>

<line x1="650" y1="179" x2="670" y2="179" class="arch-line" marker-end="url(#detail-arch-head)" />

<rect x="672" y="132" width="184" height="94" rx="2" class="arch-inner" />
<rect x="673" y="133" width="182" height="24" class="arch-strip" />
<line x1="673" y1="157" x2="855" y2="157" class="arch-rule" />
<text x="688" y="150" class="arch-label">Dispatch pool</text>
<rect x="688" y="172" width="22" height="16" rx="1" class="arch-cell" />
<rect x="716" y="172" width="22" height="16" rx="1" class="arch-cell" />
<rect x="744" y="172" width="22" height="16" rx="1" class="arch-cell" />
<rect x="772" y="172" width="22" height="16" rx="1" class="arch-cell" />
<rect x="800" y="172" width="22" height="16" rx="1" class="arch-cell" />
<rect x="828" y="172" width="12" height="16" rx="1" class="arch-cell" />
<text x="688" y="211" class="arch-tiny">scan · filter · group · sort</text>

<path d="M384 226 V286 H175 V320" class="arch-line" marker-start="url(#detail-arch-head)" />
<text x="196" y="278" class="arch-muted">opens + configures</text>

<path d="M764 226 V320" class="arch-line" marker-end="url(#detail-arch-head)" />
<text x="778" y="278" class="arch-muted">scan + commit</text>

<rect x="40" y="322" width="270" height="174" rx="3" class="arch-panel" />
<text x="64" y="352" class="arch-title">Metastore</text>
<text x="286" y="352" text-anchor="end" class="arch-muted">Disk · PostgreSQL upcoming</text>
<rect x="64" y="370" width="222" height="30" rx="2" class="arch-inner" />
<text x="78" y="390" class="arch-tiny">datastores + default</text>
<rect x="64" y="408" width="222" height="30" rx="2" class="arch-inner" />
<text x="78" y="428" class="arch-tiny">users + authentication</text>
<rect x="64" y="446" width="222" height="30" rx="2" class="arch-inner" />
<text x="78" y="466" class="arch-tiny">storage credentials</text>

<rect x="360" y="322" width="520" height="174" rx="3" class="arch-panel" />
<text x="384" y="352" class="arch-title">Datastore: analytics</text>
<text x="856" y="352" text-anchor="end" class="arch-muted">file:// · s3:// · gs://</text>

<rect x="384" y="370" width="136" height="106" rx="2" class="arch-inner" />
<rect x="385" y="371" width="134" height="24" class="arch-strip" />
<line x1="385" y1="395" x2="519" y2="395" class="arch-rule" />
<text x="400" y="388" class="arch-label">Pivot manifest</text>
<text x="400" y="420" class="arch-tiny">schemas</text>
<text x="400" y="438" class="arch-tiny">tables</text>
<text x="400" y="456" class="arch-tiny">locations</text>

<rect x="536" y="370" width="136" height="106" rx="2" class="arch-inner" />
<rect x="537" y="371" width="134" height="24" class="arch-strip" />
<line x1="537" y1="395" x2="671" y2="395" class="arch-rule" />
<text x="552" y="388" class="arch-label">Delta log</text>
<text x="552" y="420" class="arch-tiny">schema</text>
<text x="552" y="438" class="arch-tiny">versions</text>
<text x="552" y="456" class="arch-tiny">active files</text>

<rect x="688" y="370" width="168" height="106" rx="2" class="arch-inner" />
<rect x="689" y="371" width="166" height="24" class="arch-strip" />
<line x1="689" y1="395" x2="855" y2="395" class="arch-rule" />
<text x="704" y="388" class="arch-label">Parquet</text>
<text x="704" y="420" class="arch-tiny">column data</text>
<text x="704" y="438" class="arch-tiny">row groups</text>
<text x="704" y="456" class="arch-tiny">statistics</text>
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

The Delta datastore keeps durable state alongside the data:

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
