---
title: Architecture
description: How a query becomes parallel work over parquet files.
sidebar:
  order: 1
---

Pivot is built to be *fast*, *open*, *scalable* and *portable*.

To achieve this, it utilizes open data formats that sit on object storage, alongside aggresive caching methods and query execution optimizations that alow users to receive the scalability and openess of a data warehouse, with the speed and concurrency of a real time analytics engine.

To support the wide varaity of use cases for the engine, it can be deployed in multiple ways.

## Standalone client

A standalone client is a local binary that runs on a local machine and connects directly to the source of truth:

<figure class="arch-figure">
<svg viewBox="0 0 920 484" role="img" aria-labelledby="standalone-title standalone-desc">
<title id="standalone-title">Standalone client architecture</title>
<desc id="standalone-desc">Object storage holds the tables in Delta Lake format. A single pivot process on one machine reads and writes them directly over HTTP: a SQL shell submits the query, a query engine runs it across one worker per core, and an in-memory cache holds 2 MB regions of the parquet files so repeat reads never reach the network.</desc>
<defs>
<marker id="standalone-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="156" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Object storage</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">S3 · GCS · Azure Blob</text>
<text x="64" y="74" class="arch-muted">analytics-lake</text>
<rect x="64" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="65" y="89" width="248" height="24" class="arch-strip" />
<line x1="65" y1="113" x2="313" y2="113" class="arch-rule" />
<text x="76" y="106" class="arch-label">events</text>
<text x="302" y="106" text-anchor="end" class="arch-tiny">Delta Lake format</text>
<text x="76" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="76" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<rect x="335" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="336" y="89" width="248" height="24" class="arch-strip" />
<line x1="336" y1="113" x2="584" y2="113" class="arch-rule" />
<text x="347" y="106" class="arch-label">orders</text>
<text x="573" y="106" text-anchor="end" class="arch-tiny">Delta Lake format</text>
<text x="347" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="347" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<rect x="606" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="607" y="89" width="248" height="24" class="arch-strip" />
<line x1="607" y1="113" x2="855" y2="113" class="arch-rule" />
<text x="618" y="106" class="arch-label">sessions</text>
<text x="844" y="106" text-anchor="end" class="arch-tiny">Delta Lake format</text>
<text x="618" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="618" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<line x1="760" y1="250" x2="760" y2="182" class="arch-line" marker-start="url(#standalone-head)" marker-end="url(#standalone-head)" />
<text x="774" y="214" class="arch-muted">read + write</text>
<text x="774" y="232" class="arch-tiny">on cache miss</text>
<rect x="40" y="250" width="840" height="210" rx="3" class="arch-panel" />
<text x="64" y="280" class="arch-title">Your machine</text>
<text x="856" y="280" text-anchor="end" class="arch-muted">one process, no server</text>
<rect x="64" y="296" width="250" height="140" rx="3" class="arch-inner" />
<rect x="65" y="297" width="248" height="24" class="arch-strip" />
<line x1="65" y1="321" x2="313" y2="321" class="arch-rule" />
<text x="76" y="314" class="arch-label">SQL shell</text>
<text x="302" y="314" text-anchor="end" class="arch-tiny">pivot open</text>
<circle cx="80" cy="345" r="3.5" class="arch-bullet" />
<text x="92" y="349" class="arch-cmd">pivot=&gt; SELECT count(*)</text>
<text x="92" y="367" class="arch-cmd">pivot-&gt;   FROM events;</text>
<text x="92" y="393" class="arch-tiny">1204831</text>
<text x="92" y="411" class="arch-tiny">(1 row)</text>
<rect x="335" y="296" width="250" height="140" rx="3" class="arch-inner" />
<rect x="336" y="297" width="248" height="24" class="arch-strip" />
<line x1="336" y1="321" x2="584" y2="321" class="arch-rule" />
<text x="347" y="314" class="arch-label">Query engine</text>
<text x="573" y="314" text-anchor="end" class="arch-tiny">one worker per core</text>
<rect x="347" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="369" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="391" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="413" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="435" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="457" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="479" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="501" y="340" width="16" height="16" rx="1" class="arch-cell" />
<rect x="347" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="369" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="391" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="413" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="435" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="457" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="479" y="362" width="16" height="16" rx="1" class="arch-cell" />
<rect x="501" y="362" width="16" height="16" rx="1" class="arch-cell" />
<text x="347" y="400" class="arch-tiny">morsel-driven parallelism</text>
<text x="347" y="418" class="arch-tiny">SIMD · NUMA-aware</text>
<rect x="606" y="296" width="250" height="140" rx="3" class="arch-inner" />
<rect x="607" y="297" width="248" height="24" class="arch-strip" />
<line x1="607" y1="321" x2="855" y2="321" class="arch-rule" />
<text x="618" y="314" class="arch-label">Cache</text>
<text x="844" y="314" text-anchor="end" class="arch-tiny">half of RAM</text>
<text x="618" y="344" class="arch-tiny">2 MB regions, compressed</text>
<rect x="618" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="636" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="654" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="672" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="690" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="708" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="726" y="356" width="14" height="14" rx="1" class="arch-bar" />
<rect x="744" y="356" width="14" height="14" rx="1" class="arch-inner" />
<rect x="762" y="356" width="14" height="14" rx="1" class="arch-inner" />
<rect x="780" y="356" width="14" height="14" rx="1" class="arch-inner" />
<text x="618" y="400" class="arch-tiny">hit → answered from RAM</text>
<text x="618" y="418" class="arch-tiny">miss → HTTP range request</text>
</svg>
</figure>



It is ideal for: 
- **Agentic / local ad-hoc analytics** - Users and agents query the source of truth directly, enabling local exploration without impacting a centralized server. Each client has its own cache and resources, providing predictable performance without competing with other workloads.


