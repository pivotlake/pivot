---
title: Introduction
description: What pivotdb is and how the pieces fit together.
---

pivotdb is a columnar analytics engine. It stores tables as parquet files in
object storage in an open format, so the cluster is one reader among several
rather than the only way in.

<figure class="arch-figure">
<svg viewBox="0 0 920 552" role="img" aria-labelledby="arch-title arch-desc">
<title id="arch-title">pivotdb architecture</title>
<desc id="arch-desc">Object storage holds pivotlake tables in Delta Lake format. A Pivot cluster reads and writes them and serves SQL clients over the Postgres wire. Agents, each embedding its own pivot open, and third-party engines such as DuckDB, read the same files directly without going through the cluster.</desc>
<defs>
<marker id="arch-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="156" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Object storage</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">S3 · GCS · Azure Blob</text>
<text x="64" y="74" class="arch-muted">pivotlake</text>
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
<line x1="215" y1="250" x2="215" y2="182" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="229" y="220" class="arch-muted">read + write</text>
<line x1="530" y1="250" x2="530" y2="182" class="arch-line" marker-end="url(#arch-head)" />
<text x="544" y="220" class="arch-muted">read</text>
<line x1="775" y1="250" x2="775" y2="182" class="arch-line" marker-end="url(#arch-head)" />
<text x="789" y="220" class="arch-muted">read</text>
<rect x="40" y="250" width="350" height="160" rx="3" class="arch-panel" />
<text x="64" y="280" class="arch-title">Pivot cluster</text>
<text x="366" y="280" text-anchor="end" class="arch-muted">writer</text>
<rect x="64" y="296" width="94" height="86" rx="2" class="arch-inner" />
<rect x="65" y="297" width="92" height="23" class="arch-strip" />
<line x1="65" y1="320" x2="157" y2="320" class="arch-rule" />
<text x="111" y="313" text-anchor="middle" class="arch-muted">node 1</text>
<rect x="70" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="92" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="114" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="136" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="70" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="92" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="114" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="136" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="168" y="296" width="94" height="86" rx="2" class="arch-inner" />
<rect x="169" y="297" width="92" height="23" class="arch-strip" />
<line x1="169" y1="320" x2="261" y2="320" class="arch-rule" />
<text x="215" y="313" text-anchor="middle" class="arch-muted">node 2</text>
<rect x="174" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="196" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="218" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="240" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="174" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="196" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="218" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="240" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="272" y="296" width="94" height="86" rx="2" class="arch-inner" />
<rect x="273" y="297" width="92" height="23" class="arch-strip" />
<line x1="273" y1="320" x2="365" y2="320" class="arch-rule" />
<text x="319" y="313" text-anchor="middle" class="arch-muted">node 3</text>
<rect x="278" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="300" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="322" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="344" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="278" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="300" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="322" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="344" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="410" y="250" width="240" height="160" rx="3" class="arch-panel" />
<text x="530" y="280" text-anchor="middle" class="arch-title">Agents</text>
<rect x="426" y="296" width="208" height="30" rx="2" class="arch-inner" />
<text x="436" y="316" class="arch-prompt">$</text>
<text x="449" y="316" class="arch-cmd">pivot open s3://pivotlake</text>
<rect x="426" y="336" width="208" height="30" rx="2" class="arch-inner" />
<text x="436" y="356" class="arch-prompt">$</text>
<text x="449" y="356" class="arch-cmd">pivot open s3://pivotlake</text>
<rect x="670" y="250" width="210" height="160" rx="3" class="arch-panel" />
<text x="775" y="280" text-anchor="middle" class="arch-title">Other engines</text>
<rect x="684" y="296" width="182" height="32" rx="2" class="arch-inner" />
<rect x="696" y="305" width="14" height="14" rx="1" class="arch-cell" />
<text x="718" y="317" class="arch-muted">DuckDB</text>
<rect x="684" y="332" width="182" height="32" rx="2" class="arch-inner" />
<rect x="696" y="341" width="14" height="14" rx="1" class="arch-cell" />
<text x="718" y="353" class="arch-muted">Spark</text>
<rect x="684" y="368" width="182" height="32" rx="2" class="arch-inner" />
<rect x="696" y="377" width="14" height="14" rx="1" class="arch-cell" />
<text x="718" y="389" class="arch-muted">pandas · Trino</text>
<line x1="215" y1="460" x2="215" y2="412" class="arch-line" marker-end="url(#arch-head)" />
<text x="229" y="440" class="arch-muted">Postgres wire</text>
<rect x="40" y="460" width="350" height="72" rx="3" class="arch-panel" />
<text x="64" y="484" class="arch-title">SQL clients</text>
<rect x="64" y="494" width="76" height="26" rx="2" class="arch-inner" />
<text x="102" y="511" text-anchor="middle" class="arch-muted">backend</text>
<rect x="150" y="494" width="60" height="26" rx="2" class="arch-inner" />
<text x="180" y="511" text-anchor="middle" class="arch-muted">psql</text>
<rect x="220" y="494" width="84" height="26" rx="2" class="arch-inner" />
<text x="262" y="511" text-anchor="middle" class="arch-muted">BI tools</text>
</svg>
</figure>

The cluster owns writes: it lands parquet and commits to the Delta log. Reads
are not exclusive to it. Each agent embeds its own pivot open and reads
the tables in its own process, so agents scale out without queueing behind a
shared server, and anything else that speaks Delta Lake can point at the same
prefix and get the same data. That is why those readers sit next to the
cluster in the drawing rather than behind it.

## Where to start

- [Quickstart](/docs/quickstart/) runs a server and issues a first query.
- [Architecture](/docs/database/architecture/) explains how a query becomes
  work across the dispatch pool.

## Writing docs

Every page under `docs/src/content/docs/` is a markdown file with a
`title` in its frontmatter. Directories become sidebar groups. Nothing else is
required to add a page.
