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
<desc id="arch-desc">Object storage holds pivotlake tables in Delta Lake format. A pivotdb cluster reads and writes them and serves SQL clients over the Postgres wire. Agents, each embedding its own pivot open, and third-party engines such as DuckDB, read the same files directly without going through the cluster.</desc>
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
<line x1="230" y1="250" x2="230" y2="182" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="244" y="220" class="arch-muted">read + write</text>
<line x1="570" y1="250" x2="570" y2="182" class="arch-line" marker-end="url(#arch-head)" />
<text x="584" y="220" class="arch-muted">read</text>
<line x1="795" y1="250" x2="795" y2="182" class="arch-line" marker-end="url(#arch-head)" />
<text x="809" y="220" class="arch-muted">read</text>
<rect x="40" y="250" width="380" height="160" rx="3" class="arch-panel" />
<text x="64" y="280" class="arch-title">pivotdb cluster</text>
<text x="396" y="280" text-anchor="end" class="arch-muted">writer</text>
<rect x="64" y="296" width="104" height="86" rx="2" class="arch-inner" />
<rect x="65" y="297" width="102" height="23" class="arch-strip" />
<line x1="65" y1="320" x2="167" y2="320" class="arch-rule" />
<text x="116" y="313" text-anchor="middle" class="arch-muted">node 1</text>
<rect x="75" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="97" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="119" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="141" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="75" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="97" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="119" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="141" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="178" y="296" width="104" height="86" rx="2" class="arch-inner" />
<rect x="179" y="297" width="102" height="23" class="arch-strip" />
<line x1="179" y1="320" x2="281" y2="320" class="arch-rule" />
<text x="230" y="313" text-anchor="middle" class="arch-muted">node 2</text>
<rect x="189" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="211" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="233" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="255" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="189" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="211" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="233" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="255" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="292" y="296" width="104" height="86" rx="2" class="arch-inner" />
<rect x="293" y="297" width="102" height="23" class="arch-strip" />
<line x1="293" y1="320" x2="395" y2="320" class="arch-rule" />
<text x="344" y="313" text-anchor="middle" class="arch-muted">node 3</text>
<rect x="303" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="325" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="347" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="369" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="303" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="325" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="347" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="369" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="450" y="250" width="240" height="160" rx="3" class="arch-panel" />
<text x="570" y="280" text-anchor="middle" class="arch-title">Agents</text>
<rect x="470" y="296" width="94" height="46" rx="2" class="arch-inner" />
<rect x="471" y="297" width="92" height="16" class="arch-strip" />
<line x1="471" y1="313" x2="563" y2="313" class="arch-rule" />
<text x="517" y="309" text-anchor="middle" class="arch-tiny">agent</text>
<rect x="471" y="318" width="92" height="20" class="arch-term" />
<text x="477" y="332" class="arch-prompt">$</text>
<text x="490" y="332" class="arch-cmd">pivot open</text>
<rect x="576" y="296" width="94" height="46" rx="2" class="arch-inner" />
<rect x="577" y="297" width="92" height="16" class="arch-strip" />
<line x1="577" y1="313" x2="669" y2="313" class="arch-rule" />
<text x="623" y="309" text-anchor="middle" class="arch-tiny">agent</text>
<rect x="577" y="318" width="92" height="20" class="arch-term" />
<text x="583" y="332" class="arch-prompt">$</text>
<text x="596" y="332" class="arch-cmd">pivot open</text>
<rect x="470" y="352" width="94" height="46" rx="2" class="arch-inner" />
<rect x="471" y="353" width="92" height="16" class="arch-strip" />
<line x1="471" y1="369" x2="563" y2="369" class="arch-rule" />
<text x="517" y="365" text-anchor="middle" class="arch-tiny">agent</text>
<rect x="471" y="374" width="92" height="20" class="arch-term" />
<text x="477" y="388" class="arch-prompt">$</text>
<text x="490" y="388" class="arch-cmd">pivot open</text>
<rect x="576" y="352" width="94" height="46" rx="2" class="arch-inner" />
<rect x="577" y="353" width="92" height="16" class="arch-strip" />
<line x1="577" y1="369" x2="669" y2="369" class="arch-rule" />
<text x="623" y="365" text-anchor="middle" class="arch-tiny">agent</text>
<rect x="577" y="374" width="92" height="20" class="arch-term" />
<text x="583" y="388" class="arch-prompt">$</text>
<text x="596" y="388" class="arch-cmd">pivot open</text>
<rect x="710" y="250" width="170" height="160" rx="3" class="arch-panel" />
<text x="795" y="280" text-anchor="middle" class="arch-title">Other engines</text>
<rect x="724" y="296" width="142" height="32" rx="2" class="arch-inner" />
<rect x="736" y="305" width="14" height="14" rx="1" class="arch-cell" />
<text x="758" y="317" class="arch-muted">DuckDB</text>
<rect x="724" y="332" width="142" height="32" rx="2" class="arch-inner" />
<rect x="736" y="341" width="14" height="14" rx="1" class="arch-cell" />
<text x="758" y="353" class="arch-muted">Spark</text>
<rect x="724" y="368" width="142" height="32" rx="2" class="arch-inner" />
<rect x="736" y="377" width="14" height="14" rx="1" class="arch-cell" />
<text x="758" y="389" class="arch-muted">pandas · Trino</text>
<line x1="230" y1="460" x2="230" y2="412" class="arch-line" marker-end="url(#arch-head)" />
<text x="244" y="440" class="arch-muted">Postgres wire</text>
<rect x="40" y="460" width="380" height="72" rx="3" class="arch-panel" />
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
