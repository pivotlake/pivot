---
title: Introduction
description: What pivotdb is and how the pieces fit together.
---

pivotdb is a columnar analytics engine. It stores tables as parquet files in
object storage in an open format, so the cluster is one reader among several
rather than the only way in.

<figure class="arch-figure">
<svg viewBox="0 0 920 580" role="img" aria-labelledby="arch-title arch-desc">
<title id="arch-title">pivotdb architecture</title>
<desc id="arch-desc">Object storage holds pivotlake tables in Delta Lake format. A pivotdb cluster reads and writes those tables and serves SQL clients over the Postgres wire. A laptop and other engines such as DuckDB read the same files directly, without going through the cluster.</desc>
<defs>
<marker id="arch-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
<marker id="arch-head-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-accent-head" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="132" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Object storage</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">S3 · GCS · Azure Blob</text>
<rect x="64" y="84" width="792" height="52" rx="3" class="arch-inner" />
<text x="84" y="115" class="arch-label">pivotlake · Delta Lake format</text>
<rect x="420" y="97" width="100" height="26" rx="2" class="arch-panel" />
<text x="470" y="114" text-anchor="middle" class="arch-muted">part-000</text>
<rect x="530" y="97" width="100" height="26" rx="2" class="arch-panel" />
<text x="580" y="114" text-anchor="middle" class="arch-muted">part-001</text>
<rect x="640" y="97" width="100" height="26" rx="2" class="arch-panel" />
<text x="690" y="114" text-anchor="middle" class="arch-muted">part-002</text>
<rect x="750" y="97" width="100" height="26" rx="2" class="arch-panel" />
<text x="800" y="114" text-anchor="middle" class="arch-muted">_delta_log</text>
<line x1="230" y1="250" x2="230" y2="158" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="244" y="208" class="arch-muted">read + write</text>
<line x1="560" y1="250" x2="560" y2="158" class="arch-accent-line" marker-end="url(#arch-head-accent)" />
<text x="574" y="208" class="arch-accent-text">read</text>
<line x1="780" y1="250" x2="780" y2="158" class="arch-accent-line" marker-end="url(#arch-head-accent)" />
<text x="794" y="208" class="arch-accent-text">read</text>
<rect x="40" y="250" width="380" height="176" rx="3" class="arch-panel" />
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
<rect x="64" y="392" width="332" height="24" rx="2" class="arch-inner" />
<text x="76" y="408" class="arch-muted">scan · filter · join · aggregate</text>
<rect x="470" y="250" width="180" height="176" rx="3" class="arch-panel" />
<line x1="473" y1="251" x2="647" y2="251" class="arch-accent-line" stroke-width="2.5" />
<text x="560" y="280" text-anchor="middle" class="arch-title">Laptop</text>
<rect x="502" y="296" width="116" height="76" rx="3" class="arch-inner" />
<rect x="510" y="304" width="100" height="60" rx="2" class="arch-strip" />
<rect x="518" y="314" width="10" height="5" rx="2" class="arch-bar" />
<rect x="532" y="314" width="50" height="5" rx="2" class="arch-bar" />
<rect x="518" y="328" width="72" height="5" rx="2" class="arch-bar" />
<rect x="518" y="342" width="44" height="5" rx="2" class="arch-bar" />
<rect x="486" y="376" width="148" height="8" rx="4" class="arch-inner" />
<text x="560" y="408" text-anchor="middle" class="arch-muted">pivotdb shell</text>
<rect x="680" y="250" width="200" height="176" rx="3" class="arch-panel" />
<line x1="683" y1="251" x2="877" y2="251" class="arch-accent-line" stroke-width="2.5" />
<text x="780" y="280" text-anchor="middle" class="arch-title">Other engines</text>
<rect x="704" y="298" width="152" height="34" rx="2" class="arch-inner" />
<rect x="716" y="308" width="14" height="14" rx="1" class="arch-cell" />
<text x="740" y="320" class="arch-muted">DuckDB</text>
<rect x="704" y="338" width="152" height="34" rx="2" class="arch-inner" />
<rect x="716" y="348" width="14" height="14" rx="1" class="arch-cell" />
<text x="740" y="360" class="arch-muted">Spark</text>
<rect x="704" y="378" width="152" height="34" rx="2" class="arch-inner" />
<rect x="716" y="388" width="14" height="14" rx="1" class="arch-cell" />
<text x="740" y="400" class="arch-muted">pandas · Trino</text>
<line x1="230" y1="474" x2="230" y2="428" class="arch-line" marker-end="url(#arch-head)" />
<text x="244" y="455" class="arch-muted">Postgres wire</text>
<rect x="40" y="474" width="380" height="72" rx="3" class="arch-panel" />
<text x="64" y="498" class="arch-title">SQL clients</text>
<rect x="64" y="508" width="104" height="26" rx="2" class="arch-inner" />
<text x="116" y="525" text-anchor="middle" class="arch-muted">psql</text>
<rect x="178" y="508" width="104" height="26" rx="2" class="arch-inner" />
<text x="230" y="525" text-anchor="middle" class="arch-muted">BI tools</text>
<rect x="292" y="508" width="104" height="26" rx="2" class="arch-inner" />
<text x="344" y="525" text-anchor="middle" class="arch-muted">drivers</text>
<text x="675" y="498" text-anchor="middle" class="arch-accent-text">same files, no cluster</text>
<text x="675" y="518" text-anchor="middle" class="arch-accent-text">in the read path</text>
</svg>
</figure>

The cluster owns writes: it lands parquet and commits to the Delta log. Reads
are not exclusive to it. Anything that speaks Delta Lake can point at the same
prefix and get the same tables, which is why a laptop and a DuckDB process sit
next to the cluster in the drawing rather than behind it.

## Where to start

- [Quickstart](/docs/quickstart/) runs a server and issues a first query.
- [Architecture](/docs/database/architecture/) explains how a query becomes
  work across the dispatch pool.

## Writing docs

Every page under `docs/src/content/docs/` is a markdown file with a
`title` in its frontmatter. Directories become sidebar groups. Nothing else is
required to add a page.
