---
title: Why is Pivot fast?
description: How Pivot avoids unnecessary work and uses the machine efficiently.
sidebar:
  order: 2
---

Although database performance comes from many different optimization strategies and architectural decisions, most fall into of couple of main categories. To fully maximize performance, pivot tries to utilize optimizations in each one of these categories:

### It processes less data
The fastest way to process data is to avoid processing it in the first place. Pivot uses several strategies to minimize the amount of data that needs to be read and processed for each query. Some of these include:

#### Soft ordering data

Parquet stores min/max statistics for each column in every row group. When evaluating a query, query engines can use these statistics to determine that certain row groups cannot contain matching rows and skip them entirely.

For example, consider the following query:

```sql
SELECT * FROM sales WHERE user_id=1377;
```

Any row group whose user_id range does not include 1377 can be skipped. A row group with a minimum user_id of 10 and a maximum of 200, for example, cannot possibly contain the requested user and therefore does not need to be read.

While row group pruning is common among engines that query Parquet, Pivot takes greater advantage of it by allowing users to **soft-sort** their tables:

```sql
CREATE TABLE sales (
  user_id BIGINT,
  amount DOUBLE,
  ts TIMESTAMP
) WITH (
  sort_by = 'user_id'
);
```

Soft sorting organizes rows within generated Parquet files to make row group statistics more selective for columns commonly used by a workload. Pivot also prioritizes compacting overlapping files, helping keep similar values clustered together and further improving pruning efficiency.

For example, a sales table can be configured to soft-sort data by user_id. Without sorting, values may be scattered across row groups, producing heavily overlapping min/max ranges:

| Row Group | Min User ID | Max User ID |
|-----------|-------------|-------------|
| 1         | 10          | 2000        |
| 2         | 5           | 1500        |
| 3         | 1270        | 1800        |
| 4         | 300         | 1350        |

A query filtering for user_id = 1377 would need to read row groups 1, 2, and 3 because all three ranges could contain the requested value.

After soft-sorting by user_id, the ranges become much more selective:

| Row Group | Min User ID | Max User ID |
|-----------|-------------|-------------|
| 1         | 5           | 500         |
| 2         | 501         | 1000        |
| 3         | 1001        | 1500        |
| 4         | 1501        | 2000        |

Now, the same query only needs to read row group 3, allowing Pivot to prune the other three entirely.

Soft sorting alone, however, does not guarantee that data remains well clustered over time. As new Parquet files are written, their value ranges can begin to overlap:

| File | Min User ID | Max User ID |
|------|-------------|-------------|
| A    | 1           | 1000        |
| B    | 2           | 1300        |
| C    | 400         | 1400        |
| D    | 900         | 1700        |

Even if each file is internally sorted, a query for user_id = 1200 would need to inspect files B, C, and D.

Pivot’s compaction strategy prioritizes files with overlapping ranges. By compacting and re-sorting these files, the data can be reorganized into less-overlapping ranges:

| File | Min User ID | Max User ID |
|------|-------------|-------------|
| A    | 1           | 500         |
| B    | 501         | 1000        |
| C    | 1001        | 1500        |
| D    | 1501        | 2000        |

The same query for user_id = 1200 can now prune three of the four files and only read file C.

Background compaction does this in bulk. Files whose sort-key ranges overlap are rewritten together, six at a time, once six such files have accumulated: sorting six files at once and cutting the result back into full-size files narrows each file's range about sixfold per rewrite, so a row reaches its final place in a few rewrites however the data arrived. Files whose ranges do not overlap are never rewritten, so a table whose sort key arrives in order, such as a timestamp, costs no compaction work beyond merging small files. `COMPACT table FINAL` finishes the job on demand, rewriting until no two files' ranges overlap.

Together, soft sorting and overlap-aware compaction keep similar values clustered as the table evolves, improving pruning at both the file and row-group level.

This is conceptually similar to ClickHouse’s sparse primary-key index / table order by definition, but applied to Parquet files and open table formats.

#### Late materialization

Queries often return many columns even though only a small subset is needed to determine which rows belong in the final result. Reading and decoding all selected columns upfront can therefore waste significant I/O and CPU on rows that will eventually be discarded.

Pivot uses late materialization to delay reading columns until they are actually needed.

For example, consider a query that returns detailed information about the 10 most recently active users in US:
```sql
SELECT
    user_id,
    name,
    email,
    country,
    city,
    company,
    job_title,
    profile_image_url,
    bio,
    last_activity
FROM users
WHERE country = 'US'
ORDER BY last_activity DESC
LIMIT 10;
```

Although the query selects many columns, only country and last_activity are needed to determine which 10 rows should be returned. Instead of immediately reading and decoding every selected column, Pivot can first process the columns required for filtering and Top-K selection.

Once the 10 matching rows have been identified, the remaining columns—such as name, email, city, company, job_title, profile_image_url, and bio—can be materialized only for the surviving rows.

For wide tables or highly selective queries, this can significantly reduce the amount of data read, decoded, and processed. For example, on ClickBench Q23 (a query of a similar shape) running on a c8g.4xlarge instance with late materialization on and off:

<figure class="arch-figure">
<svg viewBox="0 0 920 216" role="img" aria-labelledby="latmat-title latmat-desc">
<title id="latmat-title">Late materialization speedup</title>
<desc id="latmat-desc">Two bar charts comparing query time with late materialization on and off. On a cold run the query takes 2.03 seconds with late materialization and 8.92 seconds without, 4.4 times faster. On a hot run it takes 33.4 milliseconds with late materialization and 181.4 milliseconds without, 5.4 times faster. Each chart uses its own time scale.</desc>
<rect x="40" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Cold run</text>
<text x="64" y="90" class="arch-label">Late materialization on</text>
<text x="420" y="90" text-anchor="end" class="arch-tiny">4.4× faster</text>
<line x1="64.5" y1="96" x2="64.5" y2="176" class="arch-rule" />
<rect x="65" y="100" width="59" height="16" rx="1" class="arch-bar-strong" />
<text x="134" y="113" class="arch-label">2.03 s</text>
<text x="64" y="146" class="arch-label">Late materialization off</text>
<rect x="65" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="335" y="169" class="arch-label">8.92 s</text>
<rect x="476" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="500" y="54" class="arch-title">Hot run</text>
<text x="500" y="90" class="arch-label">Late materialization on</text>
<text x="856" y="90" text-anchor="end" class="arch-tiny">5.4× faster</text>
<line x1="500.5" y1="96" x2="500.5" y2="176" class="arch-rule" />
<rect x="501" y="100" width="48" height="16" rx="1" class="arch-bar-strong" />
<text x="559" y="113" class="arch-label">33.4 ms</text>
<text x="500" y="146" class="arch-label">Late materialization off</text>
<rect x="501" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="771" y="169" class="arch-label">181.4 ms</text>
</svg>
</figure>


#### Parquet-optimized pruning
Pivot is built specifically to perform well on Parquet. After laying the data out for efficient pruning and filtering, Pivot tries to leverage a couple of Parquet-specific strategies to prune data as much as possible.

For example, in a query such as:
```sql
SELECT * FROM events WHERE country = 'IL';
```

Pivot can eliminate work at several levels:

- **Dictionary-based pruning** — if the country column is [dictionary-encoded](https://parquet.apache.org/docs/file-format/data-pages/encodings/#DICTIONARY), Pivot can first check whether "IL" exists in the dictionary of that column. If it does not, every data page referencing that dictionary can be skipped without decompressing or decoding it.
- **Page-level pruning** — even when a row group contains matching rows, many of its pages may contain none of the rows that need to be read. Pivot uses the selected row positions to identify these pages and skips their decompression and decoding entirely.
- **Selective decoding** — for pages that do need to be read, Pivot passes the surviving row positions down into the Parquet reader. Where the encoding allows, values belonging to unwanted rows can be skipped without being decoded at all.
- **Bloom-filter pruning (coming soon)** — Pivot will also use Parquet Bloom filters to quickly determine that a row group cannot contain a requested value, allowing it to be skipped entirely.

### It works well with the OS
Linux is an amazing operating system. It can run a wide variety of workloads with great performance and stability. However, like many low-level systems, it is difficult to build a “generalist” system or algorithm that performs optimally across very different workloads.

Because of this, some databases have explored building specialized operating systems (e.g. [DBOS](https://dbos-project.github.io/)) or bypassing parts of the OS entirely, giving the database more direct control over how resources are managed.

Rather than forcing Pivot users to move to an unfamiliar or less mature operating system, Pivot tries to get the best of both worlds: taking as much control as possible over scheduling, I/O, and memory management from the OS, while still benefiting from the stability, ecosystem, and extensive set of libraries that Linux has to offer.

#### Memory: custom memory management
Instead of relying on a general-purpose allocator, Pivot uses its own block-based allocator for most of its memory needs. Every operation in Pivot (decompression, decoding, and so on) works over a collection of fixed-size, non-contiguous 2 MB blocks rather than a single contiguous buffer.

This predictable and simple memory model provides several advantages over a traditional general-purpose allocator:

**Fewer page faults** - Since all of Pivot's buffers are in a constant size -- they can be `mmap`ed and faulted in once at startup, then kept for the lifetime of the process. Acquiring a buffer is simply a pop from a per-worker free list, and releasing one is a push. There are no size classes to select, no fragmentation to manage, and no mmap or munmap calls on the allocation path like other allocators have:

<figure class="arch-figure">
<svg viewBox="0 0 920 216" role="img" aria-labelledby="alloc-title alloc-desc">
<title id="alloc-title">Buffer ring versus jemalloc for 2 MB allocations</title>
<desc id="alloc-desc">Two bar charts comparing Pivot's buffer ring with jemalloc when handling 2 MB buffers. Acquiring and releasing one buffer costs 46 nanoseconds through the ring and 249 nanoseconds through jemalloc, 5.4 times cheaper. Writing to 2 GB of freshly acquired buffers for the first time takes 3.5 milliseconds with the ring and 413 milliseconds with jemalloc, 118 times faster, because the ring's memory is already faulted in. Each chart uses its own scale.</desc>
<rect x="40" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Acquire + release</text>
<text x="420" y="54" text-anchor="end" class="arch-muted">per 2 MB buffer</text>
<text x="64" y="90" class="arch-label">Pivot buffer ring</text>
<text x="420" y="90" text-anchor="end" class="arch-tiny">5.4× cheaper</text>
<line x1="64.5" y1="96" x2="64.5" y2="176" class="arch-rule" />
<rect x="65" y="100" width="48" height="16" rx="1" class="arch-bar-strong" />
<text x="123" y="113" class="arch-label">46 ns</text>
<text x="64" y="146" class="arch-label">jemalloc</text>
<rect x="65" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="335" y="169" class="arch-label">249 ns</text>
<rect x="476" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="500" y="54" class="arch-title">First write</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">2 GB of fresh buffers</text>
<text x="500" y="90" class="arch-label">Pivot buffer ring</text>
<text x="856" y="90" text-anchor="end" class="arch-tiny">118× faster</text>
<line x1="500.5" y1="96" x2="500.5" y2="176" class="arch-rule" />
<rect x="501" y="100" width="3" height="16" rx="1" class="arch-bar-strong" />
<text x="514" y="113" class="arch-label">3.5 ms</text>
<text x="500" y="146" class="arch-label">jemalloc</text>
<rect x="501" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="771" y="169" class="arch-label">413 ms</text>
</svg>
<figcaption>100 rounds of acquiring 1000 × 2 MB buffers, touching every page, and releasing them. AWS c8g.4xlarge.</figcaption>
</figure>

**Huge pages can be utilized.** Because every buffer is exactly 2 MB, Pivot asks Linux to back the buffer ring with transparent 2 MB huge pages instead of ordinary 4 KB pages. When a query's working set spans more pages than the [TLB](https://en.wikipedia.org/wiki/Translation_lookaside_buffer) can hold, each miss costs a multi-level walk through the page tables. With huge pages, one TLB entry covers 512 times as much memory, so operators that jump around large structures, such as the hash tables behind joins and aggregations, hit in the dTLB far more often and walk the page tables far less:

<figure class="arch-figure">
<svg viewBox="0 0 920 216" role="img" aria-labelledby="tlb-title tlb-desc">
<title id="tlb-title">TPC-H q21 with and without huge pages</title>
<desc id="tlb-desc">Two bar charts for TPC-H SF100 query 21 run hot with the buffer ring on huge pages and on ordinary pages. Query time is 2.47 seconds with huge pages and 3.13 seconds without, 21 percent faster. Data TLB page walks are 1.2 billion with huge pages and 5.8 billion without, 4.8 times fewer.</desc>
<rect x="40" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Query time</text>
<text x="420" y="54" text-anchor="end" class="arch-muted">hot run</text>
<text x="64" y="90" class="arch-label">Huge pages on</text>
<text x="420" y="90" text-anchor="end" class="arch-tiny">21% faster</text>
<line x1="64.5" y1="96" x2="64.5" y2="176" class="arch-rule" />
<rect x="65" y="100" width="205" height="16" rx="1" class="arch-bar-strong" />
<text x="280" y="113" class="arch-label">2.47 s</text>
<text x="64" y="146" class="arch-label">Huge pages off</text>
<rect x="65" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="335" y="169" class="arch-label">3.13 s</text>
<rect x="476" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="500" y="54" class="arch-title">Data TLB page walks</text>
<text x="500" y="90" class="arch-label">Huge pages on</text>
<text x="856" y="90" text-anchor="end" class="arch-tiny">4.8× fewer</text>
<line x1="500.5" y1="96" x2="500.5" y2="176" class="arch-rule" />
<rect x="501" y="100" width="54" height="16" rx="1" class="arch-bar-strong" />
<text x="565" y="113" class="arch-label">1.2 billion</text>
<text x="500" y="146" class="arch-label">Huge pages off</text>
<rect x="501" y="156" width="260" height="16" rx="1" class="arch-bar" />
<text x="771" y="169" class="arch-label">5.8 billion</text>
</svg>
<figcaption>TPC-H SF100 q21, hot, with transparent huge pages enabled and disabled in the kernel. AWS c8g.4xlarge.</figcaption>
</figure>

#### Scheduling: a thread per core architecture with custom internal scheduling:
In the world of database performance, context switches are one of the worst enemies of efficient execution. When multiple tasks or threads compete for the same CPU core, constantly switching between them adds overhead and, more importantly, disrupts the CPU caches that each task relies on. The result is more time spent switching between work and less time actually doing it.

To minimize this overhead, Pivot uses a thread-per-core architecture, designed so that context switches almost never occur during query execution. Each CPU core runs a dedicated Pivot “worker” responsible for scheduling, executing, and orchestrating the work assigned to that core.

To make the most of the CPU caches, each worker also prefers to schedule operations whose data is already “hot” in cache over operations that would require bringing new data in. For example, after decoding a batch of data, a worker will prefer to immediately run a filter over that batch rather than start decoding a new one.

When a worker runs out of work, it can also steal work from a neighboring worker. This allows idle cores to help busy ones, keeping CPU utilization high while still preserving the cache locality of the thread-per-core model:

<figure class="arch-figure">
<svg viewBox="0 0 920 216" role="img" aria-labelledby="steal-title steal-desc">
<title id="steal-title">The same query on two cores, without and with work stealing</title>
<desc id="steal-desc">Two timelines. Without stealing, worker 1 has nine batches and worker 2 has three, so worker 2 finishes early and sits idle while worker 1 works through the rest, and the query finishes when worker 1 does. With stealing, once worker 2 runs out of its own batches it steals the three batches at the oldest end of worker 1 queue. Worker 1 keeps running its newest batches, both workers finish after six batches, and the query completes a third sooner.</desc>
<defs>
<marker id="steal-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Without stealing</text>
<text x="420" y="54" text-anchor="end" class="arch-muted">time</text>
<line x1="64.5" y1="70" x2="64.5" y2="176" class="arch-rule" />
<text x="64" y="90" class="arch-label">Worker 1</text>
<rect x="65" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="101" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="137" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="173" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="209" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="245" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="281" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="317" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="353" y="100" width="34" height="16" rx="1" class="arch-bar" />
<text x="64" y="146" class="arch-label">Worker 2</text>
<rect x="65" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="101" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="137" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="173" y="156" width="216" height="16" rx="1" class="arch-inner" />
<text x="281" y="168" text-anchor="middle" class="arch-tiny">idle</text>
<path d="M389.5,96 V176" class="arch-line" stroke-dasharray="3 3" />
<text x="389" y="192" text-anchor="middle" class="arch-tiny">query done</text>
<rect x="476" y="24" width="404" height="172" rx="3" class="arch-panel" />
<text x="500" y="54" class="arch-title">With stealing</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">time</text>
<line x1="500.5" y1="70" x2="500.5" y2="176" class="arch-rule" />
<text x="500" y="90" class="arch-label">Worker 1</text>
<text x="825" y="90" text-anchor="end" class="arch-tiny">oldest end of its queue</text>
<rect x="501" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="537" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="573" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="609" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="645" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="681" y="100" width="34" height="16" rx="1" class="arch-bar" />
<rect x="717" y="100" width="34" height="16" rx="1" class="arch-inner" />
<rect x="753" y="100" width="34" height="16" rx="1" class="arch-inner" />
<rect x="789" y="100" width="34" height="16" rx="1" class="arch-inner" />
<text x="500" y="146" class="arch-label">Worker 2</text>
<rect x="501" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="537" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="573" y="156" width="34" height="16" rx="1" class="arch-bar" />
<rect x="609" y="156" width="34" height="16" rx="1" class="arch-bar-strong" />
<rect x="645" y="156" width="34" height="16" rx="1" class="arch-bar-strong" />
<rect x="681" y="156" width="34" height="16" rx="1" class="arch-bar-strong" />
<path d="M770,118 C770,140 662,132 662,153" class="arch-line" marker-end="url(#steal-arrow)" />
<path d="M717.5,96 V176" class="arch-line" stroke-dasharray="3 3" />
<text x="717" y="192" text-anchor="middle" class="arch-tiny">query done</text>
</svg>
<figcaption>Twelve equal batches on two cores. Without stealing, Worker 2 finishes its share and idles while Worker 1 works through the rest. With stealing, Worker 2 takes batches from the oldest end of Worker 1's queue, the ones Worker 1 would have reached last and that have long left its cache, so Worker 1 keeps running its newest, still-hot batches and both cores stay busy until the query is done. Stealing stays within a NUMA node, since a batch's memory lives on the node that produced it.</figcaption>
</figure>

When multiple queries run in parallel, Pivot workers schedule work across them in a way that balances CPU cache locality with fairness. Workers try to balance between executing operations whose data is already hot in cache, and ensuring that no single query consumes all available CPU time and causes others to starve.

#### IO: direct IO with IO uring:
As a complement to its thread-per-core architecture, Pivot uses io_uring for I/O. This allows each worker to efficiently dispatch and manage many concurrent I/O operations while reducing the syscall overhead associated with reads and writes.

Pivot also avoids relying on the operating system’s page cache for caching disk data, as some other databases such as ClickHouse do by default. Instead, it uses direct I/O together with its own dedicated caching system.

Managing the cache directly gives Pivot more control over what memory is used for. Rather than having disk pages cached independently by the operating system, Pivot can make eviction decisions across different types of cached data—for example, choosing between compressed data read from disk and decompressed or otherwise processed data that is more expensive to reconstruct.

Because these decisions are made by the database itself, Pivot can prioritize cached objects based on their actual value to query execution, rather than relying on the more general-purpose caching policies of the operating system.

### It utilizes the hardware well.

#### Utilizing the CPU cache
Pivot’s execution strategy is inspired by [Morsel driven parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf). It prioritizes cache locality by processing data in small batches (i.e. morsels) sized to fit within the CPU’s L1 cache, and tries to perform as much work as possible on each morsel while its data remains hot in the cache: 

<figure class="arch-figure">
<svg viewBox="0 0 920 260" role="img" aria-labelledby="pipeline-title pipeline-desc">
<title id="pipeline-title">A table split into morsels, each run through the whole pipeline</title>
<desc id="pipeline-desc">A table with columns A, B and C is split into four morsels of rows. Morsel 1 is taken and passed through a pipeline of three operators, decode, then filter, then aggregate, and on into the result, while morsels 2 to 4 wait. A bracket under the pipeline notes that every operator works on the same morsel while it is still in the L1 cache.</desc>
<defs>
<marker id="pipeline-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="212" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Morsel-at-a-time execution</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">one worker</text>
<text x="138" y="90" text-anchor="middle" class="arch-muted">A</text>
<text x="174" y="90" text-anchor="middle" class="arch-muted">B</text>
<text x="210" y="90" text-anchor="middle" class="arch-muted">C</text>
<text x="64" y="116" class="arch-tiny">Morsel 1</text>
<rect x="120" y="100" width="35" height="11" class="arch-bar-strong" />
<rect x="156" y="100" width="35" height="11" class="arch-bar-strong" />
<rect x="192" y="100" width="35" height="11" class="arch-bar-strong" />
<rect x="120" y="112" width="35" height="11" class="arch-bar-strong" />
<rect x="156" y="112" width="35" height="11" class="arch-bar-strong" />
<rect x="192" y="112" width="35" height="11" class="arch-bar-strong" />
<text x="64" y="146" class="arch-tiny">Morsel 2</text>
<rect x="120" y="130" width="35" height="11" class="arch-bar" />
<rect x="156" y="130" width="35" height="11" class="arch-bar" />
<rect x="192" y="130" width="35" height="11" class="arch-bar" />
<rect x="120" y="142" width="35" height="11" class="arch-bar" />
<rect x="156" y="142" width="35" height="11" class="arch-bar" />
<rect x="192" y="142" width="35" height="11" class="arch-bar" />
<text x="64" y="176" class="arch-tiny">Morsel 3</text>
<rect x="120" y="160" width="35" height="11" class="arch-bar" />
<rect x="156" y="160" width="35" height="11" class="arch-bar" />
<rect x="192" y="160" width="35" height="11" class="arch-bar" />
<rect x="120" y="172" width="35" height="11" class="arch-bar" />
<rect x="156" y="172" width="35" height="11" class="arch-bar" />
<rect x="192" y="172" width="35" height="11" class="arch-bar" />
<text x="64" y="206" class="arch-tiny">Morsel 4</text>
<rect x="120" y="190" width="35" height="11" class="arch-bar" />
<rect x="156" y="190" width="35" height="11" class="arch-bar" />
<rect x="192" y="190" width="35" height="11" class="arch-bar" />
<rect x="120" y="202" width="35" height="11" class="arch-bar" />
<rect x="156" y="202" width="35" height="11" class="arch-bar" />
<rect x="192" y="202" width="35" height="11" class="arch-bar" />
<path d="M230,112 C264,112 264,156 296,156" class="arch-line" marker-end="url(#pipeline-arrow)" />
<rect x="300" y="136" width="100" height="40" rx="2" class="arch-inner" />
<text x="350" y="161" text-anchor="middle" class="arch-label">Decode</text>
<rect x="460" y="136" width="100" height="40" rx="2" class="arch-inner" />
<text x="510" y="161" text-anchor="middle" class="arch-label">Filter</text>
<rect x="620" y="136" width="100" height="40" rx="2" class="arch-inner" />
<text x="670" y="161" text-anchor="middle" class="arch-label">Aggregate</text>
<path d="M402,156 H411" class="arch-line" />
<rect x="412" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="424" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="436" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="412" y="156" width="11" height="5" class="arch-bar-strong" />
<rect x="424" y="156" width="11" height="5" class="arch-bar-strong" />
<rect x="436" y="156" width="11" height="5" class="arch-bar-strong" />
<path d="M448,156 H458" class="arch-line" marker-end="url(#pipeline-arrow)" />
<path d="M562,156 H571" class="arch-line" />
<rect x="572" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="584" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="596" y="150" width="11" height="5" class="arch-bar-strong" />
<rect x="572" y="156" width="11" height="5" class="arch-bar-strong" />
<rect x="584" y="156" width="11" height="5" class="arch-bar-strong" />
<rect x="596" y="156" width="11" height="5" class="arch-bar-strong" />
<path d="M608,156 H618" class="arch-line" marker-end="url(#pipeline-arrow)" />
<path d="M722,156 H776" class="arch-line" marker-end="url(#pipeline-arrow)" />
<rect x="780" y="136" width="76" height="40" rx="2" class="arch-inner" />
<text x="818" y="161" text-anchor="middle" class="arch-label">Result</text>
<path d="M300.5,188 v6 H719.5 v-6" class="arch-line" />
<text x="510" y="214" text-anchor="middle" class="arch-muted">the same morsel, still in L1</text>
</svg>
</figure>

By processing data in a cache-aware manner, Pivot significantly reduces the time the CPU spends waiting on memory accesses: 

<figure class="arch-figure">
<svg viewBox="0 0 920 256" role="img" aria-labelledby="memory-access-title memory-access-desc">
<title id="memory-access-title">Pivot's estimated cache access breakdown on ClickBench</title>
<desc id="memory-access-desc">A stacked bar shows the reported counter-based estimates as percentages of all loads and stores: 97.9 percent hit L1, approximately 1.09 percent are served by L2 after an L1 miss, and 1.01 percent go beyond L2. The shares sum to 100 percent, with 98.99 percent attributed to L1 or L2. L2's share is calculated as 98.99 minus 97.9, using rounded reported rates. These are estimates from hardware-counter ratios, not an exact classification of individual loads and stores. Beyond L2 can mean the shared system cache or DRAM; DRAM-only accesses were not measurable on this VM. Next to each share is the typical latency of that layer: about 1 nanosecond for L1, about 4 nanoseconds for L2, and 10 to 120 nanoseconds beyond L2.</desc>
<rect x="40" y="24" width="840" height="208" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Pivot cache access breakdown</text>
<text x="64" y="76" class="arch-muted">Estimated share of all loads and stores</text>
<!-- One common denominator: 97.9% + 1.09% + 1.01% = 100%. -->
<rect x="64" y="100" width="775.368" height="24" class="arch-bar-strong" />
<rect x="839.368" y="100" width="8.6328" height="24" class="arch-bar" />
<rect x="848.0008" y="100" width="7.9992" height="24" class="arch-inner" />
<text x="64" y="144" class="arch-tiny">0%</text>
<text x="856" y="144" text-anchor="end" class="arch-tiny">100%</text>
<rect x="64" y="164" width="12" height="12" class="arch-bar-strong" />
<text x="84" y="175" class="arch-label">L1 cache</text>
<text x="64" y="209" class="arch-title" style="font-size: 28px;">97.9%</text>
<text x="160" y="208" class="arch-label" style="font-family: var(--docs-font-display); font-size: 20px; font-weight: 500;">~1 ns</text>
<rect x="338" y="164" width="12" height="12" class="arch-bar" />
<text x="358" y="175" class="arch-label">L2 cache</text>
<text x="338" y="209" class="arch-title" style="font-size: 28px;">1.09%</text>
<text x="434" y="208" class="arch-label" style="font-family: var(--docs-font-display); font-size: 20px; font-weight: 500;">~4 ns</text>
<rect x="610" y="164" width="12" height="12" class="arch-inner" />
<text x="630" y="175" class="arch-label">Beyond L2 (“cold”)</text>
<text x="610" y="209" class="arch-title" style="font-size: 28px;">1.01%</text>
<text x="706" y="208" class="arch-label" style="font-family: var(--docs-font-display); font-size: 20px; font-weight: 500;">~10–120 ns</text>
</svg>
<figcaption>ClickBench, 43 queries, AWS c8g.4xlarge. Pivot main 253313c1, PGO; approximately 162 billion loads and stores. Engine process only: mean of hot tries 2 and 3 per query, summed across all 43 queries. Percentages are approximate shares derived from Arm PMU counters.</figcaption>
</figure>

#### Utilizing memory throughput
Over the past few years, memory bandwidth has been growing significantly faster than single-core CPU performance. As a result, many engines that were designed around the compute-to-bandwidth ratios of earlier hardware can't fully take advantage of modern DRAM speeds.

<figure class="arch-figure">
<svg viewBox="0 0 920 324" role="img" aria-labelledby="bandwidth-growth-title bandwidth-growth-desc">
<title id="bandwidth-growth-title">Memory bandwidth versus compute across Graviton generations</title>
<desc id="bandwidth-growth-desc">A line chart across three AWS Graviton generations, normalized to Graviton2. Memory bandwidth grows from 1 times on Graviton2 (c6g, 205 gigabytes per second peak) to 1.5 times on Graviton3 (c7g, 307 gigabytes per second) and 2.6 times on Graviton4 (c8g, 538 gigabytes per second). Compute performance grows from 1 times to 1.25 times and then 1.6 times over the same generations.</desc>
<rect x="40" y="24" width="840" height="276" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Memory bandwidth vs compute</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">relative to Graviton2</text>
<line x1="120" y1="236.5" x2="856" y2="236.5" class="arch-rule" />
<text x="104" y="240" text-anchor="end" class="arch-tiny">1×</text>
<line x1="120" y1="166.5" x2="856" y2="166.5" class="arch-rule" />
<text x="104" y="170" text-anchor="end" class="arch-tiny">2×</text>
<line x1="120" y1="96.5" x2="856" y2="96.5" class="arch-rule" />
<text x="104" y="100" text-anchor="end" class="arch-tiny">3×</text>
<text x="200" y="260" text-anchor="middle" class="arch-label">Graviton2</text>
<text x="200" y="278" text-anchor="middle" class="arch-muted">c6g</text>
<text x="450" y="260" text-anchor="middle" class="arch-label">Graviton3</text>
<text x="450" y="278" text-anchor="middle" class="arch-muted">c7g</text>
<text x="700" y="260" text-anchor="middle" class="arch-label">Graviton4</text>
<text x="700" y="278" text-anchor="middle" class="arch-muted">c8g</text>
<polyline points="200,236.0 450,218.5 700,192.2" fill="none" style="stroke: var(--sl-color-gray-3); stroke-width: 2;" stroke-dasharray="5 4" />
<circle cx="200" cy="236.0" r="4" style="fill: var(--sl-color-gray-3);" />
<circle cx="450" cy="218.5" r="4" style="fill: var(--sl-color-gray-3);" />
<circle cx="700" cy="192.2" r="4" style="fill: var(--sl-color-gray-3);" />
<polyline points="200,236.0 450,201.0 700,122.2" fill="none" style="stroke: var(--sl-color-gray-2); stroke-width: 2;" />
<circle cx="200" cy="236.0" r="4" style="fill: var(--sl-color-gray-2);" />
<circle cx="450" cy="201.0" r="4" style="fill: var(--sl-color-gray-2);" />
<circle cx="700" cy="122.2" r="4" style="fill: var(--sl-color-gray-2);" />
<text x="200" y="224.0" text-anchor="middle" class="arch-tiny">205 GB/s</text>
<text x="450" y="189.0" text-anchor="middle" class="arch-tiny">307 GB/s</text>
<text x="700" y="110.2" text-anchor="middle" class="arch-tiny">538 GB/s</text>
<text x="718" y="123.2" class="arch-label">Memory bandwidth</text>
<text x="718" y="139.2" class="arch-muted">2.6×</text>
<text x="718" y="193.2" class="arch-label">Compute</text>
<text x="718" y="209.2" class="arch-muted">1.6×</text>
</svg>
<figcaption>Gains AWS reports between generations.</figcaption>
</figure>

To fully utilize this rising memory throughput, pivot utilizes a handful of strategies:

##### Aggressive software prefetches
A modern CPU can only keep a limited number of instructions in flight at once, capped by the size of its [reorder buffer](https://en.wikipedia.org/wiki/Re-order_buffer) (ROB). As a result, the number of memory accesses a single core can have outstanding before its ROB fills up is often lower than the number of concurrent requests the memory subsystem could actually serve.

For example, if a hash table lookup takes 20 instructions and the ROB holds 100, a single core can have at most 5 lookups (and therefore 5 memory accesses) in flight at a time, which may leave the memory bus underutilized. In practice, the core issues those 5 requests almost immediately (the compute is negligible next to memory latency), stalls for hundreds of nanoseconds while DRAM responds, and only then moves on to the next 5.

To work around this, modern CPUs provide a [prefetch instruction](https://en.wikipedia.org/wiki/Cache_prefetching), which Pivot uses to hint to the CPU that it will soon need a given memory address. For example, when performing hash table lookups over a [morsel](#utilizing-the-cpu-cache), each lookup issues a prefetch for the bucket of the key *N* positions ahead. By the time the CPU reaches that key, its bucket is already in cache, so the lookup no longer stalls on DRAM latency:

<figure class="arch-figure">
<svg viewBox="0 0 920 326" role="img" aria-labelledby="prefetch-title prefetch-desc">
<title id="prefetch-title">Prefetching hash table buckets N keys ahead</title>
<desc id="prefetch-desc">An animation of a loop over twelve keys from a morsel, k0 to k11, above a row of hash table buckets, with N equal to 4. In every loop iteration the loop looks up key i, whose bucket is already in cache because it was prefetched earlier, and issues a prefetch for the bucket of key i plus N, which is sent to DRAM. The buckets of the keys in between were prefetched in earlier iterations and are on their way. Each step of the animation advances i by one key to the right, from k0 to k7.</desc>
<defs>
<marker id="prefetch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="278" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Prefetching N keys ahead</text>
<text x="64" y="86" class="arch-muted">Loop advances</text>
<path d="M180,82 H800" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="64" y="124" class="arch-label">Morsel keys</text>
<text x="64" y="242" class="arch-label">Hash table</text>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.125;1" values="1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="204" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="256" y="124" text-anchor="middle" class="arch-label">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="308" y="124" text-anchor="middle" class="arch-label">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="360" y="124" text-anchor="middle" class="arch-label">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="412" y="124" text-anchor="middle" class="arch-label">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="464" y="124" text-anchor="middle" class="arch-label">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="204" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="204" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M204,184 L308,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="412" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="412" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M412,184 L516,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.125;0.25;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="256" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="308" y="124" text-anchor="middle" class="arch-label">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="360" y="124" text-anchor="middle" class="arch-label">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="412" y="124" text-anchor="middle" class="arch-label">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="464" y="124" text-anchor="middle" class="arch-label">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="256" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="256" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M256,184 L204,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="464" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="464" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M464,184 L412,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.25;0.375;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="308" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="360" y="124" text-anchor="middle" class="arch-label">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="412" y="124" text-anchor="middle" class="arch-label">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="464" y="124" text-anchor="middle" class="arch-label">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="308" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="308" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M308,184 L360,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="516" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="516" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M516,184 L568,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.375;0.5;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="308" y="124" text-anchor="middle" class="arch-muted">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="360" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="412" y="124" text-anchor="middle" class="arch-label">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="464" y="124" text-anchor="middle" class="arch-label">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="360" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="360" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M360,184 L256,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="568" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="568" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M568,184 L464,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="1">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.5;0.625;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="308" y="124" text-anchor="middle" class="arch-muted">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="360" y="124" text-anchor="middle" class="arch-muted">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="412" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="464" y="124" text-anchor="middle" class="arch-label">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="412" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="412" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M412,184 L516,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="620" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="620" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M620,184 L724,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.625;0.75;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="308" y="124" text-anchor="middle" class="arch-muted">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="360" y="124" text-anchor="middle" class="arch-muted">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="412" y="124" text-anchor="middle" class="arch-muted">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="464" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="516" y="124" text-anchor="middle" class="arch-label">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="464" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="464" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M464,184 L412,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="672" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="672" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M672,184 L620,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-inner" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.75;0.875;1" values="0;1;0;0" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="308" y="124" text-anchor="middle" class="arch-muted">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="360" y="124" text-anchor="middle" class="arch-muted">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="412" y="124" text-anchor="middle" class="arch-muted">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="464" y="124" text-anchor="middle" class="arch-muted">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="516" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="568" y="124" text-anchor="middle" class="arch-label">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="516" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="516" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M516,184 L568,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="724" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="724" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M724,184 L776,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-bar" />
</g>
<g opacity="0">
<animate attributeName="opacity" dur="11.2s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.875;1" values="0;1;1" />
<rect x="180" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="204" y="124" text-anchor="middle" class="arch-muted">k0</text>
<rect x="232" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="256" y="124" text-anchor="middle" class="arch-muted">k1</text>
<rect x="284" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="308" y="124" text-anchor="middle" class="arch-muted">k2</text>
<rect x="336" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="360" y="124" text-anchor="middle" class="arch-muted">k3</text>
<rect x="388" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="412" y="124" text-anchor="middle" class="arch-muted">k4</text>
<rect x="440" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="464" y="124" text-anchor="middle" class="arch-muted">k5</text>
<rect x="492" y="104" width="48" height="30" rx="1" class="arch-inner" />
<text x="516" y="124" text-anchor="middle" class="arch-muted">k6</text>
<rect x="544" y="104" width="48" height="30" rx="1" class="arch-bar-strong" />
<text x="568" y="124" text-anchor="middle" class="arch-label" style="fill: var(--sl-color-black);">k7</text>
<rect x="596" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="620" y="124" text-anchor="middle" class="arch-label">k8</text>
<rect x="648" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="672" y="124" text-anchor="middle" class="arch-label">k9</text>
<rect x="700" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="724" y="124" text-anchor="middle" class="arch-label">k10</text>
<rect x="752" y="104" width="48" height="30" rx="1" class="arch-bar" />
<text x="776" y="124" text-anchor="middle" class="arch-label">k11</text>
<text x="568" y="158" text-anchor="middle" class="arch-title">i</text>
<text x="568" y="176" text-anchor="middle" class="arch-tiny">look up: cache hit</text>
<path d="M568,184 L464,219" class="arch-line" marker-end="url(#prefetch-arrow)" />
<text x="776" y="158" text-anchor="middle" class="arch-title">i + N</text>
<text x="776" y="176" text-anchor="middle" class="arch-tiny">prefetch: sent to DRAM</text>
<path d="M776,184 L672,219" class="arch-line" stroke-dasharray="5 3" marker-end="url(#prefetch-arrow)" />
<rect x="180" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="232" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="284" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="336" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="388" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="440" y="222" width="48" height="30" rx="1" class="arch-bar-strong" />
<rect x="492" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="544" y="222" width="48" height="30" rx="1" class="arch-inner" />
<rect x="596" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="648" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="700" y="222" width="48" height="30" rx="1" class="arch-bar" />
<rect x="752" y="222" width="48" height="30" rx="1" class="arch-bar" />
</g>
<rect x="180" y="272" width="12" height="12" class="arch-bar-strong" />
<text x="200" y="283" class="arch-label">In cache</text>
<rect x="300" y="272" width="12" height="12" class="arch-bar" />
<text x="320" y="283" class="arch-label">Prefetched, on its way</text>
</svg>
</figure>



##### Multiplexing unrelated memory accesses

We measured DRAM traffic with the memory controllers' hardware counters while
Pivot, ClickHouse, and DuckDB ran the 22 TPC-H queries at scale factor 100
(600 million `lineitem` rows) on the same machine. Pivot runs the memory bus
far harder than the others:

<figure class="arch-figure">
<svg viewBox="0 0 920 432" role="img" aria-labelledby="dram-timeline-title dram-timeline-desc">
<title id="dram-timeline-title">DRAM bandwidth over a warm pass of TPC-H</title>
<desc id="dram-timeline-desc">Three area charts, one per engine, show DRAM bandwidth from 0 to 200 gigabytes per second over a warm pass of the 22 TPC-H queries, with a dashed line at the 190 gigabytes per second measured ceiling. Each engine's pass is stretched to the same width. Pivot's trace stays close to the ceiling for most of the pass. ClickHouse's and DuckDB's stay mostly below 100 gigabytes per second.</desc>
<rect x="40" y="24" width="840" height="388" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">DRAM bandwidth over the warm pass</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">read + write, 5 ms samples</text>
<text x="64" y="76" class="arch-muted">All 22 queries back to back, each pass stretched to the same width</text>
<text x="64" y="140" class="arch-label">Pivot</text>
<line x1="170" y1="174.5" x2="856" y2="174.5" class="arch-rule" />
<line x1="170" y1="99.9" x2="856" y2="99.9" class="arch-line" style="stroke-dasharray: 4 4; stroke-width: 1;" />
<polygon points="170.0,174 170.9,128.4 171.9,120.5 172.6,120.5 173.7,119.8 174.4,120.1 175.6,119.6 176.3,119.2 177.1,119.3 178.9,119.2 179.6,119.8 180.8,119.7 181.2,118.8 182.4,120.0 183.6,119.9 184.3,119.7 185.7,119.4 186.9,119.7 187.4,119.8 188.8,119.3 189.2,119.5 190.2,110.6 191.4,119.5 192.1,120.0 193.3,119.4 194.7,120.1 195.4,120.5 196.1,119.7 197.3,120.1 199.0,120.1 199.2,120.4 200.4,120.0 201.8,120.0 202.3,119.7 203.5,119.6 204.2,119.8 205.4,120.3 206.1,119.5 207.2,120.0 208.9,119.7 209.1,119.9 210.1,120.6 211.1,119.7 212.2,120.2 213.4,120.0 214.6,120.2 215.8,119.9 216.3,119.7 217.4,119.7 218.2,119.8 219.3,119.8 221.0,119.4 221.5,120.1 222.1,120.2 223.4,119.6 224.3,120.0 225.7,119.9 226.4,119.8 227.4,119.8 228.3,120.1 229.7,119.6 230.2,119.8 231.6,119.8 232.3,111.8 233.1,173.0 234.8,102.3 235.5,103.4 236.0,103.7 237.4,104.3 238.7,102.8 239.3,172.7 240.8,117.5 241.7,107.9 242.7,107.5 243.8,107.7 244.1,107.8 245.3,107.3 246.5,107.8 247.6,111.3 248.6,102.3 249.4,102.5 250.3,102.3 251.8,102.7 252.1,102.5 253.0,102.5 255.0,96.0 255.5,102.8 256.2,102.5 257.7,102.4 258.9,102.7 259.1,102.3 260.1,102.7 261.3,102.7 262.6,103.9 263.5,103.5 264.1,103.3 265.7,96.0 266.1,161.2 267.8,110.4 268.8,104.0 269.3,104.4 271.0,113.2 271.7,110.8 272.1,111.5 273.1,112.0 274.8,112.1 275.3,111.7 276.3,112.0 277.3,112.0 278.4,112.0 279.8,112.0 280.3,111.9 281.0,112.4 283.0,96.3 283.3,96.1 284.2,172.8 285.9,133.6 286.9,107.3 287.3,107.0 288.3,107.0 289.5,106.8 291.0,107.0 291.9,105.1 292.9,105.0 293.4,105.2 294.6,104.6 295.3,104.9 296.5,105.2 297.5,105.2 298.7,105.0 299.9,104.7 300.9,105.2 301.8,104.7 302.1,105.3 304.0,96.7 304.5,105.0 306.0,105.2 307.0,105.2 307.9,105.4 308.4,105.4 309.1,104.8 310.4,96.0 311.2,173.0 312.7,129.7 313.8,107.1 314.3,106.0 316.0,106.0 316.5,105.9 317.7,105.6 318.2,105.9 319.6,106.2 320.8,105.5 321.1,106.0 322.0,106.5 324.0,172.9 324.5,137.1 325.7,105.4 326.9,105.0 327.4,105.4 328.1,106.4 329.3,106.0 330.1,105.3 331.8,105.7 333.0,106.2 333.2,105.7 334.4,106.1 335.4,106.1 336.6,106.0 337.5,106.2 338.3,106.5 339.7,105.7 340.2,106.7 341.9,106.3 342.7,105.8 343.2,105.8 344.1,106.4 345.5,111.6 346.7,105.9 347.4,104.8 349.0,104.9 350.0,96.0 350.3,96.0 351.1,173.1 352.6,138.6 353.9,112.9 354.8,108.8 355.3,108.2 357.0,108.2 357.4,108.6 358.2,109.0 359.8,113.5 360.6,102.6 361.6,102.1 362.1,102.5 363.1,102.2 364.8,102.5 365.5,102.0 366.2,101.9 367.3,102.2 368.4,102.5 369.4,101.9 370.1,101.9 371.4,102.4 372.9,101.7 373.4,102.5 374.6,101.6 375.1,102.5 376.3,102.4 377.6,102.2 378.3,101.6 379.3,102.4 380.5,102.3 381.5,103.4 382.3,96.2 383.2,172.9 384.9,116.6 385.7,112.8 386.9,112.3 387.8,112.9 388.1,112.7 389.5,112.5 390.9,112.8 391.6,112.6 392.1,112.9 393.5,120.1 394.9,101.0 395.1,103.3 396.0,103.4 397.4,103.4 398.0,119.8 399.4,102.6 400.1,103.3 401.1,103.3 402.0,109.2 403.7,117.4 404.5,117.7 405.9,117.4 406.0,117.8 407.8,126.6 408.1,121.4 409.2,121.6 410.8,104.3 411.1,100.0 412.5,103.5 413.7,103.7 414.9,103.7 415.4,103.7 416.3,104.0 417.2,121.5 418.6,121.6 419.7,99.8 420.9,103.5 421.1,103.5 422.7,103.3 423.3,103.7 424.4,104.4 425.2,110.8 427.0,134.1 428.0,107.4 428.9,109.1 429.4,110.7 430.9,109.6 431.1,109.4 432.7,109.2 433.7,110.1 434.6,109.3 435.4,110.2 436.3,107.7 437.3,107.8 438.3,110.1 440.0,108.3 440.9,109.0 441.7,109.4 442.6,108.1 443.6,109.1 444.3,110.0 445.2,109.2 446.2,109.4 447.1,110.1 449.0,110.4 449.7,109.3 450.7,108.4 451.6,110.0 452.3,109.8 453.3,109.6 454.2,110.2 455.4,109.6 456.2,109.8 457.1,110.0 458.1,109.7 459.8,110.4 460.2,109.3 461.4,110.6 462.6,109.7 463.3,110.1 464.0,109.0 465.9,108.6 466.2,117.5 467.5,96.0 468.2,96.0 469.8,96.0 470.2,96.0 471.4,96.0 472.1,96.0 473.3,96.0 474.9,107.9 475.9,103.7 476.7,103.4 477.2,103.8 478.7,104.3 479.1,104.6 480.3,104.8 481.1,104.6 482.0,104.1 483.5,104.7 484.9,104.8 485.6,104.8 486.9,104.6 487.1,104.8 488.1,104.9 489.1,104.8 490.5,104.5 491.7,104.8 492.2,104.7 493.7,104.8 494.7,104.4 495.2,104.6 496.1,104.8 497.7,110.3 498.5,103.4 499.9,104.2 500.4,96.0 501.9,105.2 502.4,104.9 503.7,110.8 504.4,96.0 505.0,96.0 506.9,162.3 507.6,105.1 508.6,104.1 509.5,118.9 510.2,173.0 511.8,96.0 512.0,156.8 513.2,173.4 514.2,173.6 515.0,173.0 516.7,173.6 517.7,173.3 518.2,173.4 519.2,173.6 520.5,173.0 521.1,173.1 522.6,173.5 523.8,173.4 524.2,173.4 525.5,173.3 526.9,108.3 527.6,106.2 528.6,106.7 529.3,107.1 530.3,107.5 531.3,107.4 532.5,107.5 533.9,107.5 535.0,107.2 535.9,107.5 536.2,107.2 537.8,104.0 538.3,103.2 539.8,103.5 540.5,103.3 541.7,96.0 542.1,172.3 543.7,131.5 544.8,108.1 545.2,108.6 546.9,116.2 547.5,115.7 548.7,115.6 549.4,115.1 550.1,114.8 551.2,115.1 552.5,116.8 553.8,115.9 554.5,115.5 555.2,115.6 556.3,115.9 557.7,115.0 558.9,115.9 559.6,115.0 560.3,115.3 561.4,115.7 562.1,115.1 564.0,115.0 564.7,116.4 565.9,115.4 566.6,114.9 567.3,116.0 568.5,116.0 569.2,115.9 571.0,115.2 571.7,116.1 572.9,115.6 573.5,115.9 574.7,115.9 575.4,115.7 576.1,123.6 577.7,130.7 578.2,130.7 579.9,96.1 580.9,96.0 581.9,96.0 582.5,96.0 583.8,96.0 584.0,96.0 585.9,106.6 586.6,103.7 587.9,103.4 588.6,103.1 589.4,103.6 590.8,103.1 591.6,103.2 592.6,103.3 593.0,103.4 595.0,103.7 595.8,104.0 596.3,104.2 597.5,102.7 598.5,103.3 599.9,124.6 600.4,122.9 601.3,106.6 602.1,173.0 603.8,106.2 604.2,103.8 605.5,103.3 606.3,102.6 607.7,103.1 608.7,103.2 609.2,103.1 610.1,102.6 611.4,102.5 612.4,103.4 613.8,103.2 614.3,102.8 615.0,102.9 616.6,103.3 617.5,103.3 618.3,125.3 619.5,96.2 620.1,172.8 621.8,129.2 622.6,116.4 623.7,111.3 624.2,113.7 625.6,113.3 626.1,114.8 627.2,146.3 628.4,120.5 629.9,124.1 630.1,138.9 631.7,140.1 632.9,96.0 633.9,96.0 634.5,96.0 635.9,173.1 636.1,172.6 638.0,173.2 638.4,173.4 639.9,173.1 640.5,173.6 642.0,173.5 642.6,173.5 643.9,173.0 644.5,139.0 645.2,102.9 646.3,103.1 647.5,103.1 648.7,102.7 649.2,103.3 650.7,104.5 651.4,104.7 652.9,104.5 653.4,104.5 654.6,104.5 655.9,105.9 656.8,104.7 657.8,105.0 658.5,104.5 659.5,104.5 660.8,104.6 661.5,104.3 662.2,104.0 663.7,105.0 664.6,96.3 665.3,173.0 666.8,116.4 667.8,107.7 668.5,110.2 669.4,137.9 670.1,136.8 671.5,138.1 672.2,138.3 673.1,138.7 674.0,138.1 675.6,138.4 676.1,138.2 677.8,138.6 678.7,138.5 679.1,138.5 680.2,138.8 681.2,138.4 682.3,138.6 683.5,138.4 684.2,138.4 685.3,138.5 686.9,138.5 687.1,138.2 689.0,120.6 690.0,118.7 690.6,119.5 691.8,119.6 692.3,119.6 693.9,119.7 694.6,119.4 695.3,119.2 696.9,119.4 697.8,107.1 698.9,105.7 699.6,104.8 700.8,104.9 701.1,104.8 702.2,136.5 703.1,137.1 704.4,137.1 705.5,137.1 706.4,137.1 707.4,137.2 708.9,96.0 709.3,96.0 711.0,96.0 711.5,96.0 712.8,96.0 713.1,96.0 714.9,96.0 715.3,96.0 716.8,96.0 717.1,96.0 718.9,140.1 719.4,104.8 720.4,104.8 721.1,104.1 722.6,104.7 723.8,104.4 724.3,104.5 725.6,104.2 726.3,104.8 727.9,104.3 728.4,104.7 729.7,104.6 730.8,104.3 731.6,104.3 732.4,104.6 733.3,104.0 734.5,104.3 736.0,104.0 736.7,104.0 737.7,104.5 738.4,104.4 739.9,104.4 740.1,104.3 741.1,104.1 742.8,104.3 743.6,104.5 744.1,104.6 745.0,136.4 746.8,163.6 747.3,135.9 748.0,102.2 750.0,103.6 750.2,103.7 752.0,102.8 752.4,103.5 753.9,103.9 754.7,103.4 755.1,103.4 756.9,103.6 757.9,103.5 758.9,103.8 759.3,103.3 760.6,103.3 761.1,104.3 762.3,103.3 763.6,103.0 764.8,96.0 765.2,97.2 766.6,172.6 767.5,173.1 768.9,173.0 770.0,173.4 770.2,173.5 771.9,173.0 772.9,113.2 773.7,109.9 774.6,109.2 775.1,109.6 776.3,109.6 777.9,122.0 778.5,122.0 779.5,105.1 780.9,105.1 781.1,105.1 782.1,121.6 783.3,102.4 784.4,105.1 785.2,105.4 786.6,108.7 787.1,108.3 788.7,108.7 789.5,108.7 790.7,109.2 791.4,109.1 792.6,109.3 793.0,109.3 794.8,109.2 795.5,109.0 796.4,109.1 797.1,109.4 798.5,109.3 799.0,109.6 800.1,119.7 802.0,122.1 802.7,121.8 803.1,121.1 804.5,120.8 805.9,121.7 806.6,121.6 807.8,121.5 808.5,121.8 809.2,121.4 810.9,121.7 811.7,121.6 812.2,121.2 813.4,121.3 814.6,121.7 815.0,121.1 816.6,116.0 817.3,121.3 818.8,112.5 819.5,109.6 820.8,109.7 821.8,109.7 823.0,109.5 823.2,109.5 824.2,109.6 825.6,109.6 826.6,109.6 827.3,109.4 828.7,109.5 829.2,109.6 830.2,109.3 831.1,109.7 832.5,109.4 833.2,109.3 834.7,109.4 835.9,109.4 836.8,109.1 837.9,96.2 838.3,96.0 839.2,96.0 840.9,96.0 841.5,96.0 842.1,148.6 843.8,127.0 844.1,127.5 845.4,152.0 846.1,151.4 847.9,153.7 848.8,150.6 849.7,112.8 850.9,112.2 851.1,112.1 852.4,112.2 853.8,96.0 854.1,142.4 855.2,173.6 856.0,173.8 856.0,174" class="arch-bar-strong" />
<text x="64" y="240" class="arch-label">ClickHouse</text>
<line x1="170" y1="274.5" x2="856" y2="274.5" class="arch-rule" />
<line x1="170" y1="199.9" x2="856" y2="199.9" class="arch-line" style="stroke-dasharray: 4 4; stroke-width: 1;" />
<polygon points="170.0,274 170.9,263.0 171.3,258.1 172.4,257.4 173.4,258.2 174.0,258.0 175.3,258.0 176.5,258.9 177.2,258.7 178.4,258.4 179.2,258.4 180.6,258.6 181.2,258.5 182.4,257.4 183.6,258.6 184.4,258.1 185.0,257.6 186.4,258.1 187.0,257.8 188.3,253.0 189.5,257.8 190.7,257.8 191.8,257.8 192.7,257.8 193.3,257.8 194.3,257.7 195.6,258.3 196.3,259.0 197.2,258.3 198.3,258.1 199.9,258.8 200.4,258.6 201.1,258.9 202.7,258.7 203.8,258.9 204.9,258.3 205.9,258.7 206.6,258.6 208.0,259.0 208.9,258.2 209.3,258.0 211.0,252.4 211.5,245.8 212.7,245.2 213.1,241.5 214.8,232.6 215.4,219.9 216.8,240.1 217.8,220.3 218.8,216.1 219.1,218.7 220.1,242.1 221.6,242.4 222.6,240.8 223.7,212.0 224.1,208.7 225.1,216.3 226.5,223.0 227.3,241.4 228.1,236.7 229.1,267.3 230.9,268.5 231.0,268.0 232.2,268.4 233.1,272.3 234.3,272.1 235.7,241.9 236.7,224.4 237.5,249.5 238.3,247.5 239.6,249.9 240.3,250.9 241.2,241.3 242.5,251.4 243.7,250.8 244.5,250.3 245.9,249.6 246.2,251.2 247.7,250.8 248.5,249.9 249.6,250.3 250.7,250.4 251.8,250.0 252.8,249.4 253.8,249.6 254.1,250.6 255.1,251.2 256.1,251.2 257.5,251.8 258.9,249.5 260.0,246.7 260.8,271.2 261.9,264.0 262.3,230.6 263.6,255.9 264.0,256.2 265.8,250.5 266.6,258.7 267.2,256.7 268.8,257.0 269.3,257.9 270.5,257.5 271.9,254.9 272.1,258.4 273.6,256.5 274.3,258.3 275.9,257.9 276.7,257.5 277.9,254.9 278.1,265.6 279.8,265.8 280.3,240.4 281.8,249.6 282.1,256.1 283.4,255.8 284.5,246.6 285.5,265.6 286.2,265.1 287.7,265.0 288.2,265.8 289.6,265.5 290.6,265.5 291.6,266.0 292.4,265.1 293.8,265.9 294.4,265.8 295.8,265.3 296.7,265.2 297.0,265.4 298.8,263.5 299.2,265.2 300.1,264.6 301.5,264.5 302.5,265.3 303.0,265.1 304.4,264.7 306.0,264.6 306.5,264.9 307.8,264.9 308.9,265.6 309.3,265.2 310.8,264.0 311.2,265.8 312.9,263.6 313.3,266.0 314.4,265.5 315.0,266.1 316.9,266.3 317.5,266.4 318.6,260.8 319.6,265.8 320.4,265.2 321.9,264.8 322.3,262.7 323.6,265.0 324.1,265.1 325.1,265.4 326.9,264.9 328.0,263.9 328.8,264.1 329.1,264.1 330.8,265.1 331.3,265.0 332.5,270.4 333.8,253.0 334.9,244.2 335.9,249.8 336.8,250.8 337.2,250.7 338.8,251.0 339.9,243.5 340.1,246.8 341.6,266.7 342.5,260.5 343.9,267.1 344.8,258.6 345.6,267.0 346.9,264.6 347.0,260.1 348.2,267.2 349.3,259.7 350.5,266.9 351.5,261.5 352.2,266.3 353.8,262.1 355.0,266.5 355.9,263.3 356.0,261.7 357.2,266.7 358.1,261.4 359.4,265.9 360.4,263.3 361.4,266.3 362.8,263.0 363.0,266.8 364.8,256.2 365.7,259.4 366.1,264.1 367.5,266.9 368.6,256.2 369.1,260.3 370.7,265.6 371.5,262.5 372.0,262.7 373.9,265.1 374.6,262.9 375.0,263.3 376.9,265.0 377.1,263.7 378.4,265.0 379.8,263.0 380.8,263.3 381.9,265.5 382.5,263.1 383.3,263.8 384.9,264.6 385.0,263.7 386.0,264.6 387.6,262.5 388.6,264.6 389.8,264.7 390.6,264.4 391.0,264.5 392.9,264.0 393.0,263.8 394.1,264.3 395.2,250.9 396.1,248.2 397.2,256.1 398.7,256.1 399.8,253.1 400.1,250.1 401.7,271.9 402.9,254.1 403.6,254.0 404.8,238.6 405.6,248.8 406.8,251.5 407.5,245.6 408.7,250.5 409.4,255.9 410.4,256.4 411.1,245.3 412.1,256.5 413.2,257.6 414.6,257.2 415.5,257.8 416.5,255.6 417.4,257.0 418.0,256.4 420.0,256.2 420.3,241.7 421.8,257.8 422.1,257.2 423.0,257.1 424.7,257.7 425.8,257.3 426.9,257.7 427.2,257.3 428.2,256.6 429.5,257.2 430.2,254.8 432.0,239.8 432.3,236.5 433.8,257.5 434.1,256.9 435.6,258.3 436.3,255.3 437.5,255.6 438.7,256.6 440.0,257.1 441.0,257.3 441.8,257.3 442.8,256.8 443.5,256.3 444.9,256.6 445.8,256.9 446.5,257.8 447.8,252.8 448.4,256.6 449.4,254.4 450.8,240.0 451.9,239.2 452.5,238.2 454.0,236.1 454.8,235.2 455.0,231.9 456.1,231.0 457.6,272.1 458.9,265.2 459.5,241.5 460.8,255.5 461.9,255.5 462.7,221.8 463.8,233.5 464.2,237.7 465.1,242.3 466.1,245.1 467.6,221.0 468.5,255.1 469.5,254.3 470.8,254.5 471.7,252.9 472.6,255.2 473.1,255.1 474.5,255.2 475.9,252.7 476.4,255.0 477.2,256.0 478.1,255.8 479.6,255.8 480.4,256.1 481.3,256.1 482.6,256.0 483.9,254.6 484.1,254.4 485.1,253.5 486.4,253.6 487.6,255.4 488.9,216.6 489.3,215.4 490.9,271.5 491.8,266.9 492.5,265.3 493.8,263.7 494.2,266.5 495.1,259.8 496.1,261.9 497.3,252.2 498.1,272.8 499.6,273.3 501.0,271.2 501.9,265.9 502.4,261.4 503.8,259.5 504.1,263.9 505.4,255.9 506.4,262.5 507.1,263.4 508.6,264.3 509.2,262.6 510.8,263.6 511.4,262.2 512.1,262.7 513.2,261.7 514.2,258.8 515.6,263.2 516.1,262.0 517.8,263.1 518.9,257.8 519.3,260.5 520.4,260.8 521.7,261.4 522.9,259.5 523.4,260.9 524.6,255.3 525.4,259.3 526.9,269.4 527.5,221.4 528.2,217.5 529.9,235.3 530.0,233.3 531.8,236.8 532.2,235.5 533.4,231.6 534.8,233.1 535.4,233.1 536.2,233.8 537.8,235.0 538.4,234.3 539.2,234.1 540.8,234.1 541.2,234.9 542.7,234.9 544.0,234.7 544.4,235.5 545.0,235.1 546.5,233.5 547.5,234.6 548.7,223.7 550.0,210.5 550.5,209.0 551.4,210.8 552.2,271.3 553.6,268.3 554.8,250.6 555.3,241.6 556.0,261.1 557.2,261.0 558.9,261.9 559.5,260.3 560.6,261.7 561.3,260.7 562.8,258.2 563.3,261.8 565.0,263.4 565.1,263.0 566.9,262.7 567.5,262.7 568.4,262.4 569.2,261.5 570.8,262.2 571.8,244.2 572.6,243.8 573.6,271.1 574.5,249.4 575.3,262.6 576.6,260.9 577.9,249.2 578.0,247.5 579.1,263.4 580.8,263.2 582.0,263.2 582.1,262.7 583.9,262.6 584.8,263.0 585.5,262.8 586.9,262.9 587.1,262.8 588.5,262.7 590.0,262.7 590.3,260.2 591.9,262.3 592.8,234.1 593.0,236.8 594.9,264.6 595.2,262.9 596.8,249.3 597.0,254.7 598.2,264.6 599.9,264.3 601.0,263.5 601.8,264.1 602.1,264.1 603.9,264.3 604.0,264.1 605.3,264.4 606.2,262.1 607.7,264.1 608.4,264.2 609.5,264.2 610.8,263.8 611.8,233.6 612.5,271.1 614.0,252.1 615.0,252.7 615.6,247.6 616.4,247.4 617.6,246.7 618.5,242.9 619.1,242.5 620.9,272.4 621.9,261.9 622.4,258.6 623.6,266.2 624.4,267.0 625.6,266.0 626.1,265.9 627.6,266.6 628.2,266.0 629.3,266.5 630.4,265.6 631.3,266.3 632.1,266.6 633.6,266.6 634.5,266.6 635.1,266.5 636.7,251.6 637.9,266.9 638.3,268.0 639.4,267.6 640.5,268.1 641.6,268.1 642.2,268.2 644.0,268.2 644.2,268.0 646.0,268.1 646.3,268.3 647.4,268.0 648.3,268.4 649.4,267.8 650.9,268.2 651.5,268.2 652.2,267.8 653.8,268.0 654.0,267.4 655.4,264.6 657.0,240.2 657.3,233.9 659.0,234.3 659.3,235.3 660.6,234.4 661.8,234.4 662.3,235.3 664.0,234.9 664.1,235.6 666.0,235.5 666.9,234.8 667.5,235.0 668.8,235.8 669.3,235.6 670.8,235.6 671.4,235.2 672.9,229.4 673.9,235.9 674.8,235.2 675.0,235.7 676.6,207.4 677.2,203.7 678.2,205.5 679.2,202.3 680.4,205.4 681.4,203.7 682.6,204.5 684.0,206.4 684.6,208.4 685.0,219.4 686.9,261.5 688.0,259.9 688.2,261.3 689.1,266.7 690.5,270.6 691.5,264.6 692.8,263.1 693.5,265.7 694.2,265.7 695.9,265.2 696.0,266.1 697.4,266.0 698.0,265.9 699.5,266.0 700.3,266.1 702.0,266.4 702.4,265.7 703.0,266.0 704.2,266.0 705.6,265.6 706.2,265.8 707.8,263.8 709.0,266.1 709.3,266.0 710.1,264.4 711.2,266.1 712.7,265.8 713.8,266.0 714.5,265.7 716.0,265.8 716.3,263.3 717.7,266.0 718.8,265.5 719.9,265.7 720.0,265.9 721.2,265.8 723.0,266.2 723.8,265.5 724.4,264.9 725.1,262.6 726.4,265.6 727.2,264.9 728.0,272.0 729.7,256.3 730.1,263.8 731.4,264.2 732.4,265.1 733.7,266.4 734.2,266.8 735.9,264.1 736.1,266.1 737.4,266.1 738.6,266.7 739.7,266.4 740.7,266.1 742.0,266.3 742.3,266.3 743.6,266.3 745.0,266.3 745.3,266.3 746.3,266.3 747.3,266.1 748.0,266.2 750.0,266.7 750.7,248.9 751.7,264.5 752.7,261.9 753.9,265.1 754.7,271.8 755.7,258.8 756.2,266.0 757.4,242.9 758.6,240.4 759.5,243.3 760.4,245.9 761.3,243.6 762.1,244.4 763.0,244.8 764.8,244.9 765.7,250.1 766.5,245.5 767.5,251.3 768.3,247.6 769.1,249.0 770.1,250.8 771.8,250.8 772.9,244.1 773.4,233.7 774.8,235.5 775.9,237.3 776.1,239.1 777.8,239.4 778.2,232.2 779.9,270.2 780.6,255.9 781.5,271.7 782.6,271.0 784.0,271.5 784.4,269.4 785.1,267.3 786.9,267.8 787.0,270.3 788.2,268.0 789.9,269.4 790.2,268.1 791.9,266.5 792.9,250.5 793.7,251.8 794.2,252.8 795.5,251.9 796.5,254.7 797.7,252.9 798.4,252.8 799.5,239.8 800.2,255.0 801.2,252.2 802.9,250.7 803.0,252.2 804.8,252.6 805.5,255.1 806.3,252.9 807.3,253.9 808.2,250.6 809.9,253.7 810.0,251.8 811.7,245.0 812.9,271.0 813.9,264.1 814.1,255.2 815.9,272.3 816.1,271.8 817.7,270.1 818.4,268.1 819.3,272.1 820.2,268.8 821.9,270.8 822.3,268.0 824.0,268.9 824.2,268.4 825.9,256.9 826.1,255.7 827.2,255.3 828.7,255.5 829.9,254.6 830.8,256.3 831.4,255.3 832.0,255.6 833.1,255.2 834.5,255.3 835.1,255.2 836.2,251.8 837.9,254.9 838.3,254.8 839.5,255.6 840.9,254.4 841.8,254.1 842.6,255.0 843.8,254.7 844.6,244.0 845.1,259.7 847.0,272.3 847.1,272.0 848.9,256.1 850.0,251.4 850.4,247.0 851.4,257.6 852.4,254.4 853.6,257.4 854.8,252.4 855.0,269.9 856.0,273.8 856.0,274" class="arch-bar" />
<text x="64" y="340" class="arch-label">DuckDB</text>
<line x1="170" y1="374.5" x2="856" y2="374.5" class="arch-rule" />
<line x1="170" y1="299.9" x2="856" y2="299.9" class="arch-line" style="stroke-dasharray: 4 4; stroke-width: 1;" />
<polygon points="170.0,374 170.7,365.5 171.9,363.9 172.7,363.3 173.7,363.4 174.2,363.1 175.4,362.7 176.9,361.7 177.2,363.5 178.5,362.9 179.4,361.8 180.7,362.5 181.5,362.6 182.2,362.8 183.6,360.4 184.8,363.3 186.0,362.7 186.8,363.7 187.8,362.2 188.7,363.8 189.0,362.9 190.7,362.6 191.3,363.5 192.7,363.0 193.9,363.1 194.5,363.7 195.3,363.8 196.1,362.1 197.2,362.5 198.9,362.8 199.2,365.2 200.5,372.3 202.0,372.5 202.8,372.1 203.8,370.3 204.9,372.9 205.2,372.5 206.4,364.0 207.6,353.8 208.9,369.6 209.9,367.7 210.9,359.5 211.3,357.5 212.4,361.1 213.1,368.4 214.4,372.7 215.8,355.0 216.8,346.0 217.1,337.6 218.8,345.8 219.7,345.3 220.9,341.0 221.6,330.6 222.8,355.3 223.7,353.9 224.4,353.3 225.9,350.1 226.7,353.1 227.4,348.9 228.6,353.9 229.6,352.1 230.3,355.1 231.8,352.3 232.6,350.8 233.5,352.3 234.2,349.5 235.5,354.0 236.4,354.2 237.3,362.5 238.6,372.4 239.9,371.7 240.4,372.0 241.9,371.8 242.9,372.9 243.0,372.6 245.0,363.1 245.5,360.0 246.3,358.7 247.5,359.4 248.7,330.8 249.8,332.5 250.0,357.4 251.3,352.9 252.8,349.3 253.9,357.5 254.2,355.0 256.0,352.2 256.8,356.2 257.7,354.8 258.4,354.0 259.9,356.2 260.4,353.2 261.4,356.2 262.7,351.2 263.5,352.6 264.6,353.3 265.2,351.1 267.0,372.2 267.8,372.7 268.7,372.4 269.3,371.2 270.6,373.2 271.3,372.7 272.9,359.3 273.1,345.8 274.8,342.3 275.1,345.2 276.9,339.0 277.1,366.9 279.0,348.0 279.5,346.0 280.6,350.6 281.3,339.1 282.5,346.1 283.3,345.8 284.2,341.9 285.4,347.1 286.3,344.9 288.0,348.4 288.6,346.8 289.3,346.2 290.7,348.2 291.3,349.6 292.8,346.1 293.6,347.3 294.2,347.4 295.3,372.8 296.3,372.2 297.8,372.3 298.1,372.0 299.4,371.8 300.7,372.2 301.5,373.0 302.9,358.1 303.5,354.8 304.5,355.3 305.8,355.6 306.4,353.8 307.8,355.7 308.8,354.0 309.9,355.0 310.3,356.2 311.1,356.3 312.6,353.7 313.0,357.8 314.9,371.9 315.0,372.1 316.6,370.8 317.7,372.5 319.0,367.7 319.1,363.5 320.9,351.8 321.2,348.8 323.0,349.7 323.5,352.4 324.9,351.8 325.6,352.6 326.3,352.2 327.5,352.1 328.2,352.3 329.7,351.9 330.0,352.8 331.4,351.7 332.8,352.3 333.3,354.1 334.3,326.5 335.9,343.2 336.7,341.3 337.6,341.1 338.4,341.9 340.0,341.6 340.5,346.6 341.1,359.1 342.8,371.8 343.8,372.4 344.3,372.1 346.0,372.3 346.9,372.2 347.0,371.8 348.4,372.1 349.7,366.7 350.8,363.5 352.0,357.9 352.7,355.0 353.2,355.5 354.9,354.0 355.7,355.2 356.9,355.6 357.6,354.6 358.3,354.8 359.5,355.6 360.5,353.5 361.6,353.1 362.1,354.9 363.2,354.3 364.8,353.5 365.3,354.9 366.6,336.9 367.3,347.2 368.7,345.8 369.0,342.5 370.5,347.0 371.1,361.4 372.9,372.5 373.9,372.3 374.5,371.9 375.2,372.2 376.6,372.3 377.2,371.8 378.6,372.5 379.9,367.8 380.2,366.0 381.4,356.3 383.0,355.8 383.5,352.9 384.5,354.8 385.4,352.6 386.2,350.5 387.4,350.6 388.8,340.1 389.7,343.9 390.3,350.1 391.8,349.4 392.1,352.1 394.0,334.5 394.9,331.9 395.2,331.8 396.2,332.1 397.1,332.5 398.2,333.0 399.1,333.3 400.7,349.3 401.7,345.0 402.8,341.1 403.9,346.3 404.2,339.4 405.6,339.3 406.4,337.0 407.5,342.0 408.6,332.4 409.1,339.0 410.0,342.3 411.8,344.0 412.4,344.1 413.9,338.2 414.3,342.6 415.4,339.9 416.9,344.3 418.0,345.5 418.7,338.8 419.2,343.6 420.2,344.8 421.6,341.4 423.0,343.0 423.5,344.3 424.1,339.3 425.6,343.4 426.6,342.4 427.5,345.2 428.2,342.7 429.7,338.8 430.0,346.3 431.7,331.2 432.4,345.8 433.9,373.2 434.0,373.1 435.0,373.3 436.9,373.2 437.2,373.2 438.4,373.1 439.8,372.9 440.1,372.9 441.8,371.7 442.5,372.8 444.0,372.2 444.1,372.1 445.3,372.6 446.5,372.4 447.1,372.0 448.5,371.6 449.1,372.5 450.9,372.3 451.2,371.5 452.7,372.7 453.1,371.5 454.7,359.0 455.9,356.2 456.4,356.2 457.1,328.8 458.8,349.4 459.6,349.0 460.7,348.0 461.1,350.0 462.3,351.3 463.4,315.9 464.9,350.0 465.7,350.5 466.2,349.7 467.7,350.1 468.3,348.0 469.1,348.8 470.7,347.9 471.7,345.6 472.0,348.4 473.5,345.9 474.2,350.3 475.9,352.9 476.7,353.2 478.0,345.6 478.7,336.8 480.0,352.6 480.5,350.7 481.7,352.8 483.0,318.8 483.4,311.9 484.2,372.7 485.8,372.5 486.3,372.7 487.4,372.7 489.0,371.8 489.5,370.9 490.4,373.1 491.5,372.5 492.7,371.6 493.9,350.2 494.3,351.2 495.6,344.9 497.0,372.4 497.4,371.6 498.9,373.1 499.6,360.5 500.7,361.1 501.1,362.2 502.7,361.0 503.6,360.6 504.0,359.4 505.5,359.5 506.9,359.2 507.7,359.7 508.3,361.9 509.0,358.5 510.2,358.8 511.5,360.0 512.4,357.5 513.4,361.2 514.9,348.7 515.8,343.0 516.7,334.2 517.9,339.2 518.3,340.6 519.1,342.6 520.2,372.3 521.6,372.3 522.7,371.2 523.0,371.6 524.5,372.3 526.0,366.1 526.8,338.2 527.1,341.1 528.8,351.8 529.6,350.8 530.2,349.4 531.7,351.2 532.5,350.7 533.6,354.3 534.4,353.7 535.4,353.4 536.6,347.5 538.0,348.9 538.6,351.1 539.1,351.1 540.1,347.1 541.5,351.4 542.0,346.8 544.0,320.9 544.3,343.4 545.5,347.1 546.7,346.5 547.2,350.2 548.6,346.6 549.0,349.2 550.2,348.6 551.9,343.6 552.4,346.6 553.3,341.2 554.9,350.2 555.2,340.3 556.0,348.6 557.2,351.0 558.8,336.2 559.4,335.4 560.3,326.0 561.7,347.8 562.0,356.0 564.0,326.5 564.6,329.2 565.8,326.7 566.4,326.3 567.6,332.0 568.0,346.0 569.4,372.3 570.4,372.5 571.6,372.4 572.0,372.4 573.4,371.4 574.1,371.7 575.8,372.5 576.1,372.5 577.5,355.6 578.3,353.0 579.8,327.2 580.0,327.7 582.0,355.3 582.5,351.2 583.5,352.0 584.6,353.2 585.4,354.0 587.0,354.2 587.4,352.6 588.8,353.0 589.9,353.2 590.8,351.9 591.7,354.6 592.3,355.5 593.3,352.8 594.6,356.0 595.3,356.9 597.0,372.7 597.5,372.5 598.9,371.3 599.2,372.1 600.6,372.3 601.5,371.7 602.7,371.7 603.6,373.0 605.0,362.4 605.7,350.7 606.4,353.4 607.0,353.5 608.8,352.1 609.4,354.1 610.2,353.4 611.9,353.9 612.9,355.2 613.0,355.0 614.0,351.8 615.2,351.9 616.2,352.2 617.9,354.1 618.8,347.6 619.3,338.9 620.8,372.8 622.0,372.2 622.5,372.2 623.9,372.5 624.8,371.5 625.8,372.0 626.9,364.8 627.9,360.1 628.3,336.6 629.6,351.6 631.0,351.9 631.7,350.5 632.4,353.1 633.5,326.6 634.5,318.2 635.8,347.3 636.3,372.5 637.4,372.6 638.4,372.4 639.5,365.8 640.8,355.6 641.4,353.1 642.1,356.8 643.0,354.2 644.1,354.8 645.6,356.5 646.1,356.4 647.0,352.1 648.2,355.5 649.2,352.6 650.5,356.1 651.2,355.0 652.0,359.0 653.5,364.1 654.4,364.2 655.2,364.1 656.1,364.2 657.5,364.2 658.4,364.2 659.9,364.2 660.0,363.4 661.9,371.5 662.1,372.1 663.9,371.8 664.1,372.2 665.1,372.9 666.9,354.3 667.2,351.0 668.8,365.3 669.0,359.4 670.5,356.8 671.6,355.4 672.3,333.5 673.2,358.7 674.0,354.1 675.8,333.6 676.4,337.2 677.6,347.2 678.3,354.3 679.4,345.6 680.8,350.1 681.8,350.6 682.7,347.7 683.1,350.1 684.8,350.2 685.4,352.8 686.1,351.4 687.1,354.3 688.9,352.9 689.7,326.7 690.9,325.8 691.6,317.1 692.2,316.2 693.7,317.3 694.5,313.0 695.9,318.0 696.8,312.4 697.9,322.6 698.5,312.8 699.9,357.3 700.3,327.7 701.3,353.3 702.4,354.6 703.4,352.6 704.4,369.9 705.2,370.6 706.5,370.6 707.1,370.6 708.9,370.0 709.3,372.4 710.3,373.0 711.5,372.6 712.7,372.0 713.7,372.1 714.1,372.5 715.8,372.2 716.6,371.5 717.1,371.5 718.4,372.0 719.9,364.6 720.8,348.1 721.9,325.5 722.5,311.5 723.9,356.4 724.2,352.2 725.4,356.1 726.0,355.3 727.5,357.1 728.9,356.9 729.9,355.4 730.7,354.8 731.0,357.2 732.7,356.8 733.4,356.7 734.2,356.7 735.2,354.8 736.2,354.5 737.1,356.6 738.5,353.9 739.7,357.9 740.4,356.5 741.7,356.9 742.4,354.2 743.3,355.4 744.7,373.2 745.7,372.7 746.4,372.4 747.2,372.0 748.8,372.0 749.5,372.4 750.5,371.8 752.0,373.3 752.5,372.6 753.6,362.8 755.0,360.7 755.7,358.4 756.2,355.5 757.4,359.9 759.0,354.9 760.0,347.9 760.8,349.1 761.1,349.2 762.4,347.7 763.4,349.9 764.9,348.3 765.7,346.9 767.0,349.1 767.4,350.5 768.2,348.6 769.3,347.2 770.3,347.2 771.8,348.6 772.4,349.0 773.0,370.7 774.9,371.9 775.3,371.7 776.2,372.3 777.7,372.4 778.3,370.4 779.8,371.9 780.1,373.1 781.2,371.0 782.9,355.2 783.5,351.5 784.4,350.6 785.3,351.2 786.4,352.2 787.1,351.9 788.0,351.3 789.4,348.6 790.9,346.1 791.5,349.3 792.7,350.3 794.0,350.1 794.4,349.1 795.4,326.9 796.6,348.8 797.3,352.7 798.8,348.9 799.8,350.0 800.6,351.6 801.4,351.3 802.1,349.2 803.2,325.9 804.4,334.8 805.7,359.1 806.3,359.1 807.9,359.1 808.7,359.1 809.9,359.1 810.2,359.0 812.0,359.0 812.9,359.1 813.6,359.0 814.5,359.1 815.7,359.1 816.9,359.1 817.6,339.8 818.6,333.2 819.1,337.0 820.9,357.5 822.0,357.4 822.9,357.5 823.6,357.5 824.6,357.4 825.1,357.5 826.8,357.5 827.8,357.1 828.9,357.4 829.1,357.1 830.3,354.6 831.6,344.1 832.2,335.9 833.7,370.1 834.5,372.9 835.8,372.5 836.9,372.6 837.8,372.2 838.3,372.1 840.0,370.7 840.1,373.2 841.8,372.2 842.9,368.5 843.6,365.4 844.2,363.8 845.9,350.7 846.9,346.2 847.0,344.4 848.5,344.8 849.4,335.1 850.9,338.9 851.6,330.0 852.6,344.5 853.0,352.5 854.1,371.6 855.1,373.9 856.0,373.9 856.0,374" class="arch-bar" />
<line x1="660" y1="72" x2="680" y2="72" class="arch-line" style="stroke-dasharray: 4 4; stroke-width: 1;" />
<text x="856" y="76" text-anchor="end" class="arch-tiny">190 GB/s measured ceiling</text>
<text x="170" y="394" class="arch-tiny">Start of pass</text>
<text x="856" y="394" text-anchor="end" class="arch-tiny">End of pass</text>
</svg>
<figcaption>TPC-H SF100, 22 queries, warm pass after one cold pass. AMD EPYC 9124 (16 cores), 8 × DDR5-4800, 125 GB. System-wide DRAM read and write CAS counts × 64 B, sampled every 5 ms with perf. The ceiling is what a STREAM-style copy sustains on this machine; sample timing jitter lets some samples pass it, and the chart clips them at 200 GB/s. Pivot reads Parquet files in place; ClickHouse reads native MergeTree tables and DuckDB its native database.</figcaption>
</figure>

Pivot runs at 100 GB/s or more for 82% of the pass, and at 150 GB/s or more
for 59% of it. ClickHouse and DuckDB spend 3% of theirs above 100 GB/s. On
average Pivot moves 142 GB/s, nearly 4× either engine:

<figure class="arch-figure">
<svg viewBox="0 0 920 256" role="img" aria-labelledby="dram-average-title dram-average-desc">
<title id="dram-average-title">Average DRAM bandwidth per engine</title>
<desc id="dram-average-desc">A bar chart of average DRAM bandwidth over the warm TPC-H pass, reads and writes combined: Pivot 142.1 gigabytes per second, ClickHouse 36.5, and DuckDB 37.3. A dashed line marks the 190 gigabytes per second measured ceiling.</desc>
<rect x="40" y="24" width="840" height="212" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Average DRAM bandwidth</text>
<text x="64" y="76" class="arch-muted">Warm pass, read + write</text>
<text x="787.5" y="104" text-anchor="middle" class="arch-tiny">190 GB/s ceiling</text>
<line x1="787.5" y1="112" x2="787.5" y2="216" class="arch-line" style="stroke-dasharray: 4 4; stroke-width: 1;" />
<line x1="169.5" y1="112" x2="169.5" y2="216" class="arch-rule" />
<text x="64" y="133" class="arch-label">Pivot</text>
<rect x="170" y="120" width="461.8" height="16" rx="1" class="arch-bar-strong" />
<text x="641.8" y="133" class="arch-label">142.1 GB/s</text>
<text x="64" y="165" class="arch-label">ClickHouse</text>
<rect x="170" y="152" width="118.6" height="16" rx="1" class="arch-bar" />
<text x="298.6" y="165" class="arch-label">36.5 GB/s</text>
<text x="64" y="197" class="arch-label">DuckDB</text>
<rect x="170" y="184" width="121.2" height="16" rx="1" class="arch-bar" />
<text x="301.2" y="197" class="arch-label">37.3 GB/s</text>
</svg>
</figure>

| Warm pass | Pivot | ClickHouse | DuckDB |
| --- | --- | --- | --- |
| Time | 20.7 s | 52.0 s | 31.2 s |
| DRAM traffic, read + write | 2,941 GB | 1,896 GB | 1,163 GB |
| Average bandwidth | 142.1 GB/s | 36.5 GB/s | 37.3 GB/s |
| Time at 150 GB/s or more | 59% | 1% | 0% |
| Average bus utilization | 46% | 12% | 12% |

Pivot moves more data through memory than the other engines on these
queries, and still finishes first: it moves that data fast enough to complete
the pass in 20.7 seconds, against 31.2 for DuckDB and 52.0 for ClickHouse.

The techniques above are what make this possible. Each core runs its own
worker, and an idle worker steals work from a busy one, so every core keeps
issuing memory reads until the query is done. Buffers come from a
pre-faulted ring on huge pages, so reads are not interrupted by page faults
or TLB misses. And because each morsel is processed while it is still in
cache, the traffic that does reach DRAM is new data being streamed in, not the
same data being read again.
