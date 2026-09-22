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
Pivot’s execution strategy is inspired by [Morsel driven parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf). It prioritizes cache locality by processing data in small batches (i.e. morsels) sized to fit within the CPU’s L1 cache, and tries to perform as much work as possible on each morsel while its data remains hot in the cache.

A single morsel of data is typically processed through the entire pipeline of operations that make up a query before the next morsel begins processing:

<!-- Morsel execution diagram placeholder. -->

On ClickBench, hardware-counter estimates indicate that **about 99% of Pivot's
memory accesses are served by L1 or L2**, with most hitting L1. Keeping data
hot across successive operations helps the CPU reuse it from these caches.

<figure class="arch-figure">
<svg viewBox="0 0 920 280" role="img" aria-labelledby="memory-access-title memory-access-desc">
<title id="memory-access-title">Pivot's estimated cache access breakdown on ClickBench</title>
<desc id="memory-access-desc">A stacked bar shows the reported counter-based estimates as percentages of all loads and stores: 97.9 percent hit L1, approximately 1.09 percent are served by L2 after an L1 miss, and 1.01 percent go beyond L2. The shares sum to 100 percent, with 98.99 percent attributed to L1 or L2. L2's share is calculated as 98.99 minus 97.9, using rounded reported rates. These are estimates from hardware-counter ratios, not an exact classification of individual loads and stores. Beyond L2 can mean the shared system cache or DRAM; DRAM-only accesses were not measurable on this VM.</desc>
<rect x="40" y="24" width="840" height="232" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Pivot cache access breakdown</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">≈99% served by L1 or L2</text>
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
<text x="64" y="233" class="arch-muted">Hit in L1</text>
<rect x="338" y="164" width="12" height="12" class="arch-bar" />
<text x="358" y="175" class="arch-label">L2 cache</text>
<text x="338" y="209" class="arch-title" style="font-size: 28px;">1.09%</text>
<text x="338" y="233" class="arch-muted">Hit after an L1 miss</text>
<rect x="610" y="164" width="12" height="12" class="arch-inner" />
<text x="630" y="175" class="arch-label">Beyond L2 (“cold”)</text>
<text x="610" y="209" class="arch-title" style="font-size: 28px;">1.01%</text>
<text x="610" y="233" class="arch-muted">Shared cache or DRAM</text>
</svg>
<figcaption>ClickBench, 43 queries, AWS c8g.4xlarge. Pivot main 253313c1, PGO; approximately 162 billion loads and stores. Engine process only: mean of hot tries 2 and 3 per query, summed across all 43 queries. Percentages are approximate shares derived from Arm PMU counters.</figcaption>
</figure>

L2 serves about 52% of L1 misses, which corresponds to approximately 1.09%
of all accesses. Combined with the 97.9% L1 estimate, this gives 98.99%
attributed to the two private caches.

These ratios are estimates: the
[PMU counters](https://github.com/ARM-software/data/blob/master/pmu/neoverse-v2.json)
count different kinds of events, including memory operations and cache-refill
transactions. They do not measure the exact cache hit rate of each load or
store. “Cold” here means beyond L2; those requests may still hit the shared
system cache, and this VM did not expose a usable DRAM-only counter.


#### Utilizing memory throughput
...
