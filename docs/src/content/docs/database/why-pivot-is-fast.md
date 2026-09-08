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
Instead of relying on an existing general-purpose allocator, Pivot uses an internal block-based allocator for most of its memory needs. Rather than dealing with fragmentation, heap size classes, and many of the other concerns general-purpose allocators are designed around, Pivot manages memory as a collection of fixed-size 2 MB buffers.

This works well because Pivot’s memory usage mostly falls into two categories:

* Large, long-lived allocations — Parquet row groups, decoded data, decompression buffers, and similar objects. These allocations tend to live longer because they may remain in cache and be reused across queries.
* Small, short-lived allocations — temporary objects and intermediate data created while executing a query. These allocations are typically no longer needed once an operator, or the query itself, finishes.

Fixed-size 2 MB buffers fit both patterns particularly well.

For large allocations, Pivot simply combines multiple 2 MB buffers into a single logical allocation. Because Pivot controls the layers operating on this memory—decryption, decompression, decoding, and so on—those operations are designed to work directly over non-contiguous blocks. A 20 MB Parquet row group, for example, can be backed by ten separate 2 MB buffers while appearing to the rest of the engine as one logically contiguous region. There is no need to find or maintain a physically contiguous 20 MB allocation.

For small allocations, Pivot uses bump allocation on top of the same 2 MB buffers. Each query or operator can allocate temporary objects by simply advancing an offset within its current buffer. Once that buffer fills up, it grabs another 2 MB buffer and continues from offset zero. When the operator finishes, all of the buffers it used can be released at once—without individually tracking or freeing every small allocation.

Thanks to its fixed 2 MB buffer design, Pivot gets several performance benefits almost for free:

**Fewer page faults** - Pivot’s 2 MB buffers are allocated and faulted in once at startup, then kept for the lifetime of the process rather than being returned to the operating system. Acquiring a buffer is simply a pop from a per-worker free list, and releasing one is a push. There are no size classes to select, no fragmentation to manage, and no mmap or munmap calls on the allocation path (like jemalloc has):

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

**Huge pages can be utilized.** Because every buffer is exactly 2 MB, Pivot asks Linux to back the buffer ring with transparent 2 MB huge pages instead of ordinary 4 KB pages. When a query's working set spans more pages than the TLB can hold, each miss costs a multi-level walk through the page tables. With huge pages, one TLB entry covers 512 times as much memory, so operators that jump around large structures, such as the hash tables behind joins and aggregations, hit in the dTLB far more often and walk the page tables far less:

<figure class="arch-figure">
<svg viewBox="0 0 920 216" role="img" aria-labelledby="tlb-title tlb-desc">
<title id="tlb-title">TPC-H q21 with and without huge pages</title>
<desc id="tlb-desc">Two bar charts for TPC-H SF100 query 21 run hot with the buffer ring on huge pages and on ordinary pages. Query time is 2.47 seconds with huge pages and 3.13 seconds without, 21 percent faster. Data TLB page walks over a 20 second window are 1.2 billion with huge pages and 5.8 billion without, 4.8 times fewer.</desc>
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
<text x="856" y="54" text-anchor="end" class="arch-muted">20 s window</text>
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

#### Schelding: a thread per core architecture with custom internal schedling:

#### IO: direct IO with IO uring:

### It utilizes the hardware well.
