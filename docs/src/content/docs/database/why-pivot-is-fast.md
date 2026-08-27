---
title: Why is Pivot fast?
description: How Pivot avoids unnecessary work and uses the machine efficiently.
sidebar:
  order: 2
---

Although database performance comes from many different optimization strategies and architectural decisions, most fall into three main categories. To fully maximize performance, pivot tries to utilize optimizations in each one of these categories:

### It processes less data
The fastest way to process data is to avoid processing it in the first place. Pivot uses several strategies to minimize the amount of data that needs to be read and processed for each query. Some of these include:

#### Example 1 - Rowgroup pruning

Parquet stores min/max statistics for each column in every row group. When evaluating a query, Pivot can use these statistics to determine that certain row groups cannot contain matching rows and skip them entirely.

For example, consider the following query:

```sql
SELECT * FROM sales WHERE user_id=1377;
```

Any row group whose user_id range does not include 1377 can be skipped. A row group with a minimum user_id of 10 and a maximum of 200, for example, cannot possibly contain the requested user and therefore does not need to be read.

While row group pruning is common among engines that query Parquet, Pivot takes greater advantage of it by allowing users to **soft-sort** their tables. Soft sorting organizes rows within generated Parquet files to make row group statistics more selective for columns commonly used by a workload. Pivot also prioritizes compacting overlapping files, helping keep similar values clustered together and further improving pruning efficiency.

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

Together, soft sorting and overlap-aware compaction keep similar values clustered as the table evolves, improving pruning at both the file and row-group level.

This is conceptually similar to ClickHouse’s sparse primary-key index / table order by definition, but applied to Parquet files and open table formats.

#### Example 2 - Late materialization

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

For wide tables or highly selective queries, this can significantly reduce the amount of data read, decoded, and processed.

TODO: give

### It works well with the OS
Linux is an amazing operating system. It can run a wide variety of workloads with great performance and stability. However, like many low-level systems, it is difficult to build a “generalist” system or algorithm that performs optimally across very different workloads.

Because of this, several databases have explored building specialized operating systems or bypassing parts of the OS entirely, giving the database more direct control over how resources are managed.

Rather than forcing Pivot users to move to an unfamiliar or less mature operating system, Pivot tries to get the best of both worlds: taking as much control as possible over scheduling, I/O, and memory management from the OS, while still benefiting from the stability, ecosystem, and extensive set of libraries that Linux has to offer.

1. Schelding: a thread per core architecture with custom internal schedling:

2. IO: direct IO with IO uring:

3. Memory: custom memory management:

### It utilizes the hardware well.
