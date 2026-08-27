---
title: Why is Pivot fast?
description: How Pivot avoids unnecessary work and uses the machine efficiently.
sidebar:
  order: 2
---

Pivot is fast because its storage and execution layers are designed together. The engine tries to eliminate work before a query starts, then spreads the remaining work across the machine.

### It reads less data

The planner pushes filters and projections into the scan. Pivot can then skip:

- Partitions that cannot contain a matching value
- Files whose column statistics rule out the filter
- Parquet row groups whose min/max statistics or dictionaries rule out the filter
- Columns that the query does not use

Late materialization goes a step further: Pivot can evaluate a selective part of a query first and fetch wider columns only for rows that survive.

### It parallelizes the whole pipeline

Surviving row groups become independent units of work for Pivot's dispatch pool. Scan, decode, filter, aggregation, sorting, encoding, and upload stages can run across worker threads instead of funneling the query through a single coordinator.

Large or expensive row groups can also be split between workers, which helps prevent one piece of input from becoming the long tail of a query.

### It operates on columnar batches

Pivot executes over Arrow record batches and keeps Parquet data columnar through the scan path. Operators process many values at a time, which reduces per-row overhead and keeps the CPU focused on the columns a query actually needs.

### It improves the physical layout

Background compaction combines small files into target-sized Parquet files and can range-order their contents. Better-sized row groups reduce metadata and scheduling overhead, while ordering makes statistics more selective for common filters.

For semi-structured data, frequently occurring `VARIANT` paths can be shredded into typed Parquet leaves. Queries can read and prune those typed fields without parsing every complete document, while uncommon or mixed-type values remain available in the original variant representation.

### It reuses safe planning work

Pivot caches reusable query plans. A cached plan is accepted only while the referenced table identity and version still match, so repeated queries avoid redundant planning without running against stale metadata.

No single optimization explains every workload; performance depends on the query, data layout, storage, and available hardware.
