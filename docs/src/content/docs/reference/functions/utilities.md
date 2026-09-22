---
title: "Utility functions"
description: Format byte counts and clear Pivot's data caches.
sidebar:
  order: 7
---

Utility functions help inspect storage and control the running engine.

| Function | Result |
| --- | --- |
| [format_bytes](#format_bytes) | A byte count formatted as text. |
| [drop_cache](#drop_cache) | Number of cache entries evicted. |

## format_bytes

```sql
format_bytes(bytes)
```

Accepts an integer byte count and returns a `VARCHAR` using binary units,
such as KiB and MiB. One KiB is 1024 bytes.

```sql
SELECT format_bytes(bytes) AS size
FROM (VALUES (1536::BIGINT)) AS input(bytes);
-- size: 1.5 KiB
```

Use it with system metadata:

```sql
SELECT name, format_bytes(bytes) AS size
FROM system.tables
ORDER BY bytes DESC;
```

## drop_cache

```sql
drop_cache()
```

Takes no arguments. Clears Pivot's compressed and decompressed memory caches,
plus its disk cache when configured. Returns a `BIGINT` count of entries
evicted across these caches.

```sql
SELECT drop_cache() AS entries_dropped;
```

The result depends on the cache contents at the time of the call.

## Related

- [System tables](/docs/reference/system-tables/)
