---
title: "Utility functions"
description: Format byte counts and evict the in-memory compressed cache.
sidebar:
  order: 7
---

Utility functions help inspect storage and control the running engine.

| Function | Result |
| --- | --- |
| [format_bytes](#format_bytes) | A byte count formatted as text. |
| [drop_cache](#drop_cache) | Number of compressed-cache regions evicted. |

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

Takes no arguments. Evicts the in-memory compressed cache and returns a
`BIGINT` count of regions dropped. Calling it changes the running engine's cache
state.

```sql
SELECT drop_cache() AS regions_dropped;
```

The result depends on the cache contents at the time of the call.

## Related

- [System tables](/docs/reference/system-tables/)
- [Server cache configuration](/docs/reference/configuration/#disk-cache)
