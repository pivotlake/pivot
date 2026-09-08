---
title: "Transaction behavior"
description: How Pivot handles BEGIN, COMMIT, and ROLLBACK.
sidebar:
  order: 12
---

Pivot commits each statement individually. `BEGIN`, `COMMIT`, and `ROLLBACK`
are accepted for driver compatibility, but they do not group statements into
a transaction.

:::caution[ROLLBACK does not undo writes]
Writes issued between `BEGIN` and `ROLLBACK` are already committed and remain
in place. Do not rely on these commands for multi-statement atomicity.
:::

## Example

```sql
CREATE TABLE transaction_demo (id BIGINT);
BEGIN;
INSERT INTO transaction_demo VALUES (1);
ROLLBACK;
SELECT count(*) FROM transaction_demo;
```

The count is `1`. The insert committed when that statement completed.

## Accepted syntax

```sql
BEGIN;
COMMIT;
ROLLBACK;
```

## Driver compatibility

Some PostgreSQL drivers wrap statements in a transaction automatically. Pivot
answers these commands with their usual response tags so those clients can
connect and issue statements. The responses do not imply that a transaction
was opened, committed as a group, or rolled back.

## Related

- [INSERT](/docs/reference/statements/insert/)
- [COPY](/docs/reference/statements/copy/)
- [SQL reference](/docs/reference/)
