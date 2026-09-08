---
title: "VARIANT access"
description: Read nested fields from semi-structured values and cast them to SQL types.
sidebar:
  order: 5
---

`VARIANT` stores semi-structured values. Read fields with dot notation or `->`,
then cast a field when a query needs a scalar SQL value.

| Form | Purpose |
| --- | --- |
| [Dot notation](#dot-notation) | Read a named field. |
| [Arrow notation](#arrow-notation) | Read a field using a string key. |
| [Scalar casts](#scalar-casts) | Convert a field to a SQL type. |

## Example data

The following examples use one document:

```sql
CREATE TABLE documents (doc VARIANT);
INSERT INTO documents VALUES ('{"user":{"name":"Ada","age":30}}');
```

## Dot notation

```sql
document.field
```

Reads a field as a variant value. Chain field names to access nested values.

```sql
SELECT doc.user.age AS age FROM documents;
-- age: 30
```

## Arrow notation

```sql
document->'field'
```

Reads a field using a string key. Arrow access can also be chained:

```sql
SELECT doc->'user'->'age' AS age FROM documents;
-- age: 30
```

## Scalar casts

```sql
CAST(document->'field' AS data_type)
```

Converts the selected field to a scalar type for filtering, arithmetic, or
aggregation.

```sql
SELECT CAST(doc->'user'->'name' AS VARCHAR) AS name
FROM documents
WHERE CAST(doc->'user'->'age' AS BIGINT) >= 18;
-- name: Ada
```

The field value must be compatible with the requested type. A missing field
casts to SQL `NULL`.

## Related

- [VARIANT data type](/docs/reference/data-types/#variant)
- [SELECT](/docs/reference/statements/select/)
