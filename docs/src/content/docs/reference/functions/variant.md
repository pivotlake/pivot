---
title: "VARIANT access"
description: Read nested fields from semi-structured values and cast them to SQL types.
sidebar:
  order: 5
---

`VARIANT` stores semi-structured values. Read fields with dot notation or `->`.
Field access returns a `VARIANT`, which Pivot renders as JSON text in query
results. Cast the field to a SQL type for comparisons, arithmetic, or
aggregation, or to return a typed value such as `INT`.

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
(document.field)::data_type
```

Field access returns a `VARIANT`, even when the field contains a number or
string. You must cast it to a SQL type such as `INT` or `VARCHAR` before
comparing it with a value of that type.

For example, cast the age to `INT` to filter for adults:

```sql
SELECT (doc.user.age)::INT AS age
FROM documents
WHERE (doc.user.age)::INT >= 18;
-- age: 30
```

The field value must be compatible with the requested type. A missing field
casts to SQL `NULL`.

## Related

- [VARIANT data type](/docs/reference/data-types/#variant)
- [SELECT](/docs/reference/statements/select/)
