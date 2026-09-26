---
title: "Strings & regular expressions"
description: Search, measure, slice, and replace text.
sidebar:
  order: 2
---

These functions operate on `VARCHAR` values. Function names are
case-insensitive.

| Function | Result |
| --- | --- |
| [contains](#contains) | Whether text contains a substring. |
| [length](#length) | Character count. |
| [substring](#substring) | A character slice. |
| [regexp_full_match](#regexp_full_match) | Whether the entire string matches a pattern. |
| [regexp_replace](#regexp_replace) | Text with the first match replaced. |
| [regexp_jit_replace](#regexp_jit_replace) | First-match replacement using PCRE2 JIT. |

## contains

```sql
contains(string, substring)
```

Returns a boolean indicating whether the string column contains the constant
`substring`.

```sql
SELECT contains(text, 'ivo') AS matches
FROM (VALUES ('Pivot'), ('Lake')) AS input(text);
-- matches: true, false
```

## length

```sql
length(string)
```

Returns the number of characters in the string. `len` and `strlen` are aliases.

```sql
SELECT length(text) AS characters
FROM (VALUES ('Pivot')) AS input(text);
-- characters: 5
```

## substring

```sql
substring(string, start[, length])
```

Returns a string slice, counting positions in characters. `substr` is an
alias. `start` and `length` must be non-negative integer constants. Omitting
`length` reads to the end of the string.

Positions start at `1`. A start of `0` is also accepted: with an explicit
length, it consumes one character of that length before the beginning of the
string. Negative positions and lengths are not supported.

```sql
SELECT substring(text, 2, 3) AS piece
FROM (VALUES ('Pivot')) AS input(text);
-- piece: ivo
```

## regexp_full_match

```sql
regexp_full_match(string, pattern)
```

Returns a boolean indicating whether the complete string matches the constant
pattern. The `~`, `!~`, and `SIMILAR TO` forms use the same matcher.

```sql
SELECT regexp_full_match(text, '[a-z]+') AS matches
FROM (VALUES ('pivot'), ('pivot42')) AS input(text);
-- matches: true, false
```

The optional regex settings argument is not supported.

## regexp_replace

```sql
regexp_replace(string, pattern, replacement)
```

Returns text with the **first** matching substring replaced. Both `pattern`
and `replacement` must be constants.

```sql
SELECT regexp_replace(text, 'a', 'X') AS replaced
FROM (VALUES ('banana')) AS input(text);
-- replaced: bXnana
```

A fourth settings argument is not supported.

## regexp_jit_replace

```sql
regexp_jit_replace(string, pattern, replacement)
```

Returns text with the first match replaced, using a pattern compiled by
PCRE2 JIT. Pattern and replacement must be constants.

```sql
SELECT regexp_jit_replace(text, 'a', 'X') AS replaced
FROM (VALUES ('banana')) AS input(text);
-- replaced: bXnana
```

## Related

- [Text types](/docs/reference/data-types/#text)
- [SELECT filtering](/docs/reference/statements/select/#filtering)
