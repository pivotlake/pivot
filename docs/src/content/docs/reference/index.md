---
title: "SQL reference"
description: Find supported SQL commands, functions, operators, and data types.
---

Use this reference to look up Pivot's supported SQL syntax and behavior.
Applications connect through the PostgreSQL wire protocol; SQL support is
specific to Pivot.

## Statements

Read data with [SELECT](/docs/reference/statements/select/), write rows with
[INSERT](/docs/reference/statements/insert/) or [COPY](/docs/reference/statements/copy/),
and define tables with [CREATE TABLE](/docs/reference/statements/create-table/).

[Browse all statements →](/docs/reference/sql-statements/)

## Functions and operators

Find [aggregates](/docs/reference/functions/aggregates/),
[string and regex functions](/docs/reference/functions/strings/),
[date functions](/docs/reference/functions/datetime/), and other expressions.
Each category includes signatures, argument restrictions, and examples.

[Browse functions and operators →](/docs/reference/functions/)

## Data types

Choose column and expression types, including integers, decimals, timestamps,
and semi-structured values. The reference distinguishes types that can be
stored in tables from types available only in expressions and results.

[Browse data types →](/docs/reference/data-types/)

## System tables

Read catalog, file, and memory metadata through the read-only tables of the
`system` schema.

[Browse system tables →](/docs/reference/system-tables/)

## Beyond SQL

- [CLI reference](/docs/reference/cli/) - `pivot open`, `pivot server`, and shell commands.
- [Configuration file](/docs/reference/configuration/) - every key of the server's YAML file, with
  sub-pages for [datastores and storage credentials](/docs/reference/server/datastores/) and
  [users and authentication](/docs/reference/server/authentication/).

## Compatibility

A SQL statement can parse successfully and still require an unsupported plan,
expression, or type. Individual command and function pages describe the
supported forms and their restrictions.

`BEGIN`, `COMMIT`, and `ROLLBACK` are accepted for driver compatibility, but
statements commit individually. Read [transaction behavior](/docs/reference/statements/transactions/)
before relying on a client's transaction handling.
