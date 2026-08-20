---
title: Limits
description: Sizes and counts the engine enforces.
sidebar:
  order: 2
---

| Limit | Value |
| --- | --- |
| Rows per `COPY` batch | 131,072 |
| Buffer pool | share of system memory, configurable |
| `DECIMAL` precision | 38, of which up to 18 rides a 64-bit column |
| Open datastore directory | one pivotdb process at a time |
| Supported column types | see [Data types](/docs/reference/data-types/) |
