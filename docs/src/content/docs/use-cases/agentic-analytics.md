---
title: Agentic analytics
description: Let agents query open lake data directly with a local SQL engine.
sidebar:
  order: 2
---

Agents often need to inspect schemas, test hypotheses, and refine queries as
part of a larger task. Pivot can run beside the agent and query the same open
data used by the rest of the platform.

### How Pivot fits

1. Store analytical tables in Delta Lake on S3 or another supported object
   store.
2. Give the agent scoped credentials for the data it may access.
3. Open the datastore with `pivot open s3://bucket/prefix`.
4. Let the agent inspect metadata and run SQL in the local shell.

The engine, planner, catalog, and dispatch workers run in one process. The
agent does not need to provision a database server or wait for data to be
copied into a separate system.

### Shared data, shared semantics

Local analysis and server queries use the same Pivot engine. An agent can
develop a query locally against the source of truth, while applications and BI
tools use a long-running Pivot server over the Postgres wire protocol.

Because the tables use an open format, other engines can continue to read and
write the same data. Pivot does not become a new data silo.

### Current fit

Pivot is currently best for supervised agents, experiments, and local
analysis. Use narrowly scoped object-store credentials, and assume SQL features
not listed in the [reference](/docs/reference/) are unavailable.
