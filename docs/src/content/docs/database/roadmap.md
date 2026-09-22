---
title: Roadmap
description: What is available in Pivot today and what the project is working toward.
sidebar:
  order: 3
---
Pivot is still an early-stage product. As with any product at this stage, the initial releases required a deliberate focus on the features and capabilities considered most important for getting Pivot into users’ hands.

This section is intended to provide visibility into where Pivot is heading and the features and capabilities that are expected to be added over the short to medium term.

Pivot was created to improve the ecosystem around open data formats and make it possible to build and operate an open data architecture without compromising on performance. The direction outlined here reflects that goal, but it is not set in stone.

If you are using Pivot, considering using it, or simply have thoughts about where the project should go, we would love to hear them. If you disagree with any of the priorities outlined here or think something important is missing, please open a GitHub issue at |fillme| or start a discussion in the community Slack.

## Planned enhancements:
### Format support
- **Deletion support** - Pivot currently focuses on append-only workloads. Adding support for deletes will allow Pivot to work with datasets that are updated or modified over time.
- **Unity catalog support** - Integrate with Unity Catalog as a new datastore.

### Execution enhancements
- **Spill to disk** - Allow memory-intensive operators such as GROUP BY, JOIN, and ORDER BY to spill intermediate state to disk when their working set exceeds the memory available to the query.
- **MPP execution** - Support distributing a single query across multiple Pivot instances, allowing larger queries to use the CPU, memory, and I/O capacity of an entire cluster. This is a longer-term goal.

### Management / visility
- **Built in web console** - Provide an easy way to inspect and manage a running Pivot deployment, including queries, resource utilization, caches, datastores, and other engine internals.
- **EXPLAIN ANALYZE** - Extend query plans with runtime statistics, making it easier to understand where time and resources are spent during execution and to diagnose query performance.
