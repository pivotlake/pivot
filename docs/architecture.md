# Architecture

Pivot is a single server process with a PostgreSQL endpoint, SQL planner,
parallel execution engine, catalog, and storage backends.

```text
PostgreSQL client
       |
       v
pivotdb-server
  wire protocol and authentication
       |
       v
DuckDB-based planner
       |
       v
Pivot dataflow compiler
       |
       v
thread-per-core dispatch workers
       |
       v
Delta metadata and Parquet files
  local filesystem or S3
```

## Server

`pivotdb-server` owns the PostgreSQL wire endpoint and optional HTTP dashboard.
It loads the YAML metastore, authenticates connections, creates query
transactions, and returns Arrow results over the PostgreSQL protocol.

## Planning

Pivot uses an embedded DuckDB planner to parse, bind, and optimize SQL against a
Pivot catalog. The resulting logical plan is translated into Pivot operators.
Unsupported operators and expressions fail during this translation.

DuckDB plans queries, but does not execute them and does not own the data.

## Execution

The `dispatch` crate compiles Pivot operators into a parallel dataflow. It runs
one pinned worker per configured core, favors worker-local queues, and uses work
stealing to balance uneven stages.

The execution path is columnar and uses Arrow arrays between operators.

## Catalog and transactions

The catalog combines every configured datastore. Exactly one datastore is the
default for unqualified names; the others are attached as databases.

A query opens a snapshot from each datastore it touches. Table creation and
inserts are staged on the query transaction, committed durably, and then
published to the process-wide catalog.

## Storage

Each datastore has one object-store root, either a local directory or an S3
prefix. A `_pivot_manifest.json` file records which tables exist and where they
live. Each table then uses:

- A Delta log for its schema, versions, and active files
- Parquet files for columnar data
- Optional partition and sort metadata for pruning

Relative table paths stay under the datastore root. A query can read across
multiple roots through qualified table names.

## Background maintenance

Each datastore periodically refreshes its in-memory table set from storage.
Optional compaction merges small files and must be enabled in only one process
for a given datastore.
