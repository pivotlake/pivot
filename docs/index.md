# Pivot

<div class="pivot-hero">
  <p class="pivot-kicker">Analytical SQL over Delta tables</p>
  <p class="pivot-lead">
    Query and write data on local storage or S3 through the PostgreSQL wire
    protocol.
  </p>
  <p class="pivot-actions">
    <a class="pivot-button pivot-button-primary" href="getting-started/">Run your first query</a>
    <a class="pivot-button" href="architecture/">See how Pivot works</a>
  </p>
</div>

!!! warning "Early-stage software"

    Pivot is under active development. SQL coverage is incomplete, and
    configuration and storage compatibility may change. It is not ready for
    production use.

## One server, familiar clients

Pivot exposes the PostgreSQL v3 wire protocol. Use `psql`, JDBC,
`tokio-postgres`, or another PostgreSQL client instead of learning a custom
query API.

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## Storage where the data lives

A datastore can be a local directory or an S3 prefix. Pivot keeps table data in
Parquet files, tracks each table with a Delta log, and can query tables from
multiple datastores in one statement.

## Built for analytical queries

Pivot currently supports the core path from table creation and inserts through
filters, projections, joins, grouped aggregates, ordering, and limits. Plans
are executed by a parallel, thread-per-core dataflow engine.

## Start here

- [Build Pivot and run a first query](getting-started.md)
- [Configure local and S3 datastores](configuration.md)
- [Review the supported SQL surface](sql.md)
- [Understand the main components](architecture.md)
