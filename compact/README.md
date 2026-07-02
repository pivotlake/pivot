# compact

Background compaction of a table's small Parquet files.

Every `INSERT` flushes its rows as new Parquet files, which keeps writes
durable and immediately queryable but litters a table with small files, and
scans pay per file. The `Compacter` merges them: whenever a table's files
smaller than the target size add up to at least one target-sized output, it
rewrites that batch as one file (with full-size row groups) and swaps it into
the table in a single catalog commit.

The compacter talks to nothing but the catalog (and through it, the table log
and object store), so it can run inside the pivotdb server (the bundled
`--compact` loop) or as a separate process over the same database root.
