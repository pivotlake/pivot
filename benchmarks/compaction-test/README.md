# Compaction test scripts

Scripts used to measure the time and memory of one large compaction while the
streaming compaction was developed. They are working notes, not a maintained
suite: paths, the bucket name and the table names are the ones of that test.

## Setup

- One test machine (16 cores) with the repository at `~/pivot-merge` and these
  scripts at `~/merge-test`.
- One S3 bucket the machine's instance role can read and write. The bucket name
  is written in `run_server.sh`, `config.yaml` and `prepare_shapes.sh`.
- The server cannot use temporary role keys as it is. `sync_tree_and_build.sh`
  patches `metastore-disk/src/secrets.rs` on the machine to read
  `AWS_SESSION_TOKEN`. That change is not in the repository.
- `loggen.rs` is the loader. Put it at `bin/examples/loggen.rs` on the machine
  and build it with `cargo build --release --example loggen`.

## The test data

`create_table.sh <name>` creates a 49-column logs table (maps and arrays as
VARIANT, sorted on 8 columns, unpartitioned).

`loggen "host=127.0.0.1 port=5432 user=pivot dbname=lake" <table> <rows> <seed> <threads>`
fills it. 5,600,000 rows make about 0.9 GB stored. Every table covers the same
hour.

The compaction input is five files of about 0.9 GB (28,000,000 rows), in three
shapes:

- Fully overlapping (`adopt2/logs`): load five tables with different seeds,
  run `COMPACT <table> FINAL` on each, copy the five files under one prefix.
- Slightly overlapping (`adopt_slight/logs`) and disjoint
  (`adopt_disjoint/logs`): built by `prepare_shapes.sh` from one 28,000,000-row
  table.

## Running

From the development machine, to copy a source tree over and build:

    KEY=<ssh key> BOX=<user@address> ./sync_tree_and_build.sh <source tree>

On the test machine:

    ./run_compaction.sh <run number> <new table name> [input prefix]
    ./run_shapes.sh <first run number> adopt2 adopt_slight adopt_disjoint

`run_compaction.sh` starts the server (`run_server.sh`, compaction on,
`COMPACT_BYTES` defaults to 1g), creates a table that adopts the input files
(`with_pre_existing_parquets`), runs `COMPACT <table> FINAL` and prints the
seconds it took. It writes:

- `pool<N>.txt`: `system.memory_blocks` by state, once a second
  (`watch_pool.sh`). Peak and typical pinned memory are read from this.
- `server<N>.log`: the server log.

Stop the server after a run. Its role keys expire after about six hours and it
then fills the log with errors.

`query_during_compaction.sh <run number> <table>` times the latest-row query
on an idle server and again while the compaction runs.

## Other directories

- `probes/`: scripts that add temporary print lines to the source on the test
  machine (`*_probe.py`, `time_operators.py`) and scripts that summarise their
  output (`summarize_*.py`). They edit by exact text match and were written
  against the code of the day, so most need adjusting before they apply again.
- `parquetsink/`: scripts used on the live test server. `sink_watch.sh` writes
  one health line a minute, `time_topn.sh` times the latest-row query,
  `unregister_table.sh` removes a table from the manifest without deleting its
  files, `inspect_row_groups.py` reads only file footers and compares row-group
  ranges, `layout_probe.py` logs what the compactor's selection sees.

## What this test does not cover

It is one compaction on an idle server with generated data. Query latency
during compaction and the selection of files to compact were only exercised on
the live test server.
