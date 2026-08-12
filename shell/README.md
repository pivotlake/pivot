# pivot shell

`pivot shell` is an interactive, local Pivot SQL shell. It embeds the planner,
catalog, Delta datastore, and dispatch workers in one process. It does not
connect to `pivotdb-server` and does not require `psql`.

## Running

```sh
cargo run --release -p shell --bin pivot -- shell ./pivot-data
```

The installed binary is named `pivot`:

```sh
cargo install --path shell
pivot shell ./pivot-data
```

The CLI uses every available dispatch worker and assigns 50% of physical
memory to the dispatch buffer pool. There are no connection or resource
options.

## Datastore directory

Every `pivot shell` invocation requires the path of one local Delta datastore.
The directory is created when it does not exist. Tables, schemas, and inserted
data remain in that directory after `\q`, Ctrl+D, and subsequent invocations.

Only one Pivot process may open a local datastore directory at a time. Pivot
holds an exclusive lock on `.pivot.lock` for as long as the datastore is open
and reports the lock owner's PID when another process tries to use it.

## Interaction

SQL may span several lines and executes at a semicolon. Several pasted
statements execute in order and stop at the first error.

Supported shell commands:

```text
\q                 quit
quit;              quit
\timing            toggle statement timing
\timing on|off     set statement timing
\h, \help           show this help
```

Ctrl+C cancels a running statement. At the prompt it clears the current input
without ending the session. Command history exists only in memory and is not
written to disk.

Results use psql's aligned layout and are written directly to the terminal.
