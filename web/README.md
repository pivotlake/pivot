# web

Three pages about a running pivot server.

**Server** is a live picture of it, drawn with
[React Flow](https://reactflow.dev/): the server in the middle, every
connected client around it, and under the server's name the number of queries
running and how its buffer pool is used.

- A connected client is joined to the server by a dotted line.
- A client running `COPY` streams along its line towards the server, with the
  rate it is sending at.

**Tables** lists the server's tables. Opening one shows its data files in two
panes, because the server's compacter treats a partition's files in two
different ways by their size.

**Query** runs a statement on the server and shows its result: an editor, a
list of the server's tables that each open with their first hundred rows, and
the result as a table. The editor colours SQL and completes keywords and the
names of the server's tables and columns, which the backend reads from
`system.columns`. The backend runs these on a session of its own, named
`pivot-web query`, so a slow statement does not hold up the other pages, and
the Server page shows that session as a client running the statement. A
result is cut at its first thousand rows. Whoever can open the page can run
any statement the backend's user may, so the backend listens on localhost
unless told otherwise.

## Small files

Files below half the target size (`compact_bytes`) are merged for their size
alone, whatever keys they hold: files are taken in order until they total the
merge target, and that group is one merge. A pile short of the target may
still merge once it holds `compact_min_files` files.

The pane draws that pile: each file a block as wide as its share of the merge
target, a full track for every merge that is ready, and one more for the
files still waiting, with how far they are from both thresholds. Only file
sizes go into this, so `src/compaction.ts` applies the server's own rule
(`datastore-pivotlake/src/compact`) and the count of merges ready is what a
routine round would start.

The server does not report a datastore's compaction settings, so the backend
is told them: `--compact-bytes` and `--compact-min-files`, which default to
1 GiB and 10.

## Large files

Files at least half the target size are merged for their layout: a group of
files of a sorted table whose keys all interleave is rewritten in key order
into files that do not overlap.

The pane draws the files in the table's sort order: each file a bar from
where its first row falls to where its last does. The widest files are placed
first, each in the highest row where it overlaps nothing already there, so
files that span everything pile up at the top and narrow ones settle beneath
them. Two bars one above the other are two files holding keys from the same
stretch of the sort order, which a layout merge rewrites into files side by
side; large files that are fully sorted make a single row.

For a table sorted by two columns the line is the first column's range, with
each of its values given a stretch along which the second column runs. So a
file holding every deployment for ten minutes spans the whole line, and a
file holding one deployment for a day is a short bar inside that
deployment's stretch. A file holding several values of the first column is
drawn between the middles of its two end values' stretches, since its bounds
do not say how much of either it holds; two files that share only an end
value then meet without overlapping, which is how the compacter treats them.

The page does not predict layout merges. The server decides them from the
bounds of each file's row groups, read from the files' footers, which
`system.table_files` does not carry; guessing from the files' own bounds
gives merges the server never runs.

## What is live and what is not

- **Connections are live.** The backend in `server/` logs in to the pivot
  server, reads its
  [`system.connections`](../docs/src/content/docs/reference/system-tables.md)
  table once a second on the page's behalf, and the page draws one client per
  row: its name, user, address, whether it is idle, running a query or
  copying, and the statement it is running. The count of running queries is
  counted from the same rows. The backend's own connection is drawn apart
  from the clients, as a greyed-out fixture, and is not counted: it would
  otherwise show as one more client that is always mid-query.
- **Memory is live.** The backend reads
  [`system.memory_blocks`](../docs/src/content/docs/reference/system-tables.md),
  grouped by NUMA node and state, in the same cycle. The server card draws
  one bar per NUMA node, split into pinned memory, the two caches and what is
  free, each named inside its part where the name fits. Above the bar is the
  amount pinned on that node, and pointing at a part raises a tooltip with
  its amount and share.
- **Tables and their files are live**, read from `system.tables` and
  `system.table_files` every few seconds while the tables page is open.
- **The server card's name and machine** come from the backend. A pivot
  server does not report the machine it runs on, so the backend knows it only
  when the server is on its own machine (a loopback address): the card then
  shows that machine's hostname, cores and memory. For a server anywhere else
  the card is named by the host it is reached at, and shows no machine.
- **The version** on the card is the workspace's, read from the root
  `Cargo.toml` when the page is built.

When the backend or the pivot server cannot be reached, the page says why and
shows no clients, rather than the last ones it read.

## Develop

Requires Node 22.12 or newer and the Rust toolchain the repository pins.

```sh
# The backend, on 127.0.0.1:5175, reading the pivot server on 127.0.0.1:5432.
cd server
cargo run -- --pivot-port 5432

# The page, on http://localhost:5174. Vite proxies /api to the backend.
npm install
npm run dev
npm run build      # type-check and write a static bundle to dist/
```

`cargo run -- --help` lists the backend's flags: the pivot server's host and
port, the user to log in as (`pivot` by default), its password
(`--password` or `PIVOT_PASSWORD`), and the address to listen on. Every flag
can also be set in the environment.

## As one process, and as an image

Given `--static-dir`, the backend serves the built page as well as the
requests, so it is the whole of the web page and Vite is not needed:

```sh
npm run build
cd server
cargo run --release -- --static-dir ../dist --listen 127.0.0.1:8080
```

`Dockerfile` builds an image that runs exactly that, listening on port 8080.
Its context is the repository root:

```sh
docker buildx build -f web/Dockerfile -t pivot-web .
docker run -p 8080:8080 -e PIVOT_HOST=pivot.internal pivot-web
```

Loopback inside a container reaches nothing, so the image has to be told
where the pivot server is with `PIVOT_HOST` (and `PIVOT_PORT`, `PIVOT_USER`,
`PIVOT_PASSWORD` as needed), unless the container shares the host's network.
The Deploy Web Dashboard workflow (`.github/workflows/deploy-dashboard.yml`)
builds it for amd64 and arm64 and pushes it as `pivotlake/pivot-web`. It
versions the image like the server's: a candidate is `<version>-rc<sha>`, a
real release is the workspace version and also moves `latest`, and each
published version gets a `web-v<version>` git tag.

## Layout of the source

- `server/`: the backend, a Rust workspace of its own. `GET /api/connections`
  answers with the rows of `system.connections` as JSON,
  `GET /api/memory` with `system.memory_blocks` grouped by node and state,
  `GET /api/tables` with `system.tables`, `GET /api/tables/{id}/files`
  with one table's rows of `system.table_files`, `GET /api/columns` with
  `system.columns`, and `POST /api/query` with the result of the statement it
  is sent.
- `src/serverConnections.ts`, `src/serverMemory.ts`, `src/tables.ts`: the
  page's side of those requests.
- `src/compaction.ts`: which small files a compaction round would merge.
- `src/keyScale.ts`: placing a column's bounds along a line.
- `src/graphLayout.ts`: where clients sit around the server, for a wide
  window and for a tall one.
- `src/App.tsx`: the header and which page is open.
- `src/ServerPage.tsx`: turns the connections into React Flow nodes and edges.
- `src/TablesPage.tsx`: the table list and one table's files.
- `src/QueryPage.tsx`: the statement editor and its result.
- `src/components/`: the server card, the client card, the connection, the
  small-file pile, the key-range map and the SQL editor.
