# web

A live picture of a pivot server, drawn with [React Flow](https://reactflow.dev/):
the server in the middle, every connected client around it, and under the
server's name the number of queries running and the progress of the current
compaction.

- A connected client is joined to the server by a dotted line.
- A client running `COPY` streams along its line towards the server, with the
  rate it is sending at.

## What is live and what is not

- **Connections are live.** The backend in `server/` logs in to the pivot
  server, reads its
  [`system.connections`](../docs/src/content/docs/reference/system-tables.md)
  table once a second on the page's behalf, and the page draws one client per
  row: its name, user, address, whether it is idle, running a query or
  copying, and the statement it is running. The count of running queries is
  counted from the same rows. The backend's own connection is left out.
- **Memory is live.** The backend reads
  [`system.memory_blocks`](../docs/src/content/docs/reference/system-tables.md),
  grouped by NUMA node and state, in the same cycle. The server card draws
  one bar per NUMA node, split into pinned memory, the two caches and what is
  free, each named inside its part where the name fits. Above the bar is the
  amount pinned on that node, and pointing at a part raises a tooltip with
  its amount and share.
- **Compaction is simulated.** `src/simulation.ts` makes its progress up, and
  the card says so.
- **The host details** on the server card (hostname, cores, memory) are those
  of the machine running Vite, read in `vite.config.ts`: a pivot server does
  not report the machine it runs on.

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
(`--password` or `PIVOT_PASSWORD`), and the address to listen on.

## Layout of the source

- `server/`: the backend, a Rust workspace of its own. `GET /api/connections`
  answers with the rows of `system.connections` as JSON, and
  `GET /api/memory` with `system.memory_blocks` grouped by node and state.
- `src/serverConnections.ts`, `src/serverMemory.ts`: the page's side of
  those requests.
- `src/simulation.ts`: the made-up compaction progress.
- `src/graphLayout.ts`: where clients sit around the server, for a wide
  window and for a tall one.
- `src/App.tsx`: turns the connections into React Flow nodes and edges.
- `src/components/`: the server card, the client card and the connection.
