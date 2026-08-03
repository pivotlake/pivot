# web

The pivotdb web dashboard: a live data-flow view (ingest receivers → tables,
with real throughput), ingest + compaction counters, system metrics
(CPU / memory / disk), per-table metadata (columns, file placement), and a SQL
console.

It is **served by `pivotdb-server` itself** - the dashboard runs in the same
process as the engine, so it reads live state directly (the catalog, the ingest
sinks' and compacter's counters, the process's own CPU/memory) and runs the
console's SQL on the same planner + dispatch pool. There is no separate backend.

This directory holds only the **frontend**: a React + TypeScript app (Vite)
whose data-flow graph uses [React Flow](https://reactflow.dev/). The server's
HTTP layer (`server/src/http.rs`) serves the built bundle and the `/api`
endpoints.

```
browser ──HTTP──> pivotdb-server  (HTTP dashboard + Postgres wire + engine)
```

## Build & run (production)

```sh
# 1. build the frontend (outputs frontend/dist/)
npm --prefix web/frontend install
npm --prefix web/frontend run build

# 2. build + run the server with the dashboard on (embeds dist/ in the binary)
cd server && cargo run -- --config ../pivot.yaml   # set `http_bind` in its `server` section
# open http://127.0.0.1:8081
```

The frontend is embedded into the server binary at build time, so a release
build is one self-contained binary. Rebuild the server after `npm run build` to
pick up frontend changes (`build.rs` re-embeds when `dist/` changes).

## Develop with hot reload

No need to rebuild the (slow) server on every UI change:

```sh
# server provides the API on :8081
pivotdb-server --config pivot.yaml   # with `http_bind: 127.0.0.1:8081` in its `server` section

# Vite serves the UI on :5173 with HMR, proxying /api to the server
npm --prefix web/frontend run dev
```

Edit React under `frontend/src` and see changes instantly at
<http://127.0.0.1:5173>.

## API (served by pivotdb-server)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/overview` | Catalog (tables, columns, file placement) + live ingest & compaction counters + CPU/mem/disk. |
| `POST /api/query` | Run SQL in-process; returns columns (with types) and rows. |
| `GET /api/health` | Liveness. |
| everything else | The frontend (SPA fallback to `index.html`). |
