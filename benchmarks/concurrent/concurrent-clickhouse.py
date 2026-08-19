#!/usr/bin/env python3
"""Concurrency benchmark against a running ClickHouse server.

Mirrors pivot-bench's --clients mode so the numbers are comparable: N clients
start together behind a barrier and each runs every query once per sweep,
either all in the file's canonical order or each in its own deterministic
permutation. The permutation generator matches the Rust harness exactly
(xorshift64 Fisher-Yates seeded by client index), so client k here runs the
same order as client k there.

Queries come one-per-line from clickhouse-official/queries.sql (line N is qNN).
Each client holds one persistent HTTP connection; a query's latency covers
sending it and draining the whole response body. A ClickHouse memory-limit
error (code 241) is recorded as a failed execution, like the Rust harness
records pivot's memory aborts; any other error aborts the run.
"""

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

MASK64 = (1 << 64) - 1


def xorshift_next(state):
    state ^= (state << 13) & MASK64
    state ^= state >> 7
    state ^= (state << 17) & MASK64
    return state & MASK64


def order_for_client(query_count, client_index, mode):
    order = list(range(query_count))
    if mode == "same":
        return order
    state = ((client_index + 1) * 0x9E3779B97F4A7C15) & MASK64
    for i in range(query_count - 1, 0, -1):
        state = xorshift_next(state)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def run_query(connection, host_path, query):
    start = time.monotonic_ns()
    connection.request("POST", host_path, body=query.encode())
    response = connection.getresponse()
    body = response.read()
    latency_ms = (time.monotonic_ns() - start) // 1_000_000
    if response.status == 200:
        return latency_ms, True
    if b"Code: 241" in body:
        return latency_ms, False
    raise RuntimeError(f"query failed ({response.status}): {body[:500].decode(errors='replace')}")


def run_client_pass(url, queries, order, sweeps, label, out):
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port)
    executions = []
    pass_start = time.monotonic_ns()
    for _ in range(sweeps):
        for idx in order:
            query_id, sql = queries[idx]
            latency_ms, completed = run_query(connection, "/", sql)
            if not completed:
                print(f"{label} {query_id}: aborted by the server (memory limit)")
            executions.append({"query": query_id, "latency_ms": latency_ms, "completed": completed})
    wall_ms = (time.monotonic_ns() - pass_start) // 1_000_000
    print(f"{label}: pass done in {wall_ms / 1000:.1f}s")
    out.append({"executions": executions, "wall_ms": wall_ms})


def render_report(queries, runs, wall_ms, args):
    print(f"\n=== Concurrency report: {args.clients} clients, {args.order} order ===")
    print(f"{'query':<8} {'min':>8} {'mean':>8} {'max':>8} {'failed':>6}")
    for query_id, _ in queries:
        latencies = [
            e["latency_ms"]
            for run in runs
            for e in run["executions"]
            if e["query"] == query_id and e["completed"]
        ]
        failed = sum(
            1
            for run in runs
            for e in run["executions"]
            if e["query"] == query_id and not e["completed"]
        )
        if not latencies:
            print(f"{query_id:<8} {'-':>8} {'-':>8} {'-':>8} {failed:>6}")
            continue
        print(
            f"{query_id:<8} {min(latencies):>8} {statistics.mean(latencies):>8.1f} "
            f"{max(latencies):>8} {failed:>6}"
        )
    client_walls = [run["wall_ms"] for run in runs]
    print(
        f"\nwall ({args.clients} clients): {wall_ms / 1000:.1f}s "
        f"(per-client min {min(client_walls) / 1000:.1f}s, max {max(client_walls) / 1000:.1f}s)"
    )
    total = sum(len(run["executions"]) for run in runs)
    failed = sum(1 for run in runs for e in run["executions"] if not e["completed"])
    if failed:
        print(f"under concurrent load: {failed} of {total} executions hit the memory limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--order", choices=["same", "shuffled"], default="same")
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--warmup-sweep", action="store_true")
    parser.add_argument("--url", default="http://localhost:8123")
    parser.add_argument(
        "--queries",
        default=str(
            Path(__file__).parent.parent / "clickbench" / "clickhouse-official" / "queries.sql"
        ),
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    lines = [line.strip() for line in Path(args.queries).read_text().splitlines() if line.strip()]
    queries = [(f"q{i:02}", sql) for i, sql in enumerate(lines)]
    canonical = list(range(len(queries)))

    if args.warmup_sweep:
        print(f"=== Warmup sweep ({len(queries)} queries) ===")
        warmup = []
        run_client_pass(args.url, queries, canonical, 1, "warmup", warmup)
        print(f"warmup done in {warmup[0]['wall_ms'] / 1000:.1f}s")

    print(f"\n=== Concurrent sweep: {args.clients} clients ===")
    runs = []
    barrier = threading.Barrier(args.clients + 1)
    threads = []
    for client_index in range(args.clients):
        order = order_for_client(len(queries), client_index, args.order)
        out = []
        runs.append(out)

        def client_main(order=order, out=out, label=f"client {client_index}"):
            barrier.wait()
            run_client_pass(args.url, queries, order, args.sweeps, label, out)

        threads.append(threading.Thread(target=client_main))
        threads[-1].start()
    barrier.wait()
    wall_start = time.monotonic_ns()
    for thread in threads:
        thread.join()
    wall_ms = (time.monotonic_ns() - wall_start) // 1_000_000
    print(f"concurrent sweep done in {wall_ms / 1000:.1f}s")

    flat_runs = [out[0] for out in runs]
    render_report(queries, flat_runs, wall_ms, args)
    if args.json_out:
        report = {
            "suite": "clickbench-clickhouse",
            "clients": args.clients,
            "order_mode": args.order.capitalize(),
            "sweeps": args.sweeps,
            "concurrent_wall_ms": wall_ms,
            "concurrent": flat_runs,
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
