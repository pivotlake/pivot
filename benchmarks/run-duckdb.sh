#!/usr/bin/env bash
#
# run-duckdb.sh — run ClickBench queries through DuckDB for a side-by-side
# comparison with pivot-bench.
#
# The query files (qNN.sql) and schema live in benchmarks/clickbench/. DuckDB
# reads the same parquet data pivot-bench points at via --source, exposed as a
# `hits` view so the unmodified qNN.sql files run as-is.
#
# Usage:
#   ./run-duckdb.sh --source ~/hits                       # all queries, 1 run
#   ./run-duckdb.sh --source ~/hits --query 7,20          # just q07 and q20
#   ./run-duckdb.sh --source ~/hits --iterations 3        # 3 timed runs each
#   ./run-duckdb.sh --source ~/hits --iterations 3 --sleep 500   # 500ms between runs
#   ./run-duckdb.sh --source ~/hits --no-drop-caches      # skip the cache drop
#   ./run-duckdb.sh --source ~/hits --query 7 --write-expected   # write q07.tsv
#   ./run-duckdb.sh --native ~/hits.db --query 7                 # native .db, not parquet
#   ./run-duckdb.sh --source ~/hits --duckdb-process single      # all iters in one process
#
# --source accepts a directory (globbed for *.parquet), a single .parquet
# file, or an explicit glob.
#
# We always report DuckDB's own `.timer` "Run Time" (query execution only —
# process spawn and DB-open are excluded).
#
# --duckdb-process per-iteration|single  (default per-iteration)
#   per-iteration: a FRESH duckdb process per timed iteration — this is exactly
#     how ClickBench measures embedded DuckDB, so it's the leaderboard-faithful
#     default. Every "hot" run is engine-cold (OS page cache warm, but DuckDB's
#     buffer pool empty each process). ⚠️ UNFAIR for a pivot-vs-DuckDB head-to-
#     head: pivot runs its iterations against ONE warm, persistent server while
#     DuckDB is forced engine-cold every iteration — so pivot's hot is
#     engine-warm and DuckDB's is engine-cold. Use it to match ClickBench, not
#     for a symmetric engine-state comparison.
#   single: run all of a query's iterations in ONE duckdb process, so iterations
#     2+ reuse DuckDB's warm buffer pool. Symmetric (engine-warm vs pivot's warm
#     server); DuckDB looks far faster on light queries. Not the ClickBench method.
#
# --native <db> queries DuckDB's own storage format instead of parquet: it
# opens a persistent .db holding a `hits` table built (by prep-modes-data.sh)
# from the OFFICIAL ClickBench native schema, and runs the OFFICIAL ClickBench
# native queries (clickbench/duckdb-official/queries.sql) — so DuckDB-native is
# measured exactly as the public leaderboard does. Mutually exclusive with
# --source; --write-expected is parquet-only.
#
# Like ClickBench, the OS page cache is dropped once before each query (Linux
# only — needs root, via sudo), so iteration 1 is a true cold read and later
# iterations measure hot, cache-resident performance.
#
# --write-expected runs each query once and writes its rows to the suite's
# qNN.tsv in pivot-bench's exact wire format (tab-separated, no header, NULL as
# empty), so the expected file is an independent oracle rather than pivot
# grading its own output. No timing, no cache drop. Two caveats: the query must
# have a *total* ORDER BY (the comparison is exact-string, so any tie in row
# order diverges), and columns whose type mapping differs from pivot's setup.sql
# — chiefly EventDate, which this script rewrites to a real DATE — will not
# match; use pivot-bench --update-results for SELECT-*/date queries instead.

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clickbench"

source_path=""
native_db=""
queries=""
iterations=1
drop_caches=1
write_expected=0
sleep_ms=0
duckdb_process="per-iteration"

usage() {
    sed -n '3,57p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     source_path="$2"; shift 2 ;;
        --native)     native_db="$2"; shift 2 ;;
        --query)      queries="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sleep)      sleep_ms="$2"; shift 2 ;;
        --duckdb-process) duckdb_process="$2"; shift 2 ;;
        --no-drop-caches) drop_caches=0; shift ;;
        --write-expected) write_expected=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$duckdb_process" in
    per-iteration|single) ;;
    *) echo "error: --duckdb-process must be per-iteration|single (got '$duckdb_process')" >&2; usage 1 ;;
esac

if [[ -n "$native_db" ]]; then
    # Native mode: query DuckDB's own storage (a persistent .db with a `hits`
    # table loaded up front), not parquet. --source is ignored.
    [[ -f "$native_db" ]] || { echo "error: native db not found: $native_db" >&2; exit 1; }
    [[ "$write_expected" == "1" ]] && { echo "error: --write-expected reads parquet, not --native" >&2; exit 1; }
elif [[ -z "$source_path" ]]; then
    echo "error: --source (or --native) is required" >&2
    usage 1
fi

command -v duckdb >/dev/null 2>&1 || {
    echo "error: duckdb not found on PATH" >&2
    exit 1
}

# Flush the Linux page cache so the next query reads from disk, the way
# ClickBench measures a cold run. Needs root, so it goes through sudo.
flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# Resolve --source into a read_parquet() argument. A directory becomes a
# *.parquet glob; a file or explicit glob is passed through. Unused in native
# mode (the data already lives in the .db's `hits` table).
if [[ -z "$native_db" ]]; then
    if [[ -d "$source_path" ]]; then
        parquet_glob="${source_path%/}/*.parquet"
    else
        parquet_glob="$source_path"
    fi
fi

# Build the list of qNN.sql files to run.
declare -a query_files=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra ids <<< "$queries"
    for id in "${ids[@]}"; do
        id="${id#q}"                       # accept both "7" and "q07"
        printf -v stem 'q%02d' "$((10#$id))"
        f="$suite_dir/$stem.sql"
        [[ -f "$f" ]] || { echo "error: no query file $f" >&2; exit 1; }
        query_files+=("$f")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql ]] && continue   # overrides, not standalone queries
        query_files+=("$f")
    done
fi

echo "duckdb $(duckdb --version)"
if [[ -n "$native_db" ]]; then
    echo "source: native db $native_db (hits table)"
else
    echo "source: $parquet_glob"
fi
if [[ "$write_expected" == "1" ]]; then
    echo "queries: ${#query_files[@]}, writing expected .tsv (no timing)"
else
    echo "queries: ${#query_files[@]}, iterations: $iterations"
fi
echo

# ClickBench's upstream DuckDB setup. `binary_as_string=True` decodes the parquet
# string columns (stored as BLOB) as VARCHAR so `URL LIKE ...` binds, and
# `make_date(EventDate)` turns EventDate (days since epoch) into a real DATE.
#
# EventTime is left as its raw packed-seconds integer — matching ClickBench's
# upstream setup and pivot (whose planner also treats EventTime as an integer).
# So `ORDER BY EventTime` (q24/q26) sorts the raw integer rather than converting
# all ~100M rows per query. The one query that needs it as a timestamp — Q42's
# `date_trunc` — converts it inline via the `toDateTime` macro in q42-duckdb.sql
# (the same trick ClickBench uses), applied only to the rows surviving Q42's
# filter. `epoch_ms` (not `to_timestamp`) avoids TIMESTAMP WITH TIME ZONE, whose
# tz/ICU handling is ~3x slower.
#
# We mirror the OFFICIAL ClickBench DuckDB harness EXACTLY: each timed iteration
# is a SEPARATE `duckdb` process. Upstream's per-query script spawns one process
# per try (`duckdb hits.db -c ".timer on" -c "$query"`), so every run — hot ones
# included — pays DuckDB's process-spawn + DB-open (+ parquet-metadata-parse)
# cost, and the numbers line up with the public leaderboard. `iter_cmd` is that
# per-iteration command; the per-query loop appends `-c "$sql"` to it.
#
# Native: run the OFFICIAL query verbatim (queries.sql line N+1 is qNN) against
# the already-typed `hits` table (EventTime is a real TIMESTAMP) — no view/macro.
# Parquet: persist the view (+ toDateTime macro) into a throwaway .db once, the
# way upstream duckdb-parquet/load does (`duckdb hits.db -f create.sql`), then
# open it fresh each iteration with parquet_metadata_cache on (as upstream does).
official_queries="$suite_dir/duckdb-official/queries.sql"
if [[ -n "$native_db" ]]; then
    setup=""
    iter_cmd=(duckdb "$native_db" -c ".timer on")
else
    setup="CREATE VIEW hits AS
SELECT *
    REPLACE (make_date(EventDate) AS EventDate)
FROM read_parquet('${parquet_glob}', binary_as_string=True);
CREATE MACRO toDateTime(t) AS epoch_ms(t * 1000);"
    # --write-expected uses its own inline duckdb call (below); the throwaway
    # view-db is only needed for the timed path.
    if [[ "$write_expected" != "1" ]]; then
        query_db="$(mktemp -u)-hits.db"
        trap 'rm -f "$query_db" "$query_db".wal' EXIT
        duckdb "$query_db" -c "$setup" >/dev/null 2>&1
        iter_cmd=(duckdb "$query_db" -c "SET parquet_metadata_cache=true" -c ".timer on")
    fi
fi

for f in "${query_files[@]}"; do
    stem="$(basename "$f" .sql)"
    if [[ -n "$native_db" ]]; then
        # Native runs the OFFICIAL ClickBench query verbatim: queries.sql line
        # N+1 is qNN (q0 → line 1). These assume EventTime is a TIMESTAMP, which
        # the official native schema provides — so DuckDB-native is measured
        # exactly as the public leaderboard does, independent of pivot's qNN.sql.
        num=$((10#${stem#q}))
        sql="$(sed -n "$((num + 1))p" "$official_queries")"
        [[ -n "$sql" ]] || { echo "error: no official query for $stem (line $((num + 1)) of $official_queries)" >&2; exit 1; }
    else
        # Parquet: prefer a DuckDB-specific override (e.g. q42-duckdb.sql) when
        # one exists, so a query can differ for DuckDB (e.g. wrapping EventTime in
        # toDateTime) while the shared qNN.sql stays the one pivot runs. This
        # parquet view + these queries are byte-identical to upstream duckdb-parquet.
        [[ -f "$suite_dir/$stem-duckdb.sql" ]] && f="$suite_dir/$stem-duckdb.sql"
        sql="$(cat "$f")"
    fi
    echo "=== $stem ==="

    if [[ "$write_expected" == "1" ]]; then
        # Emit the result rows in pivot-bench's wire format so the .tsv is a
        # ground-truth oracle. `.mode tabs` is tab-separated with no quoting (the
        # same naive concatenation pivot's collect_tsv does), `.headers off` drops
        # the column-name row, and `.nullvalue ''` renders NULL as an empty field
        # — matching pivot's pgwire output, where a NULL column contributes
        # nothing between the tabs. Errors still surface on stderr.
        out_file="$suite_dir/$stem.tsv"
        printf '%s\n.headers off\n.nullvalue '\'''\''\n.mode tabs\n.output %s\n%s\n' \
            "$setup" "$out_file" "$sql" \
            | duckdb
        echo "  wrote $out_file ($(wc -l < "$out_file" | tr -d ' ') rows)"
        echo
        continue
    fi

    # Drop the page cache once per query — upstream drops it before the whole
    # TRIES loop, not per try — so iteration 1 reads cold and later iterations
    # measure hot, cache-resident performance (pivot-bench's cold/hot split).
    flush_page_cache
    # We report DuckDB's own `.timer` "Run Time" (query execution only — process
    # spawn and DB-open are NOT in it), grepped from each run. The difference
    # between the two process modes is DuckDB's per-process buffer pool:
    if [[ "$duckdb_process" == "single" ]]; then
        # ONE process runs all iterations, so iterations 2+ reuse DuckDB's warm
        # buffer pool (engine-warm) — a symmetric engine-state head-to-head with
        # pivot's warm server. Not how ClickBench measures; DuckDB looks much
        # faster on light queries.
        cmd=("${iter_cmd[@]}")
        for ((i = 1; i <= iterations; i++)); do cmd+=(-c "$sql"); done
        "${cmd[@]}" 2>&1 | grep -E 'Run Time|Error' || true
    else
        # per-iteration: a FRESH `duckdb` process per iteration, exactly like
        # upstream's per-try `./query`. Every run is engine-cold (OS page cache
        # warm after iter 1, but DuckDB's buffer pool empty) — the
        # leaderboard-faithful default.
        for ((i = 1; i <= iterations; i++)); do
            "${iter_cmd[@]}" -c "$sql" 2>&1 | grep -E 'Run Time|Error' || true
            if [[ "${sleep_ms:-0}" -gt 0 && $i -lt $iterations ]]; then
                sleep "$(awk "BEGIN{print $sleep_ms/1000}")"
            fi
        done
    fi
    echo
done
