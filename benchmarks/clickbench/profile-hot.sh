#!/usr/bin/env bash
#
# profile-hot.sh — build one PGO pivot server from a source tree and profile the
# ClickBench queries' hot runs on this machine: per-query engine stats (exec
# wall time vs. summed worker CPU), psql-measured hot timings, and flat `perf`
# profiles of the server while each query repeats. Everything lands in an output
# directory as plain text, so the machine can be stopped once it is copied out.
#
# Usage:
#   profile-hot.sh --tree ~/perf-ab/after --clickbench-dir ~/perf-ab/ClickBench \
#       --source ~/hits --pgo-subset ~/hits-pgo-subset --out /tmp/profile \
#       [--query 0,1,2] [--callgraph-query 1,13] [--runs 7] [--workers-list 0,190]
#
# --workers-list repeats the whole measurement once per worker count, each
# under its own <out>/w<N>/ directory; 0 means the server's own default. A
# non-zero count starts the server directly with a config naming it, since the
# adapter's ./start writes a config without one.
#
# --env-list repeats it once per server environment: entries separated by
# '|', each a space-separated list of KEY=VALUE pairs exported around the
# server's start, or '-' for none. Each entry gets its own <out>/e<N>-w<M>/.
#
# The build mirrors build-ab-servers.sh (instrumented build, training run on
# the PGO subset, profile-use build), so the profiled binary is the same kind
# of binary the A/B workflow times.

set -euo pipefail
trap 'echo "error: profile-hot.sh failed at line $LINENO" >&2' ERR
echo "invoked as: $0 $*" >&2

tree=""
clickbench_dir=""
source_path=""
pgo_subset=""
out_dir="/tmp/profile"
queries=""
callgraph_queries=""
runs=7
workers_list="0"
env_list="-"
record_perf=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tree)             tree="$2"; shift 2 ;;
        --clickbench-dir)   clickbench_dir="$2"; shift 2 ;;
        --source)           source_path="$2"; shift 2 ;;
        --pgo-subset)       pgo_subset="$2"; shift 2 ;;
        --out)              out_dir="$2"; shift 2 ;;
        --query)            queries="$2"; shift 2 ;;
        --callgraph-query)  callgraph_queries="$2"; shift 2 ;;
        --runs)             runs="$2"; shift 2 ;;
        --workers-list)     workers_list="$2"; shift 2 ;;
        --env-list)         env_list="$2"; shift 2 ;;
        --no-perf)          record_perf=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for req in tree clickbench_dir source_path pgo_subset; do
    [[ -n "${!req}" ]] || { echo "error: --${req//_/-} is required" >&2; exit 2; }
done

expand_tilde() {
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
tree="$(expand_tilde "$tree")"
clickbench_dir="$(expand_tilde "$clickbench_dir")"
source_path="$(expand_tilde "$source_path")"
pgo_subset="$(expand_tilde "$pgo_subset")"
out_dir="$(expand_tilde "$out_dir")"

export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin"
export NO_COLOR=1
export LC_ALL=C.UTF-8
mkdir -p "$out_dir"

host_target="$(rustc -vV | sed -n 's/^host: //p')"
llvm_profdata="$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata"

echo ">>> building the PGO server" >&2
pgo="$(dirname "$tree")/pgo-profile"
(
    cd "$tree/benchmarks"
    rm -rf "$pgo"
    mkdir -p "$pgo"
    PGO_DIR="$pgo" PGO_GEN_TARGET_DIR=target-pgogen \
        just pgo-gen-build build --release -p bin --bin pivot --features unbounded-park
    CARGO_TARGET_DIR=target-client \
        cargo build --release -p benchmarks --bin pivot-bench
    LLVM_PROFILE_FILE="$pgo/%m-%p.profraw" PIVOT_SPIN_LIMIT=0 \
        "target-client/release/pivot-bench" \
        --server-bin "target-pgogen/$host_target/release/pivot" \
        --source "$pgo_subset" --iterations 2 --skip-check >/dev/null
    "$llvm_profdata" merge -o "$pgo/merged.profdata" "$pgo"/*.profraw
    PGO_USE_TARGET_DIR=target-pgouse \
        just pgo-use-with "$pgo/merged.profdata" build --release -p bin --bin pivot --features unbounded-park
) >&2
server="$tree/benchmarks/target-pgouse/$host_target/release/pivot"
[[ -x "$server" ]] || { echo "error: expected a server at $server" >&2; exit 1; }

adapter="$clickbench_dir/pivot-parquet"
[[ -x "$adapter/benchmark.sh" ]] || { echo "error: no ClickBench pivot-parquet adapter at $adapter" >&2; exit 2; }

IFS=',' read -ra workers_counts <<<"$workers_list"
IFS='|' read -ra env_entries <<<"$env_list"
top_out_dir="$out_dir"
env_index=0
for env_entry in "${env_entries[@]}"; do
for workers in "${workers_counts[@]}"; do
(
out_dir="$top_out_dir/e$env_index-w$workers"
mkdir -p "$out_dir"
echo ">>> env='$env_entry' workers=$workers" >&2
echo "$env_entry" > "$out_dir/env.txt"
if [[ "$env_entry" != "-" ]]; then
    for pair in $env_entry; do export "${pair?}"; done
fi
export PIVOT_SERVER_BIN="$server" PIVOT_SOURCE="$source_path" \
       PIVOT_PORT=7797 PIVOT_CATALOG=/tmp/profile-catalog
cd "$adapter"
rm -rf "$PIVOT_CATALOG" "$source_path/_delta_log"
./stop >/dev/null 2>&1 || true
if [[ "$workers" == "0" ]]; then
    ./start
else
    config="/tmp/pivot-config-$PIVOT_PORT.yaml"
    cat > "$config" <<EOF
workers: $workers
server:
  bind: 127.0.0.1:$PIVOT_PORT
datastores:
  default:
    kind: pivot
    location: $PIVOT_CATALOG
    default: true
users:
  postgres:
    auth:
      method: trust
EOF
    # Detached, so no `wait` in this script can block on the server.
    setsid nohup "$server" server --config "$config" > "/tmp/pivot-server-$PIVOT_PORT.log" 2>&1 < /dev/null &
fi
for _ in $(seq 1 300); do ./check >/dev/null 2>&1 && break; sleep 1; done
./check >/dev/null 2>&1 || {
    echo "error: the server did not come up; its log:" >&2
    tail -n 40 "/tmp/pivot-server-$PIVOT_PORT.log" >&2 || true
    echo "the config it was started with:" >&2
    cat "/tmp/pivot-config-$PIVOT_PORT.yaml" >&2 || true
    exit 1
}
./load >/dev/null
# The pid that listens on the benchmark port: a server another run left
# behind must not be the one profiled.
server_pid="$(ss -ltnpH "sport = :$PIVOT_PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
[[ -n "$server_pid" ]] || { echo "error: no process listens on port $PIVOT_PORT" >&2; exit 1; }
{
    echo "server: $server"
    echo "pid: $server_pid"
    nproc
    lscpu | grep -E 'Model name|NUMA|Socket|Core|Thread' || true
    grep -E 'workers|node group|buffer pool' "/tmp/pivot-server-$PIVOT_PORT.log" || true
} > "$out_dir/machine.txt"

query_sql() { sed -n "$(($1 + 1))p" queries.sql; }
run_query() {
    psql -h 127.0.0.1 -p "$PIVOT_PORT" -U postgres -d postgres -q -v ON_ERROR_STOP=1 \
        -c '\timing on' -c "$1" 2>&1 | awk '/^Time:/{print $2}'
}
run_query_with_stats() {
    psql -h 127.0.0.1 -p "$PIVOT_PORT" -U postgres -d postgres -c "SET pivot_stats = true" -c "$1" 2>&1 \
        | grep -o "plan=.*" | sed 's/ | disk=.*http-disk-cache=[^ ]* *//'
}

total="$(wc -l < queries.sql)"
if [[ -n "$queries" ]]; then
    IFS=',' read -ra query_list <<<"$queries"
else
    query_list=($(seq 0 $((total - 1))))
fi
IFS=',' read -ra callgraph_list <<<"${callgraph_queries:-}"

: > "$out_dir/hot-timings.tsv"
: > "$out_dir/stats.txt"
for n in "${query_list[@]}"; do
    sql="$(query_sql "$n")"
    echo ">>> Q$n (workers=$workers)" >&2
    # Warm up so the decompressed cache and plan cache are populated.
    for _ in 1 2 3; do run_query "$sql" >/dev/null; done
    timings=""
    for _ in $(seq 1 "$runs"); do
        timings="$timings $(run_query "$sql")"
        sleep 0.2
    done
    printf 'Q%s%s\n' "$n" "$timings" >> "$out_dir/hot-timings.tsv"
    {
        printf 'Q%-3s' "$n"
        for _ in 1 2 3; do printf ' | %s' "$(run_query_with_stats "$sql")"; done
        echo
    } >> "$out_dir/stats.txt"

    (( record_perf )) || continue
    # Flat profile of the server while the query repeats.
    sudo -n perf record -q -p "$server_pid" -F 1999 -o "/tmp/perf-q$n.data" -- sleep 600 >>"$out_dir/perf-record.log" 2>&1 &
    perf_job=$!
    sleep 0.3
    for _ in $(seq 1 "$runs"); do run_query "$sql" >/dev/null; done
    sudo -n pkill -INT -x perf || true
    wait "$perf_job" || true
    sudo -n chown "$(id -u)" "/tmp/perf-q$n.data" || true
    perf report -i "/tmp/perf-q$n.data" --no-children --sort symbol --stdio -g none 2>/dev/null \
        | grep -v '^#' | grep -v '^$' | head -80 > "$out_dir/perf-q$n.txt" || true
    rm -f "/tmp/perf-q$n.data"

    for cg in "${callgraph_list[@]}"; do
        [[ "$cg" == "$n" ]] || continue
        sudo -n perf record -q -p "$server_pid" -F 299 --call-graph dwarf,16384 \
            -o "/tmp/perf-cg-q$n.data" -- sleep 600 >>"$out_dir/perf-record.log" 2>&1 &
        perf_job=$!
        sleep 0.3
        for _ in $(seq 1 "$runs"); do run_query "$sql" >/dev/null; done
        sudo -n pkill -INT -x perf || true
        wait "$perf_job" || true
        sudo -n chown "$(id -u)" "/tmp/perf-cg-q$n.data" || true
        perf report -i "/tmp/perf-cg-q$n.data" --children --sort symbol --stdio -g none 2>/dev/null \
            | grep -v '^#' | grep -v '^$' | head -150 > "$out_dir/perf-cg-children-q$n.txt" || true
        perf report -i "/tmp/perf-cg-q$n.data" --no-children --sort symbol --stdio -G 2>/dev/null \
            | grep -v '^#' | grep -v '^$' | head -400 > "$out_dir/perf-cg-callers-q$n.txt" || true
        rm -f "/tmp/perf-cg-q$n.data"
    done
done

./stop >/dev/null 2>&1 || true
pkill -x pivot 2>/dev/null || true
for _ in $(seq 1 120); do ps -C pivot >/dev/null 2>&1 || break; sleep 1; done
)
done
env_index=$((env_index + 1))
done
out_dir="$top_out_dir"
./stop >/dev/null 2>&1 || true
echo ">>> done: $out_dir" >&2
