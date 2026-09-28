#!/usr/bin/env bash
#
# bench-ab.sh — A/B performance comparison of two pivotdb source trees, measured
# through the ClickBench pivot-parquet harness.
#
# Runs entirely on the benchmark box. It takes one prebuilt pivot binary per
# side ("before" and "after") and hands each to the ClickBench harness (via
# PIVOT_SERVER_BIN), which times the query set the
# faithful ClickBench way: for every query it restarts the server and drops the
# OS page cache, then runs it BENCH_TRIES times. Per query the first try is the
# cold number and the min of the rest is hot. The two runs are then diffed; if
# any query's hot time regressed past the threshold, the script exits non-zero.
#
# It is meant to be launched detached and polled from a short-lived ssh session,
# so it writes its PID and emits a sentinel on every exit path.
#
# Usage:
#   bench-ab.sh \
#     --before-bin <pivot> --after-bin <pivot> \
#     --clickbench-dir ~/perf-ab/ClickBench \
#     --source ~/hits --pgo-subset ~/hits-pgo-subset \
#     --iterations 3 --regression-pct 5 --report /tmp/ab-report.txt \
#     [--before-label <sha>] [--after-label <sha>] \
#     [--query 7,20]     # ClickBench query numbers (0-based), empty = all
#     [--server-env 'PIVOT_X=1 PIVOT_Y=true']   # env applied to both sides
#     [--sleep-between-queries 0.5]  # seconds of pause between a query's tries

set -uo pipefail

pid_file="${PID_FILE:-/tmp/bench-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== BENCH-AB COMPLETE exit=$? ==="' EXIT
set -e

before_bin=""
after_bin=""
clickbench_dir=""
source_path=""
pgo_subset=""
iterations=3
regression_pct=5
report="/tmp/ab-report.txt"
before_label="before"
after_label="after"
queries=""
duckdb_data=""
server_env=""
sleep_between_queries=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        # Prebuilt pivot binaries, one per side. This script does not
        # build: it used to run `just pgo-clean` + `just pgo-gen` + `just
        # pgo-use` per tree, which meant two full PGO builds per comparison and
        # a shared PGO dir that had to be wiped between them. Build with
        # `just setup-bench` / `just bench-build` and pass the results here.
        --before-bin)     before_bin="$2"; shift 2 ;;
        --after-bin)      after_bin="$2"; shift 2 ;;
        --clickbench-dir) clickbench_dir="$2"; shift 2 ;;
        --source)         source_path="$2"; shift 2 ;;
        --pgo-subset)     pgo_subset="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --report)         report="$2"; shift 2 ;;
        --before-label)   before_label="$2"; shift 2 ;;
        --after-label)    after_label="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        # Optional DuckDB reference: directory holding the partitioned
        # hits_*.parquet. When set, DuckDB is also timed (once) as a baseline.
        --duckdb-data)    duckdb_data="$2"; shift 2 ;;
        # Extra environment for both server builds and harness runs, given as
        # space-separated KEY=VALUE pairs. Applied to both sides identically so
        # the comparison stays a like-for-like A/B (e.g. PIVOT_* runtime knobs).
        --server-env)     server_env="$2"; shift 2 ;;
        # Seconds the harness pauses between consecutive tries of a query
        # (fractions allowed). The server reclaims its buffers after a query
        # finishes; without a pause that work runs inside the next try's
        # timing and shows up as hot-time noise. Applied to every side alike.
        --sleep-between-queries) sleep_between_queries="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_bin after_bin clickbench_dir source_path pgo_subset; do
    if [[ -z "${!req}" ]]; then
        echo "error: --${req//_/-} is required" >&2
        exit 2
    fi
done

# Paths may arrive with a leading ~ (a launcher can't expand it against this
# box's home), so expand it here against $HOME.
expand_tilde() {
    # The "~" patterns match a literal leading tilde in the argument.
    # shellcheck disable=SC2088
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
before_bin="$(expand_tilde "$before_bin")"
after_bin="$(expand_tilde "$after_bin")"
for side_bin in "$before_bin" "$after_bin"; do
    [[ -x "$side_bin" ]] || { echo "error: $side_bin is not an executable file" >&2; exit 2; }
done
clickbench_dir="$(expand_tilde "$clickbench_dir")"
source_path="$(expand_tilde "$source_path")"
pgo_subset="$(expand_tilde "$pgo_subset")"
[[ -n "$duckdb_data" ]] && duckdb_data="$(expand_tilde "$duckdb_data")"

# `sleep` takes a non-negative decimal; anything else would abort the run
# only once it reaches the first query, after the long build, so check now.
if [[ ! "$sleep_between_queries" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "error: --sleep-between-queries expects seconds (e.g. 0.5), got '$sleep_between_queries'" >&2
    exit 2
fi

# cargo / just / llvm-profdata live under the login home but not on the
# non-interactive ssh PATH, so add them explicitly.
export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
export NO_COLOR=1

# Extra environment requested for the run. Exported into this process so it is
# inherited by everything downstream: the PGO profiling run (so the profile is
# generated under the same configuration it is later measured under) and the
# harness server runs. Both sides see the identical set, keeping the A/B fair.
if [[ -n "$server_env" ]]; then
    for kv in $server_env; do
        export "${kv?}"
    done
fi

adapter="$clickbench_dir/pivot-parquet"
[[ -x "$adapter/benchmark.sh" ]] || { echo "error: no ClickBench pivot-parquet adapter at $adapter" >&2; exit 2; }

# If a query subset was requested, build a filtered queries file (ClickBench
# query numbers are 0-based line offsets into queries.sql) and remember which
# original number each kept line maps back to, for labelling the report.
queries_file=""
declare -a query_labels=()
if [[ -n "$queries" ]]; then
    queries_file="$(mktemp)"
    : >"$queries_file"
    IFS=',' read -ra want <<<"$queries"
    for n in "${want[@]}"; do
        line=$(sed -n "$((n + 1))p" "$adapter/queries.sql")
        [[ -n "$line" ]] || { echo "error: no ClickBench query $n in queries.sql" >&2; exit 2; }
        printf '%s\n' "$line" >>"$queries_file"
        query_labels+=("$n")
    done
else
    # All queries: label by their 0-based line offset.
    n=0
    while IFS= read -r _; do query_labels+=("$n"); n=$((n + 1)); done <"$adapter/queries.sql"
fi

# Run the ClickBench harness for a server binary ($1), writing its raw output to
# $2. $3/$4 are a dedicated port and a fresh catalog dir so the two sides never
# collide. The harness restarts the server and drops caches per query itself.
run_harness() {
    local bin="$1" out="$2" port="$3" catalog="$4"
    rm -rf "$catalog"
    # CREATE TABLE commits a Delta log into the table directory, which for this
    # suite is --source itself, and it outlives both the catalog wipe above and
    # the run. A later CREATE TABLE then fails with "version 0 already exists".
    # The adapter's ./load hides that twice over: its grep drops the message
    # because the message contains "already exists", and `|| true` drops the
    # exit code, so the load reports success against a table that was never
    # created and every query fails with an error ./query sends to /dev/null.
    # The symptom is a report of nothing but null timings.
    rm -rf "$source_path/_delta_log"
    (
        cd "$adapter"
        export PIVOT_SERVER_BIN="$bin" PIVOT_SOURCE="$source_path" \
               PIVOT_PORT="$port" PIVOT_CATALOG="$catalog" BENCH_TRIES="$iterations" \
               BENCH_SLEEP_BETWEEN_QUERIES="$sleep_between_queries"
        [[ -n "$queries_file" ]] && export BENCH_QUERIES_FILE="$queries_file"
        ./benchmark.sh
    ) >"$out" 2>&1 || true   # a nonzero exit (e.g. the QPS watchdog) is fine; we read the timings
    # Make sure the server is down before the other side reuses the box.
    ( cd "$adapter"; PIVOT_PORT="$port" ./stop >/dev/null 2>&1 || true )
    # The adapter's stop is an async kill, and a server whose buffer pool
    # filled up (the concurrent phase gets it to the full budget, most of the
    # machine's RAM) can take a while to hand that memory back. The next
    # side's server pre-faults its whole pool at startup, so starting it while
    # the old one is still tearing down races the two footprints and the
    # kernel OOM-kills the booting server. Memory is released before the pid
    # leaves the process table, so an empty table means teardown is done.
    local teardown_waited=0
    while ps -C pivot >/dev/null 2>&1; do
        if (( teardown_waited >= 120 )); then
            echo "warning: pivot still in the process table after ${teardown_waited}s; starting the next side anyway" >&2
            break
        fi
        sleep 1
        teardown_waited=$((teardown_waited + 1))
    done
}

# Run the DuckDB (parquet, partitioned) ClickBench adapter once, writing its raw
# output to $1. The adapter reads hits_*.parquet from its own directory, so the
# partitioned dataset ($duckdb_data) is symlinked in for the run and removed
# after. For a query subset, the DuckDB dialect queries are filtered at the same
# 0-based indices as the pivot side (queries correspond positionally).
run_duckdb_harness() {
    local out="$1"
    local dd="$clickbench_dir/duckdb-parquet-partitioned"
    [[ -x "$dd/benchmark.sh" ]] || { echo "error: no duckdb-parquet-partitioned adapter at $dd" >&2; return 1; }

    local n=0 f
    find "$dd" -maxdepth 1 -type l -name 'hits_*.parquet' -delete 2>/dev/null || true
    for f in "$duckdb_data"/hits_*.parquet; do
        [[ -e "$f" ]] || { echo "error: no hits_*.parquet in $duckdb_data" >&2; return 1; }
        ln -sf "$f" "$dd/"; n=$((n + 1))
    done
    echo "  linked $n partitioned parquet files into the DuckDB adapter" >&2

    local duck_qfile=""
    if [[ -n "$queries" ]]; then
        duck_qfile="$(mktemp)"
        local idx line
        for idx in "${query_labels[@]}"; do
            line=$(sed -n "$((idx + 1))p" "$dd/queries.sql")
            printf '%s\n' "$line" >>"$duck_qfile"
        done
    fi
    (
        cd "$dd"
        # duckdb is not on the non-interactive PATH; expose the CLI so the
        # adapter's ./install sees it and skips a network install.
        export PATH="$PATH:$HOME/.duckdb/cli/latest:$HOME/.duckdb/cli/1.5.3"
        export BENCH_TRIES="$iterations" BENCH_SLEEP_BETWEEN_QUERIES="$sleep_between_queries"
        [[ -n "$duck_qfile" ]] && export BENCH_QUERIES_FILE="$duck_qfile"
        ./benchmark.sh
    ) >"$out" 2>&1 || true
    find "$dd" -maxdepth 1 -type l -name 'hits_*.parquet' -delete 2>/dev/null || true
}

# Profile a handful of queries on the AFTER build, on this box, so the log
# says where its time goes and not only how much: `perf stat` counters and a
# `perf record` symbol breakdown per query, printed to stdout (the workflow
# keeps the full log). Best effort throughout: a box without perf, or without
# a counter, skips that part rather than failing the run.
profile_after() {
    local bin="$1" port="$2" catalog="$3"
    local adapter="$clickbench_dir/pivot-parquet"
    if ! command -v perf >/dev/null 2>&1; then
        sudo apt-get install -y linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1 \
            || sudo apt-get install -y linux-tools-common linux-tools-aws >/dev/null 2>&1 || true
    fi
    command -v perf >/dev/null 2>&1 || { echo ">>> profile: perf is not available on this box; skipping"; return 0; }
    echo ">>> profiling AFTER on this box ($(uname -m), $(nproc) cpus)"
    (
        cd "$adapter"
        export PIVOT_SERVER_BIN="$bin" PIVOT_SOURCE="$source_path" PIVOT_PORT="$port" PIVOT_CATALOG="$catalog"
        ./start >/dev/null 2>&1
        for _ in $(seq 1 300); do ./check >/dev/null 2>&1 && break; sleep 1; done
        ./load >/dev/null 2>&1 || true
        local pid
        pid="$(ps -C pivot,pivotdb-server -o pid=,args= | grep -- "-$port\." | awk '{print $1}' | head -1)"
        [[ -n "$pid" ]] || { echo "profile: no server pid found"; exit 0; }
        local qfile
        qfile="$(mktemp)"
        for n in 1 19 37 42 13 32 22 4; do
            sed -n "$((n + 1))p" queries.sql >"$qfile"
            ./query <"$qfile" >/dev/null 2>&1 || true
            echo "--- profile Q$n: $(cut -c1-110 "$qfile")"
            echo "    timing: $(./query <"$qfile" 2>&1 >/dev/null | tail -1)s"
            # perf attaches to every worker thread before it counts or
            # samples, which on a big box takes longer than a hot query
            # runs, so perf starts first, the query runs once it is
            # attached, and an interrupt ends the measurement.
            measure() {
                sudo "$@" >/tmp/ab-perf.out 2>&1 &
                local perf_pid=$!
                sleep 2
                ./query <"$qfile" >/dev/null 2>&1 || true
                sudo kill -INT "$perf_pid" 2>/dev/null || true
                wait "$perf_pid" 2>/dev/null || true
            }
            # Generic event names, one group per line: a name the box's PMU
            # lacks then costs only that line, and `cycles` is spelled out
            # because some boxes alias it to an uncore PMU as well.
            for events in task-clock,context-switches,page-faults,cpu-cycles,instructions,branch-misses \
                          stall_frontend,stall_backend l1d_cache_refill,l2d_cache_refill,l3d_cache_refill \
                          ls_dmnd_fills_from_sys.dram_io_near,de_no_dispatch_per_slot.backend_stalls; do
                measure perf stat -e "$events" -p "$pid"
                grep -E '^\s+[0-9,.]+\s+[a-z]|not (counted|supported)' /tmp/ab-perf.out | sed 's/^/    /' || true
            done
            # Time-based sampling: it needs no PMU event and so profiles the
            # same way on every box.
            measure perf record -e cpu-clock -F 4000 -p "$pid" -o /tmp/ab-prof.data
            grep -vE '^\s*$' /tmp/ab-perf.out | tail -1 | sed 's/^/    perf record: /' || true
            sudo perf report -i /tmp/ab-prof.data --no-children --sort sym --stdio -g none --percent-limit 1.5 2>&1 \
                | grep -E '^\s+[0-9.]+%|[Ee]rror|[Ff]ail' | head -30 | cut -c1-220 || true
        done
        rm -f "$qfile"; sudo rm -f /tmp/ab-prof.data
        ./stop >/dev/null 2>&1 || true
    ) || true
}

# How the AFTER build scales with its worker count on this box, and where a
# hot query's CPUs sit idle: hot times (best of six) per worker count, then a
# system-wide timeline of one hot query in 200us buckets that splits the
# samples into worker compute, worker spin, other server threads, and idle.
# Best effort, printed to stdout like the profile above.
scaling_after() {
    local bin="$1" port=7797 catalog=/tmp/ab-cat-scaling
    local adapter="$clickbench_dir/pivot-parquet"
    (
        cd "$adapter"
        export PIVOT_SOURCE="$source_path" PIVOT_PORT="$port" PIVOT_CATALOG="$catalog"
        local config=/tmp/ab-scaling.yaml pid qfile
        qfile="$(mktemp)"
        start_with_workers() {
            mkdir -p "$catalog"
            { [[ -n "$1" ]] && echo "workers: $1"
              printf 'server:\n  bind: 127.0.0.1:%s\ndatastores:\n  default:\n    kind: pivot\n    location: %s\n    default: true\n    compact: false\nusers:\n  postgres:\n    auth:\n      method: trust\n' "$port" "$catalog"
            } >"$config"
            nohup "$bin" server --config "$config" >/tmp/ab-scaling.log 2>&1 &
            pid=$!
            for _ in $(seq 1 300); do ./check >/dev/null 2>&1 && break; sleep 1; done
            ./load >/dev/null 2>&1 || true
        }
        stop_server() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
        best_of() {
            local best=""
            for _ in 1 2 3 4 5 6; do
                local t
                t="$(./query <"$qfile" 2>&1 >/dev/null | tail -1)"
                [[ -z "$best" ]] || awk -v a="$t" -v b="$best" 'BEGIN{exit !(a<b)}' && best="$t"
            done
            echo "$best"
        }
        echo ">>> scaling: hot best-of-6 ms per worker count ($(nproc) cpus)"
        for workers in "" 96 48 24; do
            start_with_workers "$workers"
            local line="    workers=${workers:-default}:"
            for n in 0 1 6 19 40 24 13 32 22; do
                sed -n "$((n + 1))p" queries.sql >"$qfile"
                ./query <"$qfile" >/dev/null 2>&1 || true
                line="$line Q$n=$(awk -v t="$(best_of)" 'BEGIN{printf "%.1f", t*1000}')"
            done
            echo "$line"
            stop_server
            rm -rf "$catalog"
        done
        start_with_workers ""
        local server_comm
        server_comm="$(cut -c1-15 /proc/"$pid"/comm)"
        for n in 1 6 13; do
            sed -n "$((n + 1))p" queries.sql >"$qfile"
            ./query <"$qfile" >/dev/null 2>&1 || true
            ./query <"$qfile" >/dev/null 2>&1 || true
            sudo perf record -a -e cpu-clock -F 5000 -o /tmp/ab-timeline.data -- \
                bash -c "sleep 0.3; ./query <'$qfile' >/dev/null 2>&1; sleep 0.3" >/dev/null 2>&1 || true
            echo "--- timeline Q$n (200us buckets: work/spin/other-server/idle cpus-equivalent, of $(nproc))"
            sudo perf script -i /tmp/ab-timeline.data -F comm,time,ip,sym 2>/dev/null | awk -v server="$server_comm" -v cpus="$(nproc)" '
                {
                    comm = $1; t = $2 + 0; sym = $4
                    for (i = 5; i <= NF; i++) sym = sym " " $i
                    bucket = int(t * 5000)
                    # A tickless idle cpu takes no samples, so idle is what
                    # the busy classes leave of the cpu count.
                    if (comm == "swapper") next
                    else if (comm == server) class = (sym ~ /Worker.*run|spin|park|yield|futex|schedule|pause/) ? "spin" : "work"
                    else if (comm ~ /^tokio/) class = "other"
                    else class = "rest"
                    count[bucket " " class]++
                    total[bucket]++
                    if (class == "work" || class == "other") busy[bucket] = 1
                }
                END {
                    first = ""; last = ""
                    for (b in busy) { if (first == "" || b + 0 < first + 0) first = b; if (last == "" || b + 0 > last + 0) last = b }
                    if (first == "") exit
                    for (b = first - 2; b <= last + 2; b++) {
                        # 5000 Hz over 200us is one sample per busy cpu per bucket
                        printf "    %+6.1fms work=%3d spin=%3d other=%2d idle=%3d\n", (b - first) * 0.2, count[b " work"], count[b " spin"], count[b " other"], cpus - total[b]
                    }
                }' | head -120
        done
        stop_server
        rm -rf "$catalog" "$qfile"
        sudo rm -f /tmp/ab-timeline.data
    ) || true
}

# Where a hot query's time goes on this box, per worker count: the server's own
# compile/exec split (its pivot_stats notice) against the client's total, and a
# scheduler trace of a couple of hot queries summarized as the gaps and wake-up
# fan-out between the query arriving and its reply. Best effort, printed to
# stdout like the profile above.
stages_after() {
    local bin="$1" port=7797 catalog=/tmp/ab-cat-stages
    local adapter="$clickbench_dir/pivot-parquet"
    (
        cd "$adapter"
        export PIVOT_SOURCE="$source_path" PIVOT_PORT="$port" PIVOT_CATALOG="$catalog"
        local config=/tmp/ab-stages.yaml pid
        start_with_workers() {
            mkdir -p "$catalog"
            { [[ -n "$1" ]] && echo "workers: $1"
              printf 'server:\n  bind: 127.0.0.1:%s\ndatastores:\n  default:\n    kind: pivot\n    location: %s\n    default: true\n    compact: false\nusers:\n  postgres:\n    auth:\n      method: trust\n' "$port" "$catalog"
            } >"$config"
            # A second argument is extra server environment (KEY=VALUE ...).
            env $2 nohup "$bin" server --config "$config" >/tmp/ab-stages.log 2>&1 &
            pid=$!
            for _ in $(seq 1 300); do ./check >/dev/null 2>&1 && break; sleep 1; done
            ./load >/dev/null 2>&1 || true
        }
        stop_server() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
        stage_medians() {
            local sql="$1"
            { echo "SET pivot_stats = 1;"; echo '\timing on'; for _ in $(seq 1 12); do echo "$sql;"; done; } \
                | psql -h 127.0.0.1 -p "$port" -U postgres -d postgres -o /dev/null 2>&1 \
                | python3 -c '
import re, sys, statistics
text = sys.stdin.read()
stats = re.findall(r"compile=([0-9.]+)ms exec=([0-9.]+)ms", text)[-10:]
times = [float(t) for t in re.findall(r"Time: ([0-9.]+)", text)][-10:]
if stats and times:
    med = lambda values: statistics.median(values)
    print("compile %.2f exec %.2f total %.2f" % (med([float(c) for c, _ in stats]), med([float(e) for _, e in stats]), med(times)))
else:
    print("no timings")
'
        }
        # Samples every CPU while a query repeats and prints, per 200us of the
        # query, how many cores ran operator work, how many polled idle in the
        # worker loop, and the top work symbols.
        cpu_timeline() {
            local n="$1" label="$2" sql
            sql="$(sed -n "$((n + 1))p" queries.sql)"
            sql="${sql%;}"
            for _ in 1 2 3; do psql -h 127.0.0.1 -p "$port" -U postgres -d postgres -q -o /dev/null -c "$sql" >/dev/null 2>&1; done
            sudo perf record -a -F 20000 -o /tmp/ab-cpu.data -- bash -c "for _ in 1 2 3 4 5; do sleep 0.1; psql -h 127.0.0.1 -p $port -U postgres -d postgres -q -o /dev/null -c \"$sql\"; done" >/dev/null 2>&1 || true
            sudo perf script -i /tmp/ab-cpu.data -F tid,cpu,time,ip,sym 2>/dev/null | c++filt >/tmp/ab-cpu.txt || true
            echo "--- cpu timeline Q$n ($label): cores per 200us, averaged over the runs"
            sudo python3 - "$pid" 20000 200 /tmp/ab-cpu.txt <<'PY' || true
import collections, os, re, sys

pid, freq, bucket_us, path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
names = {t: open(f'/proc/{pid}/task/{t}/comm').read().strip() for t in os.listdir(f'/proc/{pid}/task')}
kinds = {t: ('tokio' if n.startswith('tokio') else ('worker' if t != pid and n == names[pid] else 'other')) for t, n in names.items()}
IDLE = re.compile(r'Worker.*(run|clear_dirty|park)|maybe_finish|run_ready_cpu_work|try_steal|try_recv|deliver_ready_reads|'
                  r'completions|drain|retain|sched_yield|yield_now|schedule|syscall|el0_svc|futex|pop$|is_empty|spin')


def shorten(sym):
    sym = re.sub(r'\[[0-9a-f]{16}\]', '', sym)
    sym = re.sub(r'::h[0-9a-f]{16}$', '', sym)
    while True:
        stripped = re.sub(r'(\w)<[^<>]*>', r'\1', sym)
        stripped = re.sub(r'<([^<>]*?)(?: as [^<>]*)?>::', r'\1::', stripped)
        if stripped == sym:
            break
        sym = stripped
    parts = [p for p in sym.split('::') if p and not p.startswith('{')]
    return '::'.join(parts[-3:])[:60]


samples = []
for line in open(path):
    m = re.match(r'\s*(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+([0-9a-f]+)\s+(.*)', line)
    if m and m.group(1) in names:
        samples.append((float(m.group(3)), m.group(1), m.group(5).strip()))
samples.sort()
windows, current = [], []
for s in samples:
    if current and s[0] - current[-1][0] > 0.02:
        windows.append(current)
        current = []
    current.append(s)
if current:
    windows.append(current)
windows = [w for w in windows if any(kinds.get(tid) == 'worker' for _, tid, _ in w)]
print('    %d query windows' % len(windows))
per_cpu = freq * bucket_us / 1e6 * max(len(windows), 1)
work = collections.defaultdict(collections.Counter)
busy = collections.defaultdict(collections.Counter)
for w in windows:
    t0 = w[0][0]
    for t, tid, sym in w:
        b = int((t - t0) * 1e6 // bucket_us)
        kind = kinds.get(tid, 'other')
        idle = kind == 'worker' and IDLE.search(sym) is not None
        busy[b]['idle' if idle else kind] += 1
        if not idle:
            work[b][shorten(sym)] += 1
for b in sorted(busy):
    top = ', '.join('%s %.1f' % (s, c / per_cpu) for s, c in work[b].most_common(3))
    print('    %5dus work %5.1f idle %5.1f tokio %4.1f | %s' % (
        b * bucket_us, busy[b]['worker'] / per_cpu, busy[b]['idle'] / per_cpu,
        (busy[b]['tokio'] + busy[b]['other']) / per_cpu, top))
PY
            sudo rm -f /tmp/ab-cpu.data /tmp/ab-cpu.txt
        }
        for setup in "default:" "default:PIVOT_SPIN_LIMIT=2000000" "96:"; do
            local workers="${setup%%:*}" extra_env="${setup#*:}"
            [[ "$workers" == default ]] && workers=""
            start_with_workers "$workers" "$extra_env"
            echo ">>> stages: workers=${workers:-default} ${extra_env}, median of 10 hot runs (ms)"
            for n in 0 1 6 19 2 7 24 38 40 13 22; do
                local sql
                sql="$(sed -n "$((n + 1))p" queries.sql)"
                echo "    Q$n: $(stage_medians "${sql%;}")"
            done
            if [[ -z "$extra_env" ]]; then
                for n in 1 38 40 22; do
                    cpu_timeline "$n" "workers=${workers:-default}"
                done
            fi
            if [[ -z "$workers" && -z "$extra_env" ]]; then
                # Who takes the contended 32-bit atomics (a std Mutex lock /
                # unlock pair) that show up at the start of a small query.
                local sql
                sql="$(sed -n 41p queries.sql)"
                sql="${sql%;}"
                sudo perf record -a -g -F 10000 -o /tmp/ab-cg.data -- bash -c "for _ in 1 2 3 4 5; do sleep 0.1; psql -h 127.0.0.1 -p $port -U postgres -d postgres -q -o /dev/null -c \"$sql\"; done" >/dev/null 2>&1 || true
                echo "--- callers of contended 32-bit atomics in Q40 (workers=default)"
                sudo perf report -i /tmp/ab-cg.data --no-children --stdio --symbols=__aarch64_cas4_acq,__aarch64_swp4_rel -g caller,0.5,callee,function,percent 2>/dev/null \
                    | c++filt | grep -v '^#' | grep -v '^$' | sed -E 's/\[[0-9a-f]{16}\]//g' | cut -c1-200 | head -160 || true
                sudo rm -f /tmp/ab-cg.data
                for n in 0 40; do
                    local sql qlen
                    sql="$(sed -n "$((n + 1))p" queries.sql)"
                    sql="${sql%;}"
                    qlen=$(( $(printf '%s' "$sql" | wc -c) + 6 ))
                    for _ in 1 2 3; do psql -h 127.0.0.1 -p "$port" -U postgres -d postgres -q -o /dev/null -c "$sql" >/dev/null 2>&1; done
                    sudo perf record -a -e sched:sched_waking,sched:sched_switch,syscalls:sys_exit_recvfrom,syscalls:sys_enter_sendto \
                        -o /tmp/ab-sched.data -- bash -c "sleep 0.3; psql -h 127.0.0.1 -p $port -U postgres -d postgres -q -o /dev/null -c \"$sql\"; sleep 0.05" >/dev/null 2>&1 || true
                    sudo perf script -i /tmp/ab-sched.data -F comm,tid,cpu,time,event,trace 2>/dev/null >/tmp/ab-sched.txt || true
                    echo "--- sched Q$n (workers=default ${extra_env})"
                    sudo python3 - "$pid" "$qlen" /tmp/ab-sched.txt <<'PY' || true
import re, sys, os, collections
pid, qlen, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
names = {t: open(f'/proc/{pid}/task/{t}/comm').read().strip() for t in os.listdir(f'/proc/{pid}/task')}
rows = []
for l in open(path):
    m = re.match(r'\s*(.+?)\s+(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\S+):\s*(.*)', l)
    if m:
        rows.append((float(m.group(4)), m.group(2), m.group(5), m.group(6)))
recv = [t for t, tid, e, r in rows if 'exit_recvfrom' in e and tid in names and r.split()[-1].startswith('0x') and int(r.split()[-1], 16) == qlen]
if not recv:
    print("    no query receive found"); sys.exit()
t0 = recv[-1]
sends = [t for t, tid, e, r in rows if 'sendto' in e and tid in names and t > t0]
tend = max(sends) if sends else t0
print("    server receive -> last send: %.0f us" % ((tend - t0) * 1e6))
kinds = {t: ('tokio' if n.startswith('tokio') else ('worker' if t != pid and n == names[pid] else 'other')) for t, n in names.items()}
buckets = collections.defaultdict(lambda: collections.Counter())
first_on = {}
for t, tid, e, r in rows:
    if not (t0 <= t <= tend):
        continue
    d = (t - t0) * 1e6
    if e == 'sched:sched_waking':
        tgt = re.search(r'pid=(\d+)', r).group(1)
        if tgt in names:
            waker = kinds.get(tid, 'ext')
            buckets[int(d // 100)][waker + '>' + kinds[tgt]] += 1
    elif e == 'sched:sched_switch':
        mm = re.search(r'==> .*:(\d+) \[', r)
        if mm and mm.group(1) in names and mm.group(1) not in first_on:
            first_on[mm.group(1)] = d
for b in sorted(buckets):
    print("    %5d-%5dus wakes: %s" % (b * 100, (b + 1) * 100, ', '.join('%s x%d' % kv for kv in buckets[b].most_common(4))))
w = sorted(v for k, v in first_on.items() if kinds.get(k) == 'worker')
if w:
    print("    workers first on-cpu: n=%d first %.0fus median %.0fus last %.0fus" % (len(w), w[0], w[len(w) // 2], w[-1]))
PY
                done
            fi
            stop_server
            rm -rf "$catalog"
        done
        sudo rm -f /tmp/ab-sched.data /tmp/ab-sched.txt
    ) || true
}

# Turn a harness log into "idx cold hot" rows: idx is the 0-based position of the
# query, cold is the first try, hot is the min of the remaining tries ("null" if
# any needed value is missing). The harness prints one "[t1,t2,t3]," line per
# query, in order.
parse_timings() {
    awk '
    /^\[/ {
        line = $0
        gsub(/[][,]+$/, "", line)      # strip trailing "],"
        gsub(/^\[/, "", line)          # strip leading "["
        n = split(line, t, ",")
        cold = t[1]
        hot = ""
        for (i = 2; i <= n; i++) {
            if (t[i] == "null") continue
            if (hot == "" || t[i] + 0 < hot + 0) hot = t[i]
        }
        if (hot == "") hot = "null"
        printf "%d %s %s\n", idx++, cold, hot
    }' "$1"
}

before_out="/tmp/ab-before.out"; after_out="/tmp/ab-after.out"
before_tsv="/tmp/ab-before.tsv"; after_tsv="/tmp/ab-after.tsv"
duck_out="/tmp/ab-duck.out";     duck_tsv="/tmp/ab-duck.tsv"

echo "=== A/B via ClickBench pivot-parquet: '$before_label' (before) vs '$after_label' (after) ===" >"$report"
echo "source=$source_path tries=$iterations sleep_between_queries=${sleep_between_queries}s regression_pct=$regression_pct queries=${queries:-all}" >>"$report"
[[ -n "$server_env" ]] && echo "server_env=$server_env" >>"$report"
echo >>"$report"

echo ">>> timing BEFORE through ClickBench harness"
run_harness "$before_bin" "$before_out" 7799 /tmp/ab-cat-before
parse_timings "$before_out" >"$before_tsv"

echo ">>> timing AFTER through ClickBench harness"
run_harness "$after_bin" "$after_out" 7798 /tmp/ab-cat-after
parse_timings "$after_out" >"$after_tsv"
stages_after "$after_bin"

# Join by query index and render the cold/hot diff. Gate on hot: a query whose
# hot time is >= regression_pct slower after than before fails the run. Labels
# come from query_labels[idx].
labels_csv="$(IFS=,; echo "${query_labels[*]}")"
failures="$(awk \
    -v bf="$before_tsv" -v af="$after_tsv" -v t="$regression_pct" -v labels="$labels_csv" \
    -v blabel="$before_label" -v alabel="$after_label" '
# Timings are in seconds; floor the divisor at 1ms so sub-millisecond
# baselines do not blow the percentage up (they are not meaningful anyway).
function pct(b, a) { b = (b < 0.001 ? 0.001 : b); return (a - b) / b * 100 }
function valid(x) { return (x != "" && x != "null") }
BEGIN {
    nl = split(labels, lab, ",")
    printf "%-6s %10s %10s %8s   %10s %10s %8s  %s\n", \
        "query", "cold_b", "cold_a", "cold_Δ%", "hot_b", "hot_a", "hot_Δ%", "status" > "/dev/stderr"
}
FILENAME == bf { bc[$1] = $2; bh[$1] = $3; seen[$1] = 1; next }
FILENAME == af { ac[$1] = $2; ah[$1] = $3; seen[$1] = 1; next }
END {
    for (i = 0; i in seen; i++) {
        q = "Q" (i < nl ? lab[i + 1] : i)
        cb = bc[i]; ca = ac[i]; hb = bh[i]; ha = ah[i]
        # Cold diff (informational).
        if (valid(cb) && valid(ca)) cd = sprintf("%+.1f%%", pct(cb, ca)); else cd = "-"
        # Hot diff (the gate). A missing timing on either side (query errored
        # or produced no result) is a failure too, not a silent pass.
        status = "ok"; hd = "-"
        if (!valid(hb) || !valid(ha)) {
            status = "ERR"; print q " ERR (missing timing)"
        } else {
            hp = pct(hb, ha)
            hd = sprintf("%+.1f%%", hp)
            if (hp >= t) { status = "REGRESSION"; print q " " hd }
            else if (hp <= -t) status = "improved"
        }
        printf "%-6s %10s %10s %8s   %10s %10s %8s  %s\n", \
            q, cb, ca, cd, hb, ha, hd, status > "/dev/stderr"

        # ClickBench score: each build relative to the per-query best (min),
        # regularised with +10ms (10ms = 0.01s since timings are seconds), then
        # geomean over queries. Fastest build on every query = 1.00.
        if (valid(cb) && valid(ca)) {
            b = (cb < ca ? cb : ca)
            bcl += log((cb + 0.01) / (b + 0.01))
            acl += log((ca + 0.01) / (b + 0.01)); cn++
        }
        if (valid(hb) && valid(ha)) {
            b = (hb < ha ? hb : ha)
            bhl += log((hb + 0.01) / (b + 0.01))
            ahl += log((ha + 0.01) / (b + 0.01)); hn++
        }
    }
    printf "\nClickBench score - geomean of (t+10ms)/(best+10ms) per query (1.00 = fastest on every query):\n" > "/dev/stderr"
    printf "  %-14s %8s %8s\n", "", "cold", "hot" > "/dev/stderr"
    printf "  %-14s %7.2fx %7.2fx\n", blabel, (cn ? exp(bcl / cn) : 0), (hn ? exp(bhl / hn) : 0) > "/dev/stderr"
    printf "  %-14s %7.2fx %7.2fx\n", alabel, (cn ? exp(acl / cn) : 0), (hn ? exp(ahl / hn) : 0) > "/dev/stderr"
    printf "  scored %d/%d queries (cold/hot)\n", cn, hn > "/dev/stderr"
}' "$before_tsv" "$after_tsv" 2>>"$report")"

{
    echo
    echo "=== failures: hot regressions (>= ${regression_pct}%) or missing timings ==="
    if [[ -n "$failures" ]]; then echo "$failures"; else echo "none"; fi
} >>"$report"

# Optional DuckDB reference: time DuckDB (parquet, partitioned) once and compare
# the after build against it. DuckDB is not part of the pass/fail gate — it is a
# baseline, reported with a ClickBench-style score (geomean of pivot/duckdb).
if [[ -n "$duckdb_data" ]]; then
    echo ">>> timing DuckDB (parquet, partitioned) through ClickBench harness"
    run_duckdb_harness "$duck_out"
    parse_timings "$duck_out" >"$duck_tsv"

    { echo; echo "=== pivot ($after_label) vs DuckDB (parquet, partitioned) ==="; } >>"$report"
    awk -v af="$after_tsv" -v dk="$duck_tsv" -v labels="$labels_csv" '
    function valid(x) { return (x != "" && x != "null") }
    function ratio(p, d) { return (p + 0.01) / (d + 0.01) }   # <1 => pivot faster
    BEGIN {
        nl = split(labels, lab, ",")
        printf "%-6s %10s %10s %9s   %10s %10s %9s\n", \
            "query", "piv_cold", "duck_cold", "c p/d", "piv_hot", "duck_hot", "h p/d" > "/dev/stderr"
    }
    FILENAME == af { pc[$1] = $2; ph[$1] = $3; seen[$1] = 1; next }
    FILENAME == dk { dc[$1] = $2; dh[$1] = $3; seen[$1] = 1; next }
    END {
        for (i = 0; i in seen; i++) {
            q = "Q" (i < nl ? lab[i + 1] : i)
            cr = (valid(pc[i]) && valid(dc[i])) ? sprintf("%.2fx", ratio(pc[i], dc[i])) : "-"
            hr = (valid(ph[i]) && valid(dh[i])) ? sprintf("%.2fx", ratio(ph[i], dh[i])) : "-"
            printf "%-6s %10s %10s %9s   %10s %10s %9s\n", \
                q, pc[i], dc[i], cr, ph[i], dh[i], hr > "/dev/stderr"
            # ClickBench score: each system relative to the per-query best
            # (min), regularised with +10ms, then geomean over queries.
            if (valid(pc[i]) && valid(dc[i])) {
                b = (pc[i] < dc[i] ? pc[i] : dc[i])
                pcl += log((pc[i] + 0.01) / (b + 0.01))
                dcl += log((dc[i] + 0.01) / (b + 0.01)); cn++
            }
            if (valid(ph[i]) && valid(dh[i])) {
                b = (ph[i] < dh[i] ? ph[i] : dh[i])
                phl += log((ph[i] + 0.01) / (b + 0.01))
                dhl += log((dh[i] + 0.01) / (b + 0.01)); hn++
            }
        }
        printf "\nClickBench score - geomean of (t+10ms)/(best+10ms) per query (1.00 = fastest on every query):\n" > "/dev/stderr"
        printf "  %-8s %8s %8s\n", "", "cold", "hot" > "/dev/stderr"
        printf "  %-8s %7.2fx %7.2fx\n", "pivot",  (cn ? exp(pcl / cn) : 0), (hn ? exp(phl / hn) : 0) > "/dev/stderr"
        printf "  %-8s %7.2fx %7.2fx\n", "duckdb", (cn ? exp(dcl / cn) : 0), (hn ? exp(dhl / hn) : 0) > "/dev/stderr"
        printf "  scored %d/%d queries (cold/hot)\n", cn, hn > "/dev/stderr"
    }' "$after_tsv" "$duck_tsv" 2>>"$report"
fi

cat "$report"

if [[ -n "$failures" ]]; then
    exit 3
fi
