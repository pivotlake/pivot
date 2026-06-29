---
name: deploy-parquetsink
description: Build and deploy the pivotdb-server OTLP→parquet sink on the parquetsink box (ssh ubuntu@parquetsink). Use when redeploying a new binary, swapping versions, or rolling back.
---

# Deploy pivotdb-server to parquetsink

Box: `ssh ubuntu@parquetsink` (x86_64). Live dir: `~/pivotdb-deploy/` — `run.sh` (restart-loop supervisor), `pivotdb-server` (active binary), `pivot.log`, `DEPLOYED_SHA`, `pivotdb-server.prev` (rollback).

`run.sh` holds all config, so the swap is zero-config:
export GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/parquet_sink/gcs-key.json   # SA parquet-sink-writer@epsio-io (has delete)
export RUST_LOG=info PANIC_ON_EVICT=false                                       # PANIC_ON_EVICT=false is required
./pivotdb-server --path gs://epsio-io-otel-parquet/pivotdb-otel --bind 0.0.0.0:5432 \
--otel-config /home/ubuntu/pivotdb-deploy/otel.toml --compact --compact-bytes 52428800
NOTE: the compacter is opt-in. `--compact` must be present or the server refuses `--compact-bytes`/`--compact-min-files` and exits. Older run.sh lacking `--compact` will fail to start after the swap; add it.
otel.toml: OTLP/gRPC on 0.0.0.0:4317, logs→table otel_logs, DeploymentId column from resource:deployment_id, partition_by=DeploymentId sort_by=ObservedTimestamp. The otel_logs table must already exist in the catalog (ingest refuses otherwise).

Get the binary

- Preferred: download CI-built gs://pivot-private/server/pivotdb-server-x86_64-<full-sha> (gsutil cp).
- Fallback — build on box: needs rust (rustup), clang+libclang-dev (bindgen), cmake, rsync (none preinstalled). Sync the pivot crates + submodules to ~/pivotdb-build/, then cd server && cargo build --release. First build ~12min (duckdb C++); incremental ~1.5min.

Swap

cp NEWBIN ~/pivotdb-deploy/pivotdb-server.new && chmod +x ~/pivotdb-deploy/pivotdb-server.new
~/pivotdb-deploy/pivotdb-server.new --help            # smoke test
kill <supervisor_pid>                                  # STOP SUPERVISOR FIRST
kill -INT <server_pid>; # wait ~15s; kill -9 if still alive
cd ~/pivotdb-deploy
cp -f pivotdb-server pivotdb-server.prev && mv -f pivotdb-server.new pivotdb-server
echo <sha> > DEPLOYED_SHA
setsid nohup bash run.sh >>pivot.log 2>&1 </dev/null & disown
Verify: ss -ltn | grep -E ':4317|:5432' listening (~35-55s startup — catalog footer load over gs://), pivot.log shows appended parquet, run a query. Rollback: mv pivotdb-server.prev pivotdb-server + restart.

Pitfalls (all hit for real)

- pgrep self-matches your own ssh command (its cmdline contains the pattern). Kill by explicit PID, or match with a regex char class: pgrep -f "[p]ivotdb-server --path", bash[ ]run.
- Stop the supervisor BEFORE the server — else run.sh instantly relaunches the old binary. Once it's dead, the server PID is stable.
- Inline nohup … & inside a multi-line ssh '…' often doesn't persist. Use setsid nohup … </dev/null & disown, or a launch script.
- Catalog load is slow (~35-55s, thousands of tiny gs:// footers) before ports bind — don't conclude "crashed" early.
- Building competes with the live server (~15 GB prefaulted ring on a 29 GB box). Stop the server during a build or risk OOM.
- Changing cargo profile (e.g. CARGO_PROFILE_RELEASE_DEBUG=…) busts the cache → full Rust recompile. Keep the profile constant.
- Deployed (CI) binaries have no debug info → perf/addr2line mislead and stacks don't unwind. For real debugging build with CARGO_PROFILE_RELEASE_DEBUG=line-tables-only.
- Don't run unbounded full-table queries against the live sink (memory). Use DuckDB on the parquet for ad-hoc checks.
- Compaction is opt-in via `--compact` (the sink owns the data, so it should compact). A newer binary started WITHOUT `--compact` but WITH `--compact-bytes` exits immediately with "the following required arguments were not provided: --compact". When deploying a build at/after the opt-in change, make sure run.sh passes `--compact`. A read-only/external reader must omit it (no `--compact` = never merges or deletes catalog files).