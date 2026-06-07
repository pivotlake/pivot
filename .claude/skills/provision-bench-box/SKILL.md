---
name: provision-bench-box
description: Launch an AWS EC2 machine from a pre-baked pivotdb ClickBench AMI (x86 c6a or arm64 c8g, on-demand or spot), sync the latest source, and run benchmark.sh on it (PGO or non-PGO). Use when asked to create/launch a benchmark box, run ClickBench on a new/bigger/different instance type, or clone the bench environment to another machine. For an ALREADY-running box, use perf-bench-remote instead.
---

# Provision a bench box and run ClickBench (always from an AMI)

End-to-end: launch an EC2 instance **from a ready AMI** → sync current source + rebuild → run `benchmark.sh` (pivot vs DuckDB).
Always launch from one of the ready AMIs below — they carry the slow parts (toolchain, DuckDB, the 14 GB dataset) so setup is seconds, not ~30 min. For an existing box, use **perf-bench-remote**.

## AWS context (Epsio account)
- Region: `eu-central-1`. Account `411636662497` (`maor@epsio.io`). `aws` CLI is configured locally.
- Key pair: `maorgkeypair`; **private key local at `/Users/maorkern/Documents/dev/maorgkeypair.pem`**.
- Reusable network: SG `sg-04d837ce979c15d02` (allows SSH), subnet `subnet-0ab5f218e2c889ee6` (AZ `eu-central-1b`). Reuse unless told otherwise.
- SSH user is `ubuntu`. Passwordless `sudo` works.

## Ready AMIs — pick by target arch
**c6a/c7a = x86_64. c8g/c7g/x8g = arm64 (Graviton).** An AMI only boots on its own arch — match it.

| Target instances | Arch | AMI | Name | Notes |
|---|---|---|---|---|
| c8g.*, c7g.*, x8g.* | arm64 | `ami-06c63fa89d232c7bd` | `clickbench-c8g-ready-20260607` | Fully provisioned (rust, duckdb, `~/pivotdb`, `~/hits`). 500 GB gp2 root. **Known-good.** |
| c6a.*, c7a.* | x86_64 | `ami-0f45bd48e0a5c58cf` | `pivotdb-clickbench-base-20260607` | Snapshot of the original clickbench box. 150 GB gp3 root → **override to 500 GB gp2 at launch**; verify `~/hits` + toolchain on first use. |

If a ready AMI is missing or stale for an arch, build one (see **Refresh the ready AMI**).

## Disk — match ClickBench
Official ClickBench hardware is **c6a.4xlarge + 500 GB gp2**; gp2's IOPS/throughput scale with size and use burst credits, which is what ClickBench measures — **use 500 GB gp2, not gp3.** Always pass the block-device override at launch (required for the x86 AMI whose root is 150 GB gp3; harmless for the arm AMI which is already 500 GB gp2 — the FS auto-grows on boot):
`--block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp2","DeleteOnTermination":true}}]'`

## Launch
On-demand:
```bash
AMI=ami-06c63fa89d232c7bd          # arm64; use ami-0f45bd48e0a5c58cf for x86
TYPE=c8g.4xlarge
aws ec2 run-instances --region eu-central-1 --image-id $AMI --instance-type $TYPE \
  --key-name maorgkeypair --subnet-id subnet-0ab5f218e2c889ee6 \
  --security-group-ids sg-04d837ce979c15d02 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp2","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=clickbench-$TYPE}]"
```
**Spot** (much cheaper for big boxes — e.g. c8g.metal-48xl ~$1.2/hr spot vs ~$6.9/hr on-demand): add
`--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'`.
Check price/capacity first: `aws ec2 describe-spot-price-history --instance-types $TYPE --product-descriptions Linux/UNIX --query 'SpotPriceHistory[0:3]'`. Spot can be reclaimed with a 2-min warning.

Then `aws ec2 wait instance-running --instance-ids <ID>`, read `PublicIpAddress` from `describe-instances`, and poll SSH (metal boots take several min):
`ssh -i /Users/maorkern/Documents/dev/maorgkeypair.pem -o StrictHostKeyChecking=accept-new ubuntu@<IP>`

## After launch — two things the AMI does NOT give you fresh
1. **The baked source is point-in-time → sync the current repo, or you benchmark stale code.** From the Mac:
   `rsync -az --exclude '.git' --exclude '**/target/' -e "ssh -i /Users/maorkern/Documents/dev/maorgkeypair.pem" /Users/maorkern/Documents/dev/pivotdb/ ubuntu@<IP>:~/pivotdb/`
   Then rebuild on the box: `source ~/.cargo/env && cd ~/pivotdb/benchmarks && cargo build --release --bin pivot-bench` (~4 min; arrow-rs + DuckDB C++ bridge).
2. **`/tmp` is wiped on first boot → the PGO profile is gone.** Regenerate it (see PGO section). `~/hits` and `~/pivotdb` survive (they're under `$HOME`).

On the x86 AMI, also confirm `ls ~/hits/*.parquet`, `duckdb --version`, `which just`, `source ~/.cargo/env && rustc --version` — provision anything missing (apt: `build-essential cmake clang libclang-dev pkg-config git`; `rustup` + `rustup component add llvm-tools-preview`; `just`; `curl -sSf https://install.duckdb.org | sh`; dataset: `wget https://datasets.clickhouse.com/hits_compatible/hits.parquet` → `~/hits/`).

## Long commands → background + poll
Builds, PGO gen, and full runs exceed SSH idle limits. Launch detached and poll:
```bash
nohup bash ~/job.sh > ~/job.log 2>&1 &        # job.sh ends with: echo JOB_DONE_OK
# poll from a fresh ssh until JOB_DONE_OK / an error appears in job.log
```

## The query set
Build the list of real queries (exclude `q23` and the `-duckdb.sql` overrides):
```bash
ls ~/pivotdb/benchmarks/clickbench/q*.sql | grep -v -- '-duckdb.sql' \
  | xargs -n1 basename | sed 's/\.sql$//' | grep -v '^q23$' | paste -sd, > ~/query_ids.txt
```
- **Exclude `q23`** (`SELECT *`) — it panics `"Evicting"`.
- **`pivot-bench` / `just pgo-gen` with no `--query` enumerate ALL `q*.sql`, including `q42-duckdb.sql`** (a DuckDB-only override using the ClickHouse `toDateTime` macro pivot can't run → `Scalar Function todatetime does not exist`). Always pass `--query "$(cat ~/query_ids.txt)"`. `benchmark.sh` itself skips `-duckdb.sql`, so only the bare profile-gen run needs the explicit list. In `benchmark.sh`, DuckDB runs `q42-duckdb.sql` while pivot runs `q42.sql` — intended, fair.

## Running the benchmark

### With PGO (default for headline numbers — `benchmark.sh` requires it)
`benchmark.sh` drives pivot via `just pgo-use`, which needs a merged profile. Generate once, then run:
```bash
cd ~/pivotdb/benchmarks
IDS=$(cat ~/query_ids.txt)
just pgo-clean
just pgo-gen run --release -- --source ~/hits --query "$IDS" --iterations 2 --skip-check   # instrumented build+run, merges to /tmp/benchmarks-pgo/merged.profdata
./benchmark.sh --source ~/hits --query "$IDS" --iterations 3 --skip-check
```
- Regenerate the profile after a code/source change, and on every freshly-launched box (`/tmp` was wiped).
- `--skip-check` avoids aborting on q24's tie-ambiguous LIMIT. Drop it only when verifying correctness.
- PGO sometimes *hurts* memory-bound queries — sanity-check against non-PGO if a number looks off.

### Without PGO (quick, or to isolate a PGO effect)
Run `pivot-bench` directly on plain `--release`, no profile needed:
```bash
cd ~/pivotdb/benchmarks
cargo build --release --bin pivot-bench
./target/release/pivot-bench --source ~/hits --query "$(cat ~/query_ids.txt)" --iterations 3 --skip-check
```
For a non-PGO pivot-vs-DuckDB table, run DuckDB separately: `./run-duckdb.sh --source ~/hits --query <ids> --iterations 3`. (`benchmark.sh` has no non-PGO mode — it always calls `just pgo-use`.)

## Warm the box before measuring (critical for cold numbers)
A box launched from an AMI has a snapshot-restored root volume whose blocks **hydrate lazily from S3 on first access** (EBS lazy-init). The **first** full benchmark pass therefore inflates cold times — especially small queries (observed on c8g.metal-48xl: q42 cold 808 ms on the first pass, ~42–59 ms once warm; big I/O-bound queries look normal either way). **Always run the suite once and discard it**, then measure. Never trust the first snapshot-restored pass — it reads as a huge cold "regression" that isn't real. (To force hydration up front: `sudo fio --name=warm --filename=/dev/nvme0n1 --rw=read --bs=1M --iodepth=32 --direct=1 --runtime=60 --time_based` or `dd if=~/hits/hits.parquet of=/dev/null bs=8M`.)

## Reading the numbers
- **Cold** = iteration 1 (page cache dropped). **Hot** = mean of the rest (steady state, the headline). `benchmark.sh` prints a per-query speedup + a ClickBench geomean score (lower = better, 1.00 = fastest on every query).
- pivot defaults to **workers = core count**. On very large boxes (e.g. 192-vCPU metal) the cold path scales poorly (192-worker spin-up + huge-ring prefault on query 1 dominates small queries) — pivot's cold numbers there are much worse than on a 16-core box; hot stays competitive. Try `--workers <N>` to probe.
- `prefault_buffers` is one-time warmup — ignore for steady-state.

## Refresh the ready AMI (when the env drifts)
The AMIs go stale as the repo / dataset / toolchain change. To rebake: launch from the current ready AMI, `rsync` the latest source, rebuild, (optionally re-download the dataset / bump duckdb), then snapshot:
```bash
aws ec2 create-image --region eu-central-1 --instance-id <BOX> --no-reboot \
  --name "clickbench-<arch>-ready-<YYYYMMDD>" --description "provisioned pivotdb clickbench env"
# wait for State=available, then update the AMI table at the top of this skill with the new id.
```
`--no-reboot` does not disturb a running box (background EBS snapshot). To build the FIRST ready AMI for a new arch, launch a base Canonical Ubuntu AMI (`--owners 099720109477`, filter `ubuntu/images/hvm-ssd-gp3/ubuntu-*-{amd64,arm64}-server-*`), do the full provisioning (apt deps, rust+llvm-tools, just, duckdb, rsync source, dataset, build), then snapshot.

## Cleanup (always offer when done)
- Stop (keep disk): `aws ec2 stop-instances --instance-ids <ID>`. Terminate (gone): `aws ec2 terminate-instances --instance-ids <ID>`. Spot one-time instances should be terminated, not stopped.
- AMIs + their snapshots cost EBS-snapshot $: `aws ec2 deregister-image --image-id <AMI>` then `aws ec2 delete-snapshot --snapshot-id <SNAP>`. Don't delete the two ready AMIs above unless replacing them.
- List boxes: `aws ec2 describe-instances --filters Name=instance-state-name,Values=running,stopped --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,Name:Tags[?Key==\`Name\`]|[0].Value}' --output table`.

See `perf-bench-remote` (running on an existing box, profiling, plan inspection) and the `clickbench-bench-box-provisioning` memory.
