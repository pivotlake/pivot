#!/usr/bin/env bash
#
# bake-tpch-ab-ami.sh - build the AMI the TPC-H A/B workflow launches
# (.github/workflows/tpch-ab.yml and tpch-flat-ab.yml, and jsonbench-ab.yml,
# which run benchmarks/{tpch,tpch-flat,jsonbench}/bench-*-ab.sh).
#
# Bakes everything the box needs so a fresh instance is ready to build within
# seconds of ssh coming up: the Rust toolchain (+ llvm-tools for profdata),
# just, aws cli, s5cmd, a pinned DuckDB CLI, and a full clone of this repo
# WITH submodules at /opt/pivotdb (the run fetches only the delta and clones
# locally from it). Run this manually with AWS credentials whenever the
# toolchain or the base image should move, then update the TPCH_AB_AMI_ID
# repository variable with the printed AMI id.
#
# The GitHub token is used only during the bake to clone the private repo; the
# baked image keeps a token-free remote URL and no credential files.
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx ./bake-tpch-ab-ami.sh --key-name ec2-key-pair \
#     [--ssh-key ~/ec2-key-pair.pem] \
#     [--region eu-central-1] [--security-group sg-xxx] [--subnet subnet-xxx] \
#     [--repo Epsio-Labs/pivotdb] [--rust-toolchain stable] [--duckdb 1.5.4]

set -euo pipefail

region="eu-central-1"
instance_type="m8g.large"
key_name=""
ssh_key=""
subnet=""
security_group=""
repo="Epsio-Labs/pivotdb"
rust_toolchain="stable"
duckdb_version="1.5.4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)         region="$2"; shift 2 ;;
        --instance-type)  instance_type="$2"; shift 2 ;;
        --key-name)       key_name="$2"; shift 2 ;;
        --ssh-key)        ssh_key="$2"; shift 2 ;;
        --subnet)         subnet="$2"; shift 2 ;;
        --security-group) security_group="$2"; shift 2 ;;
        --repo)           repo="$2"; shift 2 ;;
        --rust-toolchain) rust_toolchain="$2"; shift 2 ;;
        --duckdb)         duckdb_version="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$key_name" ]] || { echo "error: --key-name is required" >&2; exit 2; }
[[ -n "${GITHUB_TOKEN:-}" ]] || { echo "error: GITHUB_TOKEN must be set (repo read access)" >&2; exit 2; }
[[ "$GITHUB_TOKEN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "error: GITHUB_TOKEN is not a single token" >&2; exit 2; }

export AWS_DEFAULT_REGION="$region"

base_ami=$(aws ssm get-parameter \
    --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)
echo ">>> base Ubuntu 24.04 arm64 AMI: $base_ami"

args=(--image-id "$base_ami" --instance-type "$instance_type" --count 1
      --key-name "$key_name"
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]'
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tpch-ab-ami-builder}]')
[[ -n "$subnet" ]] && args+=(--subnet-id "$subnet")
[[ -n "$security_group" ]] && args+=(--security-group-ids "$security_group")

id=$(aws ec2 run-instances "${args[@]}" --query 'Instances[0].InstanceId' --output text)
echo ">>> builder instance: $id"
trap 'echo ">>> terminating builder $id"; aws ec2 terminate-instances --instance-ids "$id" >/dev/null' EXIT

aws ec2 wait instance-running --instance-ids "$id"
ip=$(aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo ">>> builder IP: $ip"

ssh_cmd=(ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10)
[[ -n "$ssh_key" ]] && ssh_cmd+=(-i "$ssh_key")
ssh_cmd+=("ubuntu@$ip")
for _ in $(seq 1 60); do
    if "${ssh_cmd[@]}" true 2>/dev/null; then break; fi
    sleep 5
done

echo ">>> provisioning"
"${ssh_cmd[@]}" GITHUB_TOKEN="$GITHUB_TOKEN" REPO="$repo" \
    RUST_TOOLCHAIN="$rust_toolchain" DUCKDB_VERSION="$duckdb_version" 'bash -s' <<'PROVISION'
set -euxo pipefail

sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -qy \
    build-essential clang lld cmake git git-restore-mtime zstd curl unzip pkg-config libssl-dev python3

# aws cli v2 (the base image ships none)
curl -sSf https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip -o /tmp/awscli.zip
(cd /tmp && unzip -q awscli.zip && sudo ./aws/install && rm -rf aws awscli.zip)

# rust toolchain + llvm-tools (llvm-profdata for the PGO merge)
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN" --profile minimal
~/.cargo/bin/rustup component add llvm-tools

# just (prebuilt)
curl -sSfL https://just.systems/install.sh | bash -s -- --to /tmp/just-bin
sudo install /tmp/just-bin/just /usr/local/bin/just

# s5cmd (parallel S3 transfers; the dataset sync's speed comes from this)
S5CMD_VERSION=2.3.0
curl -sSfL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-arm64.tar.gz" \
    | tar -xz -C /tmp s5cmd
sudo install /tmp/s5cmd /usr/local/bin/s5cmd

# DuckDB CLI, pinned so reference numbers stay comparable across runs
curl -sSfL "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-arm64.zip" \
    -o /tmp/duckdb.zip
(cd /tmp && unzip -q duckdb.zip && sudo install duckdb /usr/local/bin/duckdb && rm -f duckdb duckdb.zip)

# Repo clone with submodules; the baked remote keeps no token.
sudo mkdir -p /opt/pivotdb
sudo chown ubuntu:ubuntu /opt/pivotdb
git clone --recurse-submodules "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git" /opt/pivotdb
git -C /opt/pivotdb remote set-url origin "https://github.com/${REPO}.git"

# Nothing secret may survive into the image.
rm -f ~/.gitconfig ~/.git-credentials ~/.bash_history
history -c || true
PROVISION

echo ">>> creating image"
ami=$(aws ec2 create-image --instance-id "$id" \
    --name "tpch-ab-$(date +%Y%m%d-%H%M)" \
    --description "pivotdb TPC-H A/B box: rust $rust_toolchain, duckdb $duckdb_version, repo clone" \
    --query ImageId --output text)
echo ">>> waiting for $ami to become available (this reboots the builder)"
aws ec2 wait image-available --image-ids "$ami"

echo
echo "AMI ready: $ami"
echo "Set the TPCH_AB_AMI_ID repository variable to it."
