#!/bin/bash
# usage: sync_and_build.sh [machine-only patch script in this directory]...
# Syncs the four crates the branch changes to the test machine, re-adds the
# machine-only timing probe and any other patch scripts named, and starts a
# release build in the background.
set -euo pipefail
HERE=$(dirname "$0")
cd /home/ubuntu/pivotdb/.claude/worktrees/compact
KEY=${KEY:?path of the ssh key of the test machine}
BOX=${BOX:?user@address of the test machine}
for crate in parquet-engine datastore-pivotlake dispatch object-storage; do
  rsync -az --delete -e "ssh -i $KEY" $crate/src/ $BOX:pivot-merge/$crate/src/
  scp -q -i $KEY $crate/Cargo.toml $BOX:pivot-merge/$crate/Cargo.toml
done
scp -q -i $KEY Cargo.toml Cargo.lock $BOX:pivot-merge/
PATCHES=""
for patch in "$@"; do
  scp -q -i $KEY "$HERE/$patch" $BOX:merge-test/
  PATCHES="$PATCHES python3 ~/merge-test/$patch ~/pivot-merge &&"
done
ssh -i $KEY $BOX "cd ~/merge-test && python3 time_operators.py && $PATCHES cd ~/pivot-merge && grep -c AWS_SESSION_TOKEN metastore-disk/src/secrets.rs && (RUSTC_WRAPPER= nohup ~/.cargo/bin/cargo build --release --bin pivot > ~/merge-test/build.log 2>&1 &) ; sleep 2; echo build started"
