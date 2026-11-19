#!/bin/bash
# usage: full_sync_and_build.sh [machine-only patch script in this directory]...
# Syncs the whole worktree (not the build output, git data or submodules) to
# the test machine, re-adds the machine-only session-token change, the timing
# probe and any other patch scripts named, and starts a release build in the
# background. The loader example that exists only on the machine is kept.
set -euo pipefail
HERE=$(dirname "$0")
cd /home/ubuntu/pivotdb/.claude/worktrees/compact
KEY=${KEY:?path of the ssh key of the test machine}
BOX=${BOX:?user@address of the test machine}
rsync -az -e "ssh -i $KEY" \
  --exclude target --exclude .git --exclude .claude --exclude arrow-rs \
  --exclude duckdb-planner/duckdb-sources \
  ./ $BOX:pivot-merge/
PATCHES=""
for patch in "$@"; do
  scp -q -i $KEY "$HERE/$patch" $BOX:merge-test/
  PATCHES="$PATCHES python3 ~/merge-test/$patch ~/pivot-merge &&"
done
ssh -i $KEY $BOX "sed -i 's|                session_token: None,|                session_token: std::env::var(\"AWS_SESSION_TOKEN\").ok(),|' ~/pivot-merge/metastore-disk/src/secrets.rs && cd ~/merge-test && python3 time_operators.py && $PATCHES cd ~/pivot-merge && grep -c AWS_SESSION_TOKEN metastore-disk/src/secrets.rs && (RUSTC_WRAPPER= nohup ~/.cargo/bin/cargo build --release --bin pivot > ~/merge-test/build.log 2>&1 &) ; sleep 2; echo build started"
