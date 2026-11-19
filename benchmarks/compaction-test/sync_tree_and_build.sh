#!/bin/bash
# usage: sync_tree_and_build.sh <directory holding a source tree>
# Makes the test machine's copy match that tree's crates exactly (not the build
# output, submodules or the loader example that exists only there), re-adds the
# machine-only session-token change and builds the server, waiting for it.
set -euo pipefail
TREE=$1
KEY=${KEY:?path of the ssh key of the test machine}
BOX=${BOX:?user@address of the test machine}
cd "$TREE"
for dir in dispatch parquet-engine datastore-pivotlake object-storage metastore-disk catalog planner datastore-iceberg iceberg-tables table-file-index; do
  rsync -az --delete -e "ssh -i $KEY" $dir/ $BOX:pivot-merge/$dir/
done
rsync -az --delete -e "ssh -i $KEY" bin/src/ $BOX:pivot-merge/bin/src/
rsync -az -e "ssh -i $KEY" bin/Cargo.toml $BOX:pivot-merge/bin/Cargo.toml
rsync -az --delete -e "ssh -i $KEY" --exclude duckdb-sources duckdb-planner/ $BOX:pivot-merge/duckdb-planner/
scp -q -i $KEY Cargo.toml Cargo.lock $BOX:pivot-merge/
ssh -i $KEY $BOX "sed -i 's|                session_token: None,|                session_token: std::env::var(\"AWS_SESSION_TOKEN\").ok(),|' ~/pivot-merge/metastore-disk/src/secrets.rs && cd ~/pivot-merge && grep -c AWS_SESSION_TOKEN metastore-disk/src/secrets.rs && RUSTC_WRAPPER= ~/.cargo/bin/cargo build --release --bin pivot > ~/merge-test/build.log 2>&1; tail -1 ~/merge-test/build.log; grep -E '^error' -A8 ~/merge-test/build.log | head -20"
