#!/bin/bash
# Start the pivot server over the test bucket with the instance role's keys.
# The keys are temporary: the config's S3 secret carries the key pair, and the
# session token goes through AWS_SESSION_TOKEN (a change made on this machine's
# copy of the code only).
set -euo pipefail
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')
ROLE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/)
CREDS=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE)
field() { echo "$CREDS" | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }
export AWS_SESSION_TOKEN=$(field Token)
echo "keys expire at $(field Expiration)"
cd ~/merge-test
umask 077
cat > config.yaml <<EOF
log: info
server:
  bind: 127.0.0.1:5432
datastores:
  lake:
    kind: pivotlake
    location: s3://pivot-compaction-test-20261116/lake
    default: true
    compact: true
    compact_bytes: ${COMPACT_BYTES:-1g}
    vacuum: false
secrets:
  bucket:
    type: s3
    access_key_id: $(field AccessKeyId)
    secret_access_key: $(field SecretAccessKey)
    region: eu-central-1
EOF
exec ~/pivot-merge/target/release/pivot server --config ~/merge-test/config.yaml
