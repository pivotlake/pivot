#!/bin/sh

# Verify the published repository exactly as a fresh Debian user reaches it.

set -eu

fail() {
    printf 'apt repository test: %s\n' "$*" >&2
    exit 1
}

[ "$#" -eq 3 ] || fail 'usage: test-repository.sh REPOSITORY_URL stable|testing VERSION'
repository_url=${1%/}
download_url=${repository_url%/*}
distribution=$2
version=$3
case "$distribution" in stable|testing) ;; *) fail "invalid distribution '$distribution'" ;; esac

docker run --rm \
    --env "PIVOT_APT_URL=$repository_url" \
    --env "PIVOT_DOWNLOAD_URL=$download_url" \
    --env "PIVOT_APT_DISTRIBUTION=$distribution" \
    --env "PIVOT_APT_VERSION=$version" \
    debian:bookworm-slim \
    sh -euxc '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates curl gnupg
        test ! -e /usr/lib/apt/methods/ar+https
        install -d -m 0755 /usr/share/keyrings
        curl --fail --silent --show-error --location \
            "$PIVOT_DOWNLOAD_URL/keys/pivotlake-archive-key.asc" |
            gpg --dearmor --yes --output /usr/share/keyrings/pivotlake-archive-keyring.gpg
        printf "deb [signed-by=/usr/share/keyrings/pivotlake-archive-keyring.gpg] %s %s main\n" \
            "$PIVOT_APT_URL" "$PIVOT_APT_DISTRIBUTION" > /etc/apt/sources.list.d/pivotlake.list
        native_architecture=$(dpkg --print-architecture)
        case "$native_architecture" in
            amd64) foreign_architecture=arm64 ;;
            arm64) foreign_architecture=amd64 ;;
            *) echo "unsupported test architecture: $native_architecture" >&2; exit 1 ;;
        esac
        dpkg --add-architecture "$foreign_architecture"

        found=
        for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
            apt-get update
            candidate=$(apt-cache policy pivot | sed -n "s/^  Candidate: //p")
            if [ "$candidate" = "$PIVOT_APT_VERSION" ]; then
                found=yes
                break
            fi
            echo "Waiting for Artifact Registry to index $PIVOT_APT_VERSION (candidate: $candidate, attempt: $attempt/12)"
            sleep 10
        done
        test "$found" = yes

        download_directory=$(mktemp -d)
        cd "$download_directory"
        apt-get download "pivot:$foreign_architecture=$PIVOT_APT_VERSION"
        set -- pivot_*.deb
        test "$#" -eq 1
        test "$(dpkg-deb --field "$1" Architecture)" = "$foreign_architecture"
        cd /

        apt-get install -y "pivot=$PIVOT_APT_VERSION"
        test "$(dpkg-query --show --showformat="\${Version}" pivot)" = "$PIVOT_APT_VERSION"
        pivot --version
    '

printf 'Published APT repository test passed: %s %s\n' "$distribution" "$version"
