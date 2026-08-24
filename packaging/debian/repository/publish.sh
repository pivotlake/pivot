#!/bin/sh

# Upload a complete two-architecture release to Artifact Registry.

set -eu

fail() {
    printf 'apt repository: %s\n' "$*" >&2
    exit 1
}

[ "$#" -ge 5 ] || fail 'usage: publish.sh PROJECT LOCATION stable|testing PACKAGE-amd64.deb PACKAGE-arm64.deb'
project=$1
location=$2
distribution=$3
shift 3

command -v dpkg-deb >/dev/null 2>&1 || fail "required command 'dpkg-deb' was not found"
command -v gcloud >/dev/null 2>&1 || fail "required command 'gcloud' was not found"
command -v sha256sum >/dev/null 2>&1 || fail "required command 'sha256sum' was not found"
[ -n "$project" ] || fail 'the Artifact Registry project is required'
[ -n "$location" ] || fail 'the Artifact Registry location is required'

case "$distribution" in
    stable|testing) ;;
    *) fail "unsupported distribution '$distribution'" ;;
esac

version=
amd64_package=
arm64_package=
for package in "$@"; do
    [ -f "$package" ] || fail "package '$package' was not found"
    [ "$(dpkg-deb --field "$package" Package)" = pivot ] ||
        fail "'$package' is not the pivot package"

    package_version=$(dpkg-deb --field "$package" Version)
    if [ -z "$version" ]; then
        version=$package_version
    elif [ "$package_version" != "$version" ]; then
        fail "package versions differ: '$version' and '$package_version'"
    fi

    case "$(dpkg-deb --field "$package" Architecture)" in
        amd64)
            [ -z "$amd64_package" ] || fail 'more than one amd64 package was supplied'
            amd64_package=$package
            ;;
        arm64)
            [ -z "$arm64_package" ] || fail 'more than one arm64 package was supplied'
            arm64_package=$package
            ;;
        *) fail "'$package' has an unsupported architecture" ;;
    esac
done
[ -n "$amd64_package" ] || fail 'an amd64 package is required'
[ -n "$arm64_package" ] || fail 'an arm64 package is required'

case "$distribution:$version" in
    testing:*~rc.*+git.*) ;;
    testing:*) fail "testing version '$version' must look like BASE~rc.TIMESTAMP+git.SHA" ;;
    stable:*~*) fail "stable version '$version' must not be a Debian prerelease" ;;
    stable:*) ;;
esac

# People who opt into testing receive final releases too. Publish finals to
# testing first so a failure can't advance stable while leaving testing on its
# previous candidate. Artifact Registry retains older package versions and
# generates and signs the APT indexes after each immutable upload.
repositories=testing
if [ "$distribution" = stable ]; then
    repositories='testing stable'
fi

upload_package() {
    repository=$1
    package=$2
    architecture=$(dpkg-deb --field "$package" Architecture)
    package_hash=$(sha256sum "$package" | cut -d' ' -f1)
    existing_hashes=$(gcloud artifacts files list \
        --package=pivot \
        --version="$version" \
        --repository="$repository" \
        --project="$project" \
        --location="$location" \
        --flatten='hashes[]' \
        --filter="name~'_${architecture}_' AND hashes.type=SHA256" \
        --format='value(hashes.value)')

    if [ -n "$existing_hashes" ]; then
        [ "$existing_hashes" = "$package_hash" ] ||
            fail "pivot $version/$architecture already exists in $repository with different content"
        printf 'Pivot %s/%s already exists unchanged in %s\n' \
            "$version" "$architecture" "$repository"
        return
    fi

    gcloud artifacts apt upload "$repository" \
        --project="$project" \
        --location="$location" \
        --source="$package" \
        --quiet
}

for repository in $repositories; do
    format=$(gcloud artifacts repositories describe "$repository" \
        --project="$project" --location="$location" --format='value(format)')
    [ "$format" = APT ] || fail "'$repository' is not an APT repository in $project/$location"

    # Upload the foreign architecture first. A brief partially updated window
    # can then affect only that architecture; the hosted amd64 publisher and
    # most clients see the release only after the second upload succeeds.
    for package in "$arm64_package" "$amd64_package"; do
        upload_package "$repository" "$package"
    done
    printf 'Published Pivot %s to %s\n' "$version" "$repository"
done
