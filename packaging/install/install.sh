#!/bin/sh
#
# Pivot installer for Linux.
#
#     curl https://pivotlake.io | sh
#
# Downloads the `pivot` command-line binary built for this machine and installs
# it under the home directory of the user who ran it. Nothing here needs root,
# nothing is written outside $HOME, and no shell profile is edited: when the
# install directory is not already on PATH, the line to add is printed.
#
# Environment:
#   PIVOT_VERSION   install this version instead of the latest release
#   PIVOT_HOME      install root (default: ~/.pivot)
#   PIVOT_BASE_URL  where the downloads live (default: the release bucket)
#
# It expects <base>/latest_version.txt naming the current version, and
# <base>/v<version>/pivot-<platform>.gz beside its .sha256.
#
# Everything below lives in functions that only run once the last line of the
# file has been read. Piped into a shell, this file is executed as it arrives,
# and a connection dropped halfway would otherwise run half an installer.

set -eu

fail() {
    printf 'pivot install: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found"
}

# `--fail` keeps an HTML error page from landing on disk as a binary, and
# `--location` follows the redirect a download host may serve.
download() {
    curl --fail --location --show-error "$@"
}

compute_sha256() {
    checksum_line=$(sha256sum "$1")
    printf '%s\n' "${checksum_line%% *}"
}

# A version names both a directory and a URL path, so accept only characters
# that are safe in both. Rejecting a slash here is what keeps a mangled
# latest_version.txt from writing outside the install root.
validate_version() {
    case "$1" in
        '' | *[!A-Za-z0-9.+_-]*) fail "'$1' is not a valid Pivot version" ;;
    esac
}

# A path under the home directory reads better shortened, with `~` for a line a
# person reads and `$HOME` for one they paste into a shell profile, where it
# keeps working for whoever ends up copying it.
abbreviate_home() {
    case "$1" in
        "$HOME"/*) printf '%s%s\n' "$2" "${1#"$HOME"}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

runs_correctly() {
    version_line=$("$1" --version 2>/dev/null) || return 1
    case "$version_line" in
        'pivot '*) return 0 ;;
        *) return 1 ;;
    esac
}

detect_platform() {
    operating_system=$(uname -s)
    machine=$(uname -m)
    platform=
    case "$operating_system" in
        Linux)
            case "$machine" in
                x86_64 | amd64) platform=linux-amd64 ;;
                aarch64 | arm64) platform=linux-arm64 ;;
            esac
            ;;
        Darwin)
            fail "Pivot is not built for macOS yet. Build it from source, or run it under Linux."
            ;;
    esac
    [ -n "$platform" ] || fail "there is no Pivot build for $operating_system on $machine"
}

install_binary() {
    download --progress-bar --output "$staging/pivot.gz" "$archive_url" ||
        fail "could not download $archive_url"
    download --silent --output "$staging/pivot.gz.sha256" "$archive_url.sha256" ||
        fail "could not download $archive_url.sha256"

    published_checksum=$(cut -d ' ' -f 1 < "$staging/pivot.gz.sha256")
    downloaded_checksum=$(compute_sha256 "$staging/pivot.gz")
    [ -n "$published_checksum" ] && [ "$published_checksum" = "$downloaded_checksum" ] ||
        fail "checksum mismatch for pivot-$platform.gz: published $published_checksum, downloaded $downloaded_checksum"

    gzip -dc "$staging/pivot.gz" > "$staging/pivot" || fail "could not unpack pivot-$platform.gz"
    chmod 0755 "$staging/pivot"
    runs_correctly "$staging/pivot" || fail "the downloaded binary did not run on this machine"

    mkdir -p "$install_directory"
    mv "$staging/pivot" "$binary"
    printf 'Installed Pivot %s to %s\n' "$version" "$(abbreviate_home "$binary" '~')"
}

# Put a `pivot` shortcut in ~/.local/bin, which is already on PATH on most
# systems, so the command works without editing a shell profile. It points at
# `latest`, so the next upgrade moves that one symlink and this follows.
link_into_local_bin() {
    linked_into_local_bin=0

    # A pinned install does not get to own the shared name.
    [ "$version" = "$latest_version" ] || return 0

    # Use the directory, never create it: one we invent probably is not on PATH.
    [ -d "$local_bin" ] && [ -w "$local_bin" ] || return 0

    if [ -L "$local_bin/pivot" ]; then
        # Ours: re-point it. `-L` rather than `-e` so a link left dangling by a
        # deleted install root is repaired instead of colliding. A link pointing
        # anywhere else is someone else's and is left alone.
        case "$(readlink "$local_bin/pivot")" in
            "$pivot_home"/*)
                ln -sfn "$latest_link/pivot" "$local_bin/pivot"
                linked_into_local_bin=1
                ;;
        esac
    elif [ ! -e "$local_bin/pivot" ]; then
        # Nothing there, so nothing to overwrite.
        ln -s "$latest_link/pivot" "$local_bin/pivot"
        linked_into_local_bin=1
    fi

    # Anything else already holds the name `pivot`: leave it untouched.

    if [ "$linked_into_local_bin" = 1 ]; then
        printf 'Linked %s -> %s\n' \
            "$(abbreviate_home "$local_bin/pivot" '~')" \
            "$(abbreviate_home "$latest_link/pivot" '~')"
    fi
}

report_how_to_run() {
    on_path=0
    case ":${PATH:-}:" in
        *":$binary_directory:"*) on_path=1 ;;
    esac
    if [ "$linked_into_local_bin" = 1 ]; then
        case ":${PATH:-}:" in
            *":$local_bin:"*) on_path=1 ;;
        esac
    fi

    if [ "$on_path" = 1 ]; then
        pivot_command=pivot
    else
        pivot_command=$(abbreviate_home "$binary_directory/pivot" '~')
    fi

    printf '\nGet started:\n'
    printf '    %s open ~/pivot-data           open a local datastore in the SQL shell\n' "$pivot_command"
    printf '    %s open s3://bucket/prefix     or one that lives in object storage\n' "$pivot_command"
    printf '\nDocumentation: https://pivotlake.io/docs\n\n'

    if [ "$on_path" = 0 ]; then
        bold=
        reset=
        if [ -t 1 ] && [ "${TERM:-dumb}" != dumb ]; then
            bold=$(printf '\033[1m')
            reset=$(printf '\033[0m')
        fi
        case "${SHELL:-}" in
            */zsh) shell_profile='${ZDOTDIR:-$HOME}/.zshrc' ;;
            */bash) shell_profile='$HOME/.bashrc' ;;
            *) shell_profile='$HOME/.profile' ;;
        esac
        printf '%s%s is not on your PATH. To add it permanently and activate it now, run:\n' \
            "$bold" "$(abbreviate_home "$binary_directory" '~')"
        printf '    echo '\''export PATH="%s:$PATH"'\'' >> "%s" && . "%s"%s\n' \
            "$(abbreviate_home "$binary_directory" '$HOME')" "$shell_profile" "$shell_profile" "$reset"
    fi
}

main() {
    [ -n "${HOME:-}" ] || fail "HOME is not set, so there is nowhere to install to"

    require_command curl
    require_command gzip
    require_command mktemp
    require_command sha256sum
    require_command uname

    # This script is served from pivotlake.io, but what it downloads lives in a
    # public bucket, so the two hostnames differ.
    base_url=${PIVOT_BASE_URL:-https://storage.googleapis.com/pivot-releases}
    base_url=${base_url%/}
    pivot_home=${PIVOT_HOME:-$HOME/.pivot}
    local_bin=$HOME/.local/bin
    latest_link=$pivot_home/cli/latest

    detect_platform

    mkdir -p "$pivot_home/cli" || fail "could not create $pivot_home/cli"

    # Staging sits inside the install root so the finished binary moves into
    # place with a rename on the same filesystem: an interrupted download can
    # never leave a half-written `pivot` behind.
    staging=$(mktemp -d "$pivot_home/cli/.install.XXXXXX") ||
        fail "could not create a staging directory under $pivot_home/cli"
    trap 'rm -rf "$staging"' EXIT HUP INT TERM

    # A pinned install names its own version, and only consults the latest
    # pointer to decide whether it owns the `latest` and ~/.local/bin links. A
    # release candidate can be pinned before any release exists, so a missing
    # pointer leaves those links alone rather than failing the install.
    latest_version=
    if download --silent --output "$staging/latest_version.txt" "$base_url/latest_version.txt"; then
        latest_version=$(tr -d '[:space:]' < "$staging/latest_version.txt")
        validate_version "$latest_version"
    elif [ -z "${PIVOT_VERSION:-}" ]; then
        fail "could not read $base_url/latest_version.txt"
    fi

    if [ -n "${PIVOT_VERSION:-}" ]; then
        version=$PIVOT_VERSION
        validate_version "$version"
    else
        version=$latest_version
    fi

    install_directory=$pivot_home/cli/$version
    binary=$install_directory/pivot
    archive_url=$base_url/v$version/pivot-$platform.gz

    printf '\n  pivot _   installing %s (%s)\n\n' "$version" "$platform"

    if [ -x "$binary" ] && runs_correctly "$binary"; then
        printf 'Pivot %s is already installed at %s\n' "$version" "$binary"
    else
        install_binary
    fi

    # `latest` follows the newest release only. An install pinned with
    # PIVOT_VERSION is reachable by its own path and leaves everyone else's
    # `pivot` alone.
    if [ "$version" = "$latest_version" ]; then
        ln -sfn "$install_directory" "$latest_link"
        binary_directory=$latest_link
    else
        binary_directory=$install_directory
    fi

    link_into_local_bin
    report_how_to_run
}

main
