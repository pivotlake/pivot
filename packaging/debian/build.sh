#!/bin/sh

# Build an installable Pivot Debian package for the current architecture.

set -eu
umask 022

fail() {
    printf 'debian package: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found"
}

require_command cargo
require_command dpkg
require_command dpkg-deb
require_command install
require_command mktemp
require_command sed
require_command uname

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
invocation_directory=$(pwd)
output_directory=${1:-$repository_root/dist}
case "$output_directory" in
    /*) ;;
    *) output_directory=$invocation_directory/$output_directory ;;
esac

package_id=$(cd "$repository_root" && cargo pkgid -p bin)
case "$package_id" in
    *@*) cargo_version=${package_id##*@} ;;
    *#*) cargo_version=${package_id##*#} ;;
    *) fail "could not read the workspace version from '$package_id'" ;;
esac
if [ -n "${PIVOT_DEB_VERSION:-}" ]; then
    version=$PIVOT_DEB_VERSION
else
    case "$cargo_version" in
        *-*) version=${cargo_version%%-*}~${cargo_version#*-} ;;
        *) version=$cargo_version ;;
    esac
fi
case "$version" in
    ''|*[!A-Za-z0-9.+:~-]*) fail "invalid Debian version '$version'" ;;
esac
dpkg --validate-version "$version" >/dev/null 2>&1 ||
    fail "invalid Debian version '$version'"

if [ -n "${PIVOT_DEB_ARCH:-}" ]; then
    architecture=$PIVOT_DEB_ARCH
else
    case $(uname -m) in
        x86_64|amd64) architecture=amd64 ;;
        aarch64|arm64) architecture=arm64 ;;
        *) fail "unsupported architecture $(uname -m)" ;;
    esac
fi
case "$architecture" in
    amd64|arm64) ;;
    *) fail "unsupported Debian architecture '$architecture'" ;;
esac

binary=${PIVOT_BINARY:-$repository_root/target/release/pivot}
if [ -z "${PIVOT_BINARY:-}" ]; then
    printf 'Building Pivot %s for Debian %s...\n' "$cargo_version" "$architecture"
    (cd "$repository_root" && cargo build --release -p bin --bin pivot)
fi
[ -f "$binary" ] || fail "Pivot binary '$binary' was not found"
[ -x "$binary" ] || fail "Pivot binary '$binary' is not executable"
version_output=$($binary --version 2>&1) || fail "Pivot binary did not run: $version_output"
[ "$version_output" = "pivot $cargo_version" ] ||
    fail "Pivot binary reported '$version_output', expected 'pivot $cargo_version'"

staging_directory=$(mktemp -d)
cleanup() {
    rm -rf "$staging_directory"
}
trap cleanup EXIT HUP INT TERM

package_root=$staging_directory/pivot
install -d -m 0755 \
    "$package_root/DEBIAN" \
    "$package_root/lib/systemd/system" \
    "$package_root/usr/bin" \
    "$package_root/usr/share/doc/pivot" \
    "$package_root/usr/share/pivot"
install -m 0755 "$binary" "$package_root/usr/bin/pivot"
if command -v strip >/dev/null 2>&1; then
    strip "$package_root/usr/bin/pivot"
fi
install -m 0644 "$repository_root/packaging/debian/config.yaml" \
    "$package_root/usr/share/pivot/config.yaml"
install -m 0644 "$repository_root/packaging/debian/pivot.service" \
    "$package_root/lib/systemd/system/pivot.service"
install -m 0644 "$repository_root/packaging/debian/copyright" \
    "$package_root/usr/share/doc/pivot/copyright"
# The binary links third-party code statically, so the attribution travels
# with the package rather than only with the repository.
install -m 0644 "$repository_root/NOTICE" \
    "$package_root/usr/share/doc/pivot/NOTICE"
install -m 0755 "$repository_root/packaging/debian/postinst" "$package_root/DEBIAN/postinst"
install -m 0755 "$repository_root/packaging/debian/prerm" "$package_root/DEBIAN/prerm"
install -m 0755 "$repository_root/packaging/debian/postrm" "$package_root/DEBIAN/postrm"

installed_size=$(du -sk "$package_root" | awk '{print $1}')
sed \
    -e "s/@VERSION@/$version/g" \
    -e "s/@ARCHITECTURE@/$architecture/g" \
    -e "s/@INSTALLED_SIZE@/$installed_size/g" \
    "$repository_root/packaging/debian/control.in" > "$package_root/DEBIAN/control"

mkdir -p "$output_directory"
package=$output_directory/pivot_${version}_${architecture}.deb
dpkg-deb --root-owner-group --build -Zgzip -z9 "$package_root" "$package"
printf 'Built %s\n' "$package"
