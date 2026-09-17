#!/bin/sh

set -eu

fail() {
    printf 'debian package test: %s\n' "$*" >&2
    exit 1
}

[ "$#" -eq 1 ] || fail 'usage: test-package.sh PACKAGE.deb'
package=$1
[ -f "$package" ] || fail "package '$package' was not found"

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
temporary_directory=$(mktemp -d)
cleanup() {
    rm -rf "$temporary_directory"
}
trap cleanup EXIT HUP INT TERM

for script in build.sh postinst prerm postrm test-package.sh test-install.sh; do
    sh -n "$repository_root/packaging/debian/$script"
done

[ "$(dpkg-deb --field "$package" Package)" = pivot ] || fail 'unexpected package name'
case "$(dpkg-deb --field "$package" Architecture)" in
    amd64|arm64) ;;
    *) fail 'unexpected package architecture' ;;
esac
dpkg-deb --field "$package" Depends | grep -F 'init-system-helpers' >/dev/null ||
    fail 'init-system-helpers dependency is missing'
if dpkg-deb --field "$package" Depends | grep -F 'systemd-sysv' >/dev/null; then
    fail 'package must not force systemd to be the active init system'
fi

dpkg-deb --extract "$package" "$temporary_directory/root"
dpkg-deb --control "$package" "$temporary_directory/control"

binary=$temporary_directory/root/usr/bin/pivot
[ -x "$binary" ] || fail 'package does not contain an executable /usr/bin/pivot'
case "$($binary --version)" in
    'pivot '*) ;;
    *) fail 'packaged binary did not report a Pivot version' ;;
esac
[ -f "$temporary_directory/root/usr/share/pivot/config.yaml" ] ||
    fail 'default config template is missing'
[ ! -e "$temporary_directory/root/etc/pivot/config.yaml" ] ||
    fail 'live config must be created by postinst, not owned by the package'
[ -f "$temporary_directory/root/lib/systemd/system/pivot.service" ] || fail 'unit is missing'
[ -f "$temporary_directory/root/usr/share/doc/pivot/copyright" ] || fail 'copyright is missing'
grep -Fx 'License: MIT or Apache-2.0' \
    "$temporary_directory/root/usr/share/doc/pivot/copyright" >/dev/null ||
    fail 'copyright does not declare the dual license'
grep -Fx 'ExecStart=/usr/bin/pivot server --config /etc/pivot/config.yaml' \
    "$temporary_directory/root/lib/systemd/system/pivot.service" >/dev/null ||
    fail 'systemd does not invoke pivot server'
[ ! -e "$temporary_directory/control/conffiles" ] ||
    fail 'administrator-owned config must not be registered as a conffile'
grep -F '/var/lib/pivot/datastores/default' \
    "$temporary_directory/root/usr/share/pivot/config.yaml" >/dev/null ||
    fail 'default datastore path is incorrect'
grep -Fx '  path: /var/lib/pivot/metastore.yaml' \
    "$temporary_directory/root/usr/share/pivot/config.yaml" >/dev/null ||
    fail 'config does not name the metastore file'

printf 'Debian package tests passed: %s\n' "$package"
