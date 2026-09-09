#!/bin/sh

# Install and remove a package in a clean Debian container. The service-manager
# calls are recorded because the container does not boot systemd; the foreground
# server path has its own integration test in the cli crate.

set -eu

fail() {
    printf 'debian install test: %s\n' "$*" >&2
    exit 1
}

[ "$#" -eq 1 ] || fail 'usage: test-install.sh PACKAGE.deb'
package=$(realpath "$1")
[ -f "$package" ] || fail "package '$package' was not found"
package_directory=$(dirname "$package")
package_name=$(basename "$package")

docker run --rm \
    --mount "type=bind,source=$package_directory,target=/packages,readonly" \
    --env "PIVOT_TEST_PACKAGE=$package_name" \
    debian:bookworm-slim \
    sh -euxc '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y adduser init-system-helpers libc6 libgcc-s1 libstdc++6 systemd-sysv

        systemctl_path=$(command -v systemctl)
        mv "$systemctl_path" "$systemctl_path.real"
        printf "%s\n" \
            "#!/bin/sh" \
            "printf '\''%s\\n'\'' \"\$*\" >> /tmp/systemctl.log" \
            "case \"\$*\" in" \
            "    \"--system daemon-reload\") exit 0 ;;" \
            "    *\"preset pivot.service\"*) exec \"$systemctl_path.real\" --root=/ preset pivot.service ;;" \
            "esac" \
            "exec \"$systemctl_path.real\" \"\$@\"" > "$systemctl_path"
        chmod 755 "$systemctl_path"

        deb_systemd_invoke_path=$(command -v deb-systemd-invoke)
        mv "$deb_systemd_invoke_path" "$deb_systemd_invoke_path.real"
        printf "%s\n" \
            "#!/bin/sh" \
            "printf '\''%s\\n'\'' \"\$*\" >> /tmp/deb-systemd-invoke.log" \
            "exit 0" > "$deb_systemd_invoke_path"
        chmod 755 "$deb_systemd_invoke_path"
        mkdir -p /run/systemd/system

        dpkg --install "/packages/$PIVOT_TEST_PACKAGE"

        pivot --version
        pivot server --help
        test "$(getent passwd pivot | cut -d: -f6)" = /var/lib/pivot
        case "$(getent passwd pivot | cut -d: -f7)" in
            /usr/sbin/nologin|/sbin/nologin|/bin/false) ;;
            *) exit 1 ;;
        esac
        test "$(stat -c %U:%G /var/lib/pivot/datastores/default)" = pivot:pivot
        test "$(stat -c %a /var/lib/pivot/datastores/default)" = 700
        test "$(stat -c %U:%G /var/lib/pivot/metastore.yaml)" = pivot:pivot
        test "$(stat -c %a /var/lib/pivot/metastore.yaml)" = 600
        grep -Fx "datastores: {}" /var/lib/pivot/metastore.yaml
        grep -Fx "users: {}" /var/lib/pivot/metastore.yaml
        grep -Fx "secrets: {}" /var/lib/pivot/metastore.yaml
        test "$(stat -c %U:%G /etc/pivot/config.yaml)" = root:pivot
        test "$(stat -c %a /etc/pivot/config.yaml)" = 640
        test -L /etc/systemd/system/multi-user.target.wants/pivot.service
        grep -Fx -- "--system daemon-reload" /tmp/systemctl.log
        grep -Fx "start pivot.service" /tmp/deb-systemd-invoke.log
        if grep -F "is-active" /tmp/systemctl.log; then
            exit 1
        fi

        printf "\n# administrator-owned setting\n" >> /etc/pivot/config.yaml
        dpkg --install "/packages/$PIVOT_TEST_PACKAGE"
        grep -Fx "# administrator-owned setting" /etc/pivot/config.yaml
        grep -Fx "try-restart pivot.service" /tmp/deb-systemd-invoke.log

        apt-get remove -y pivot
        test ! -e /usr/bin/pivot
        test -d /var/lib/pivot/datastores/default
        test -f /var/lib/pivot/metastore.yaml
        grep -Fx "# administrator-owned setting" /etc/pivot/config.yaml
        getent passwd pivot
        grep -Fx "stop pivot.service" /tmp/deb-systemd-invoke.log
        apt-get purge -y pivot
        test -d /var/lib/pivot/datastores/default
        test -f /var/lib/pivot/metastore.yaml
        grep -Fx "# administrator-owned setting" /etc/pivot/config.yaml
        getent passwd pivot
        test ! -e /etc/systemd/system/pivot.service
        test ! -e /etc/systemd/system/multi-user.target.wants/pivot.service
    '

printf 'Debian install tests passed: %s\n' "$package"
