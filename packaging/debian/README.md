# Debian package

The `pivot` package installs one public executable and a systemd service:

```text
/usr/bin/pivot
/etc/pivot/config.yaml
/lib/systemd/system/pivot.service
/usr/share/pivot/config.yaml
/var/lib/pivot/metastore.yaml
/var/lib/pivot/datastores/default/
```

Installation creates the dedicated `pivot` system user, enables the service at
boot, and starts it immediately. The service runs this foreground command:

```sh
pivot server \
  --config /etc/pivot/config.yaml \
  --metastore-file /var/lib/pivot/metastore.yaml
```

The package stores its default configuration template at
`/usr/share/pivot/config.yaml`. On the first installation, it copies that
template to `/etc/pivot/config.yaml` only if no file already exists there. The
live configuration belongs to the administrator and upgrades never replace or
modify it, so there is no conffile prompt. Removing or purging the package also
leaves the live configuration in place.

The metastore file is mutable server state: the package creates it only when
absent, and the server owns subsequent rewrites. Removing or purging the
package stops the service but deliberately preserves `/var/lib/pivot` and the
service account. Database data and metastore entries are never deleted as a
package-script side effect.

Build a native package into `dist/`:

```sh
packaging/debian/build.sh
packaging/debian/test-package.sh dist/pivot_*.deb
packaging/debian/test-install.sh dist/pivot_*.deb
```

The CI build uses `packaging/debian/Dockerfile` on native amd64 and arm64
runners. Its Debian Bullseye build environment targets glibc 2.31 and produces
the `.deb` as a workflow artifact; it does not publish an apt repository.

## Why CI builds the package in Docker

Docker is only used to provide a pinned build environment. It is not included
in the `.deb`, and users do not need Docker to install or run Pivot.

Rust binaries can depend on the glibc version provided by the machine that
compiled them. Building directly on GitHub's Ubuntu runner could therefore
produce a binary that does not start on an older supported distribution. The
Dockerfile compiles Pivot on Debian Bullseye with glibc 2.31, making the package
compatible with Debian 11+ and Ubuntu 20.04+.

The final `scratch` stage contains only the generated `.deb`. GitHub Actions
exports that file into `dist/`, tests it, and uploads it as a workflow artifact.
