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

## Release packages

The Deploy Binaries workflow builds the released server with this same
Dockerfile, so the `.deb` it publishes contains exactly the binary the release
uploads. The `package` stage exports both: `dist/pivot` is the binary, and
`dist/pivot_<version>_<arch>.deb` wraps it. Each release leg uploads

```text
pivot-<arch>-<sha>          pivot-<arch>-latest
pivot-<arch>-<sha>.deb      pivot-<arch>-latest.deb
```

to the destination bucket, and attaches the `.deb` as a workflow artifact.
Building the release inside Bullseye is what keeps the published binary and
package on the glibc 2.31 floor described below.

## Why the package is built in Docker

Docker is only used to provide a pinned build environment. It is not included
in the `.deb`, and users do not need Docker to install or run Pivot.

Rust binaries can depend on the glibc version provided by the machine that
compiled them. Building directly on GitHub's Ubuntu runner could therefore
produce a binary that does not start on an older supported distribution. The
Dockerfile compiles Pivot on Debian Bullseye with glibc 2.31, making the package
compatible with Debian 11+ and Ubuntu 20.04+.

The final `scratch` stage contains only the generated `.deb` and the binary it
was built from. GitHub Actions exports both into `dist/`, tests the package, and
uploads it as a workflow artifact.
