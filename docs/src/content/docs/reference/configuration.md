---
title: "Server configuration"
description: Configure server resources, TLS, and startup settings.
---

Pivot reads one YAML configuration file, the one `--config` names. The
`server` section controls the instance. The top-level `datastores`, `secrets`,
and `users` maps declare what it serves; they belong to the operator and the
server never rewrites them. The optional `metastore` section names the file
the server writes to, where `CREATE USER` lands.

You can start a Pivot server with the configuration below. Save it as
`pivot.yaml` and pass its path with `--config`:

```sh
pivot server --config pivot.yaml
```

The apt service and Docker image use these configuration files by default:

| Installation | Configuration file |
| --- | --- |
| Debian / Ubuntu (`apt install pivot`) | `/etc/pivot/config.yaml` |
| Docker (`pivotlake/pivot`) | `/etc/pivot/pivot.yaml` inside the container |

## Example configuration

This configuration serves a local datastore:

```yaml
server:
  bind: 127.0.0.1:5432
  memory: 4g
  workers: 4
  refresh_interval: 30s

datastores:
  local:
    kind: pivot
    location: ./pivot-data
    default: true
```

## Server settings

Every `server` setting is optional. Unknown fields are rejected, including
unknown fields nested inside a section.

| Key | Default | Description |
| --- | --- | --- |
| `bind` | `127.0.0.1:5432` | Address for the Postgres wire endpoint. |
| `memory` | 80% of total memory minus 4 GiB | Buffer-pool budget as a size, such as `4g` or `512m`. Overrides `PIVOT_MEMORY_PCT`. The 4 GiB is held back for allocations outside the pool. |
| `workers` | Machine core count | Dispatch worker threads. |
| `refresh_interval` | `30s` | How often catalogs check for commits made by other Pivot instances sharing the same datastore. |

### TLS

Configure both files under `server.tls` to offer TLS to clients:

```yaml
server:
  tls:
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key
```

| Key | Default | Description |
| --- | --- | --- |
| `cert` | Required when enabled | PEM certificate followed by any intermediate certificates. |
| `key` | Required when enabled | PEM private key in PKCS#8, PKCS#1, or SEC1 form. |

Enabling TLS makes it available but does not require clients to use it.

## Datastores, secrets, and users

Configure data, storage credentials, and users in three top-level maps:

| Map | Purpose | Documentation |
| --- | --- | --- |
| `datastores` | Local or object-storage datastores and their maintenance settings. | [Datastores](/docs/reference/server/datastores/#datastores) |
| `secrets` | Scoped S3 or GCS credentials. | [Storage credentials](/docs/reference/server/datastores/#storage-credentials) |
| `users` | Trust or SCRAM authentication. | [Users and authentication](/docs/reference/server/authentication/) |

## Metastore

The `metastore` section names the file the server writes to. It is optional:
without it the server serves only the configuration's own entries and refuses
`CREATE USER`.

```yaml
metastore:
  kind: file
  path: /var/lib/pivot/metastore.yaml
```

| Key | Default | Description |
| --- | --- | --- |
| `kind` | Required | `file` is the only supported value. |
| `path` | Required | A YAML file holding the same `datastores`, `secrets`, and `users` maps as the configuration, without a `server` or `metastore` section. It must exist, and only the server should be able to read it. |

The configuration's entries and the file's are served together. A name defined
in both is a startup error, and `CREATE USER` refuses a name the configuration
defines, because the server never rewrites the configuration.

## Docker configuration

On first startup, the `pivotlake/pivot` image creates
`/var/lib/pivot/metastore.yaml` if it does not exist, using these environment
variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `PIVOT_DATASTORE` | `/var/lib/pivot/datastores/default` | Location of the generated default datastore. Accepts a local path, `s3://` URI, or `gs://` URI. |
| `AWS_ACCESS_KEY_ID` | Required for S3 | Access key stored in the generated S3 secret. |
| `AWS_SECRET_ACCESS_KEY` | Required for S3 | Secret key stored in the generated S3 secret. |
| `AWS_REGION` | `AWS_DEFAULT_REGION`, then `us-east-1` | Region stored in the generated S3 secret. |
| `AWS_ENDPOINT_URL` | Unset | S3-compatible endpoint, such as MinIO. |

With a persistent `/var/lib/pivot` volume, subsequent starts use the existing
`metastore.yaml` and ignore these variables.
To change an initialized volume, edit its metastore file or start with a fresh
volume.

## Related

- [Annotated configuration example](https://github.com/Epsio-Labs/pivotdb/blob/main/bin/config.example.yaml)
- [Quickstart](/docs/quickstart/)
- [Session settings](/docs/reference/statements/set-reset/)
