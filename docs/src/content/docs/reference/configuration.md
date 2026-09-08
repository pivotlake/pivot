---
title: "Server configuration"
description: Configure server resources, caching, TLS, and startup settings.
---

Pivot reads a YAML configuration file. The `server` section controls the
instance; `metastore` declares the data and users it serves.

## Example

Save this as `pivot.yaml` to serve a local datastore:

```yaml
server:
  bind: 127.0.0.1:5432
  memory: 4g
  workers: 4
  refresh_interval: 30s

metastore:
  datastores:
    local:
      kind: pivot
      location: ./pivot-data
      default: true
```

Start the server with:

```sh
pivot server --config pivot.yaml
```

## Server

Every `server` setting is optional. Unknown fields are rejected, including
unknown fields nested inside a section.

| Key | Default | Description |
| --- | --- | --- |
| `server.bind` | `127.0.0.1:5432` | Address for the Postgres wire endpoint. |
| `server.memory` | 80% of total memory | Buffer-pool budget. Overrides `PIVOT_MEMORY_PCT`. |
| `server.workers` | Machine core count | Dispatch worker threads. |
| `server.refresh_interval` | `30s` | How often in-memory catalogs refresh commits made by other processes. |

## Size and duration values

Sizes accept whole bytes or base-1024 `k`, `m`, `g`, and `t` suffixes, such as
`512m` or `32g`. Durations require `ms`, `s`, `m`, or `h`, such as `500ms` or
`30s`.

## Disk cache

The optional disk cache stores remote reads on local disk. Add it under
`server`:

```yaml
server:
  disk_cache:
    dir: /var/cache/pivot
    size: 64g
    max_objects: 65536
```

| Key | Default | Description |
| --- | --- | --- |
| `server.disk_cache.dir` | Required when enabled | Persistent local directory for cached remote reads. |
| `server.disk_cache.size` | `64g` | Cached-byte budget. |
| `server.disk_cache.max_objects` | `65536` | Maximum cached objects and open cache file descriptors. |

## TLS

Configure both files to offer TLS to clients:

```yaml
server:
  tls:
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key
```

| Key | Default | Description |
| --- | --- | --- |
| `server.tls.cert` | Required when enabled | PEM certificate followed by any intermediate certificates. |
| `server.tls.key` | Required when enabled | PEM private key in PKCS#8, PKCS#1, or SEC1 form. |

Both files are required when `server.tls` is present. Enabling TLS makes it
available but does not require clients to use it.

## Metastore

### `metastore.datastores`

Declare local or object-storage datastores and their maintenance settings in
[Datastores and storage credentials](/docs/reference/server/datastores/#datastores).

### `metastore.secrets`

Configure scoped S3 or GCS credentials in
[Storage credentials](/docs/reference/server/datastores/#storage-credentials).

### `metastore.users`

Configure trust or SCRAM authentication in
[Users and authentication](/docs/reference/server/authentication/).

## Docker image bootstrap

On the first server start, the `pivotlake/pivot` image creates
`/var/lib/pivot/metastore.yaml` from environment variables when that file does
not exist.

| Variable | Default | Effect |
| --- | --- | --- |
| `PIVOT_DATASTORE` | `/var/lib/pivot/datastores/default` | Location of the generated default datastore. Accepts a local path, `s3://` URI, or `gs://` URI. |
| `AWS_ACCESS_KEY_ID` | Required for S3 | Access key stored in the generated S3 secret. |
| `AWS_SECRET_ACCESS_KEY` | Required for S3 | Secret key stored in the generated S3 secret. |
| `AWS_REGION` | `AWS_DEFAULT_REGION`, then `us-east-1` | Region stored in the generated S3 secret. |
| `AWS_ENDPOINT_URL` | Unset | S3-compatible endpoint, such as MinIO. |

These variables are bootstrap settings, not live overrides. If
`/var/lib/pivot` is mounted as a persistent volume and already contains
`metastore.yaml`, the image uses that file and ignores the bootstrap variables.
To change an initialized volume, edit its metastore file or start with a fresh
volume.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `PIVOT_MEMORY_PCT` | `80` | Percentage of total memory used when `server.memory` is omitted. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Unset | GCS credentials file used by the ambient credentials chain when no matching GCS secret exists. |

## Related

- [Annotated configuration example](https://github.com/Epsio-Labs/pivotdb/blob/main/bin/config.example.yaml)
- [Quickstart](/docs/quickstart/)
- [Session settings](/docs/reference/statements/set-reset/)
