---
title: "Configuration file"
description: Every key of the YAML file that configures a Pivot server, with its default.
---

A Pivot server is configured by a single YAML file, passed with `--config`:

```sh
pivot server --config pivot.yaml
```

The apt service and Docker image read these files by default:

| Installation | Configuration file |
| --- | --- |
| Debian / Ubuntu (`apt install pivot`) | `/etc/pivot/config.yaml` |
| Docker (`pivotlake/pivot`) | `/etc/pivot/pivot.yaml` inside the container |

The server reads the file once at startup. A misspelled or unknown key
anywhere in it is an error rather than a silently ignored setting.

## Example configuration

The smallest useful file names one datastore. Everything else has a default:

```yaml
datastores:
  local:
    kind: pivotlake
    location: ./pivot-data
    default: true
```

Connect as the built-in `pivot` user, which uses trust authentication unless
it is configured under `users`. See
[Built-in user](/docs/reference/server/authentication/#built-in-user).

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## File layout

Every key is shown below. Commented keys are optional, and the value shown
is their default.

Exactly one datastore must set `default: true`. It is the current database,
which unqualified table names refer to. The server refuses to start when no
datastore, or more than one, is marked as the default.

```yaml
# Instance settings
# memory: 80%
# workers: <core count>
# datastore_refresh_interval: 30s
# log: info

# PostgreSQL endpoint
server:
  # bind: 127.0.0.1:5432
  # tls:
  #   cert: /etc/pivot/server.crt
  #   key: /etc/pivot/server.key

# What the server serves
datastores:
  <name>:
    kind: pivotlake
    location: /var/lib/pivot/datastores/<name>
    default: true               # on exactly one datastore
    # compact: true
    # compact_bytes: 64m
    # compact_merge_bytes: <1.3 x compact_bytes>
    # compact_min_files: 100
    # compact_parallelism: 1
    # vacuum: true
  <name>:
    kind: iceberg
    uri: https://catalog.example.com/api
    # warehouse: s3://example-bucket/warehouse
    # secret: <secret name>
    # properties: {}
    # default: false            # true on exactly one datastore

secrets:
  <name>:
    type: s3
    access_key_id: replace-me
    secret_access_key: replace-me
    # scope: s3://example-bucket/
    # region: <discovered from the bucket>
    # endpoint: <AWS S3>
  <name>:
    type: gcs
    credentials_file: /etc/pivot/gcs-key.json
    # scope: gs://example-bucket/
  <name>:
    type: iceberg
    token: replace-me           # or credential: client_id:client_secret
    # oauth2_server_uri: <catalog token endpoint>
    # oauth2_scope: <none>

users:
  <name>:
    auth:
      method: trust             # or password, or scram-sha-256
      # password: replace-me    # with method: password
      # verifier: pivot-scram-sha-256$...   # with method: scram-sha-256

# metastore:
#   kind: file
#   path: /var/lib/pivot/metastore.yaml
```

The sections below document the instance settings, `server`, and
`metastore`. The three maps have their own pages:

| Map | Purpose | Reference |
| --- | --- | --- |
| `datastores` | Local, object-storage, or Iceberg datastores and their maintenance settings. | [Datastores](/docs/reference/server/datastores/#datastores) |
| `secrets` | Scoped S3 or GCS credentials, and Iceberg catalog credentials. | [Storage credentials](/docs/reference/server/datastores/#storage-credentials) |
| `users` | Users that can connect, trusted or with a password. | [Users and authentication](/docs/reference/server/authentication/) |

## Instance settings

The resources the instance runs on. Each has a default, so a file can leave
all of them out.

| Key | Default | Description |
| --- | --- | --- |
| `memory` | `80%` | Buffer-pool budget: a [size](#sizes-and-intervals) such as `32g`, or a percentage of the machine's physical memory. A percentage holds back a 4 GiB reserve for allocations outside the pool, so `80%` on a 64 GiB machine is a 47 GiB pool. Pivot checks that the budget fits available memory before starting. |
| `workers` | Machine core count | Dispatch worker threads. Must be at least one. |
| `datastore_refresh_interval` | `30s` | How often each datastore brings itself up to date with its store: the tables it has and each table's committed files. For a shared remote store, this bounds how stale a query's view of another process's commits can be. This process's own commits are visible immediately. |
| `log` | `info` | What the server logs to stdout. A level for everything, such as `info` or `debug`, or per-target levels such as `info,dispatch=debug`. |

## Server

The `server` section is the PostgreSQL endpoint. Every key has a default, so
the whole section can be left out.

| Key | Default | Description |
| --- | --- | --- |
| `server.bind` | `127.0.0.1:5432` | TCP socket the endpoint binds to. The default is loopback, so set this to reach the server from another machine or from outside a container. |

### TLS

Set both files under `server.tls` to offer TLS to clients:

```yaml
server:
  tls:
    cert: /etc/pivot/server.crt
    key: /etc/pivot/server.key
```

| Key | Default | Description |
| --- | --- | --- |
| `server.tls.cert` | Required | PEM certificate: the server's own, followed by any intermediate certificates a client needs to reach a root it trusts. |
| `server.tls.key` | Required | PEM private key for that certificate, in PKCS#8, PKCS#1, or SEC1 form. Keep it readable only by the server's user. |

Writing the section makes TLS available, not compulsory. A client that asks
to encrypt its connection, such as `psql "sslmode=require"`, gets an
encrypted session, and a client that asks for plaintext still gets one. Without
the section, requests to encrypt are refused and every session is plaintext.

## Metastore

The `metastore` section names the file the server writes to. Without it the
server serves only the configuration's own entries and refuses `CREATE USER`.

```yaml
metastore:
  kind: file
  path: /var/lib/pivot/metastore.yaml
```

| Key | Default | Description |
| --- | --- | --- |
| `metastore.kind` | Required | `file` is the only supported value. |
| `metastore.path` | Required | A YAML file holding the same `datastores`, `secrets`, and `users` maps as the configuration, without a `server` or `metastore` section. It must exist, and only the server should be able to read it. |

The configuration's entries and the file's are served together. A name defined
in both is a startup error, and `CREATE USER` refuses a name the configuration
defines, because the server never rewrites the configuration.

## Sizes and intervals

Sizes such as `memory` and `compact_bytes` are a whole number with a
base-1024 suffix: `k`, `m`, `g`, or `t`, or `kb`, `mb`, `gb`, `tb`. A bare
number is bytes. Suffixes are case-insensitive.

Intervals such as `datastore_refresh_interval` are a whole number with a unit:
`ms`, `s`, `m`, or `h`. The unit is mandatory, and zero is not accepted.

## Related

- [Annotated configuration example](https://github.com/pivotlake/pivot/blob/main/bin/config.example.yaml)
- [Session settings](/docs/reference/statements/set-reset/)
