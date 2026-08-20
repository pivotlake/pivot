---
title: Command line
description: The pivot binary, its subcommands, and the shell it opens.
sidebar:
  order: 7
---

One binary, `pivot`, carries both entry points.

```sh
pivot <COMMAND>
```

| Command | Does |
| --- | --- |
| [`pivot open`](#pivot-open) | Open one datastore in the local SQL shell |
| [`pivot server`](#pivot-server) | Run the server in the foreground |

`pivot --version` prints the version, and `pivot <COMMAND> --help` prints a
command's own usage.

## pivot open

```sh
pivot open <DATASTORE_LOCATION>
```

Opens a datastore and drops into the SQL shell, in-process: there is no server
and no wire protocol involved, and results are rendered straight from the
dispatch workers.

The location is either a local directory, which is created if it does not
exist, or an object-store URI:

| Location | Backend |
| --- | --- |
| `/var/lib/pivot` | Local filesystem |
| `file:///var/lib/pivot` | Local filesystem |
| `s3://bucket/prefix` | S3, also spelled `s3a://` |
| `gs://bucket/prefix` | Google Cloud Storage |

Credentials come from the environment here, not from a config file. S3 reads
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, its region from `AWS_REGION`
or `AWS_DEFAULT_REGION` (`us-east-1` when neither is set), and an optional
`AWS_ENDPOINT_URL` naming a path-style S3-compatible endpoint such as MinIO.
Google Cloud Storage follows Application Default Credentials:
`GOOGLE_APPLICATION_CREDENTIALS`, then the `gcloud` login file, then the
instance metadata server.

A local datastore directory may be open in only one pivotdb process at a time,
so a directory a server is serving cannot also be opened here.

### Shell

Statements are terminated with a semicolon and may span lines. String
literals, quoted identifiers, dollar-quoted bodies and nested block comments
are all recognised while framing, so a pasted statement holding a semicolon
inside one of them stays whole.

| Meta-command | Does |
| --- | --- |
| `\q` | Quit, as does `quit;` or end-of-file |
| `\h`, `\help` | List these commands |
| `\timing` | Toggle per-statement timing |
| `\timing on`, `\timing off` | Set it explicitly. `true`/`false` and `1`/`0` also work |

`COPY ... FROM STDIN` needs the PostgreSQL protocol's copy-in channel, which
the shell does not have, so it is refused there.

## pivot server

```sh
pivot server --config <FILE> [--metastore-file <FILE>]
```

Runs the server in the foreground until it is interrupted or its service
manager terminates it.

| Flag | Meaning |
| --- | --- |
| `--config <FILE>` | Required. The YAML file describing this instance. See [Configuration](/docs/reference/configuration/) |
| `--metastore-file <FILE>` | A second YAML file of datastores, secrets and users, without the surrounding `metastore` key, merged into the configuration. Also where [`CREATE USER`](/docs/reference/sql-statements/#create-user) writes |

Logging is controlled by `RUST_LOG`, which defaults to `info`:

```sh
RUST_LOG=server=debug,dispatch=info pivot server --config pivot.yaml
```

### Connecting

The endpoint speaks the PostgreSQL v3 wire protocol, so any PostgreSQL client
works:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

The simple and extended query protocols are both served, along with copy-in and
query cancellation. A prepared statement with bind parameters is refused, since
parameters are not implemented yet.

TLS is offered when the config file carries a `tls` block:

```sh
psql "host=pivot.example.com port=5432 user=pivot sslmode=verify-full sslrootcert=/etc/ssl/ca.crt"
```
