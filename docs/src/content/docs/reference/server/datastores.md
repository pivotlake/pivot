---
title: "Datastores & storage credentials"
description: Configure pivotlake datastores, Iceberg catalogs, storage credentials, and table maintenance.
---

A datastore is a named collection of schemas and tables that a server serves.
A server can serve several datastores at once, and a query can join tables
across them. Datastores are defined in the top-level `datastores` map of the
[configuration file](/docs/reference/configuration/), and the credentials they
are opened with in the `secrets` map.

## Example

```yaml
datastores:
  local:
    kind: pivotlake
    location: /var/lib/pivot/datastores/local
    default: true
  events:
    kind: pivotlake
    location: s3://example-bucket/pivot/events/
  lake:
    kind: iceberg
    uri: https://catalog.example.com/api
    warehouse: s3://example-bucket/warehouse/
    secret: lake-catalog

secrets:
  example-bucket:
    type: s3
    scope: s3://example-bucket/
    region: us-east-1
    access_key_id: replace-me
    secret_access_key: replace-me
  lake-catalog:
    type: iceberg
    credential: client-id:client-secret
```

This server serves three datastores. `local` is the default, so an
unqualified table name refers to it. `events` and the Iceberg tables in `lake`
are read with the `example-bucket` secret, and the Iceberg catalog itself is
authenticated to with `lake-catalog`.

## Datastores

Each entry under `datastores` is keyed by the datastore's name, which is how
SQL refers to it. `kind` selects the implementation:

| Kind | Description |
| --- | --- |
| `pivotlake` | Pivot's own read-write tables, stored as Iceberg or Delta Lake tables of Parquet files in a local directory or a bucket. Also accepted as `pivot`. See [Table formats](#table-formats). |
| `iceberg` | The tables of an Iceberg REST catalog (`uri`), read and written in place. |

### Default datastore

Exactly one datastore must set `default: true`. It is the current database:
unqualified table names, and statements such as `CREATE TABLE` that name no
datastore, resolve against it. Its name is free, and it need not be called
`default`. The server refuses to start when no datastore, or more than one, is
marked as the default.

### Naming tables

A table's full name is `datastore.schema.table`. An unqualified name refers
to the `main` schema of the default datastore:

```sql
SELECT count(*) FROM clicks;              -- local.main.clicks
SELECT count(*) FROM lake.sales.orders;   -- the orders table in the lake datastore
```

A two-part name such as `sales.orders` can name either a schema in the default
datastore or a datastore's `main` schema. Use the full three-part name when a
datastore and a schema share a name.

### pivotlake datastores

```yaml
datastores:
  events:
    kind: pivotlake
    location: s3://example-bucket/pivot/events/
```

| Key | Default | Description |
| --- | --- | --- |
| `location` | Required | Where the datastore's tables are stored. See the table below. |
| `table_format` | `iceberg` | The table format a database that does not exist yet at `location` is created in: `iceberg` or `delta` (also spelled `deltalake`). See [Table formats](#table-formats). |
| `default` | `false` | Whether this is the [default datastore](#default-datastore). |
| `compact` | `true` | Run background compaction. See [Compaction](#compaction). |
| `compact_bytes` | `64m` | The target size of a compacted file. |
| `compact_merge_bytes` | 1.3 × `compact_bytes` | How many bytes of small files trigger a merge. |
| `compact_min_files` | `100` | How many small files may trigger a merge even below `compact_merge_bytes`. |
| `compact_parallelism` | `1` | How many merges run at once. |
| `vacuum` | `true` | Run background vacuum. See [Vacuum](#vacuum). |

The storage backend is chosen from the location's scheme:

| Location | Backend |
| --- | --- |
| `/var/lib/pivot/datastores/local`, `./data` | Local directory. A relative path is resolved against the server's working directory. |
| `file:///var/lib/pivot/datastores/local` | Local directory. |
| `s3://bucket/prefix/`, `s3a://bucket/prefix/` | Amazon S3 or an S3-compatible service. |
| `gs://bucket/prefix/` | Google Cloud Storage. |

A local directory is created if it does not exist, and only one process at a
time can open it. A location with any other scheme is a startup error.

#### Table formats

A pivotlake database keeps every one of its tables in one table format,
chosen when the database is created and recorded in its index,
`_pivot_manifest.json` at the location's root. A database that does not
exist yet is created in the format `table_format` names, `iceberg` unless it says
`delta`; an existing database keeps the format it was created in, whatever
`table_format` says, and naming the other one is a startup error. A database
created before the index recorded a format is a Delta Lake database. Pivot
needs no catalog service for either format: the index names the schemas and
tables and maps each table to a directory named after its identity, and the
table's own metadata lives in that directory.

| Format | Table layout | Readable by |
| --- | --- | --- |
| `iceberg` | An Iceberg format-version-3 table: metadata under `<table>/metadata/` as `v<N>.metadata.json` files, one per commit, with their manifest lists and manifests. | Any Iceberg reader that opens a table by its path, such as DuckDB's `iceberg_scan('<location>/<table directory>')`: Pivot rewrites the `version-hint.text` such readers look for after every commit. Spark's `HadoopTables` path reader does not accept the zero-padded metadata file names; Spark reads these tables through a catalog. |
| `delta` | A Delta Lake table: a `_delta_log` of commit files and checkpoints. | Any Delta Lake reader, such as DuckDB's `delta_scan`. |

A commit to an Iceberg table creates the next numbered metadata file with a
conditional write, and a commit to a Delta table appends the next log entry
the same way, so writers in several processes can share one location:
exactly one lands each version, and the others reload and retry. The store
must support conditional writes (Amazon S3, Google Cloud Storage, MinIO, or
a local filesystem); a location on an S3-compatible service without them
fails every commit.

### Iceberg datastores

An `iceberg` datastore serves the tables of a REST catalog: every table the
catalog lists can be queried, and `INSERT`, `COPY`, `CREATE TABLE` and
`DROP TABLE` go through the catalog. The Iceberg tables Pivot keeps in a
directory of its own belong to a `pivotlake` datastore in the `iceberg`
[table format](#table-formats).

```yaml
datastores:
  lake:
    kind: iceberg
    uri: https://catalog.example.com/api
    warehouse: s3://example-bucket/warehouse/
    secret: lake-catalog
```

| Key | Default | Description |
| --- | --- | --- |
| `uri` | Required | The Iceberg REST catalog's base URI. |
| `warehouse` | None | The warehouse to serve, for a catalog that serves several. |
| `secret` | None | The name of an [`iceberg` secret](#iceberg-catalog-secrets) to authenticate to the catalog with. Omit it for a catalog that requires no authentication. |
| `properties` | `{}` | Additional REST client properties, passed to the catalog client as written. |
| `default` | `false` | Whether this is the [default datastore](#default-datastore). |

Each catalog namespace appears as a schema. Table files are read from and
written to the locations the catalog reports, using the credentials the
catalog vends, or the [storage secret](#storage-credentials) whose scope
covers the file when it vends none. Vended credentials are renewed at every
[refresh](#refresh), in place, so a statement that runs longer than they last
keeps reading and writing; when the catalog says when they expire, a refresh
runs early enough to renew them first.

Format version 3 deletion vectors are supported. A table carrying any other
kind of delete file is refused rather than read with the deleted rows present.

If the catalog becomes unreachable, the datastore keeps serving the tables it
last loaded and retries on the next [refresh](#refresh).

#### Writing to catalog tables

An `INSERT` or `COPY` writes its Parquet files under the table's data path
(`write.data.path`, else `<location>/data/`), cut by the table's partition
spec: a transform of a column (`day(ts)`, `bucket(16, id)`, `truncate(4, s)`)
is computed per row, and each file's partition values are recorded in the
manifest, so every engine prunes by them. The files are committed as one
append snapshot through the catalog's `updateTable`, which admits the commit
only if no other writer moved the table's `main` branch in the meantime; a
commit that lost that race is rebuilt on the winner. The session reads its own writes at
once, before the next refresh. `CREATE TABLE` registers a format-version-3
table through the catalog, which assigns its location; it takes the same
`partition_by` and `sort_by` options as a pivotlake table. `DROP TABLE`
unregisters the table and leaves its files in place, as `DROP TABLE` without
`PURGE` does in other engines.

Pivot never compacts, expires snapshots of, or vacuums a catalog table: that
is the catalog's own maintenance, and other engines may share the table.

A write is refused, naming the reason, when the table's shape is one this
writer cannot honour: a partition field whose key it cannot compute (a source
that is not a top-level column, a field named like a column, or a transform
the Iceberg library cannot evaluate for the column's type), a format version
below 2, delete files or deletion vectors in the current snapshot, or a
column type Pivot cannot represent. A `NULL` in a column the table declares
required fails the statement.

## Storage credentials

The `secrets` map holds two kinds of secret, told apart by `type`:

| Type | Authenticates to | Chosen by |
| --- | --- | --- |
| `s3`, `gcs` | A bucket: a pivotlake datastore's files, an Iceberg table's files, or a Parquet file read with `read_parquet`. | The secret's [`scope`](#scopes), matched against the location being read. |
| `iceberg` | An Iceberg REST catalog. | The Iceberg datastore's `secret` key. |

An Iceberg datastore can use both kinds: an `iceberg` secret to talk to the
catalog, and `s3` or `gcs` secrets to read its tables' files when the catalog
does not vend credentials.

### S3 secrets

```yaml
secrets:
  example-bucket:
    type: s3
    scope: s3://example-bucket/
    region: us-east-1
    access_key_id: replace-me
    secret_access_key: replace-me
```

| Key | Default | Description |
| --- | --- | --- |
| `type` | Required | `s3`. |
| `access_key_id` | Required | The access key ID. |
| `secret_access_key` | Required | The secret access key. |
| `scope` | Every `s3://` location | The `s3://bucket/prefix` the secret covers. See [Scopes](#scopes). |
| `region` | Discovered from the bucket | The bucket's region. |
| `endpoint` | AWS S3 | An S3-compatible endpoint, such as `http://minio:9000`. Requests to it use path-style addressing. |

### GCS secrets

```yaml
secrets:
  analytics-gcs:
    type: gcs
    scope: gs://example-bucket/
    credentials_file: /etc/pivot/gcs-key.json
```

| Key | Default | Description |
| --- | --- | --- |
| `type` | Required | `gcs`. |
| `credentials_file` | Required | A Google service-account or authorized-user JSON key file. |
| `scope` | Every `gs://` location | The `gs://bucket/prefix` the secret covers. See [Scopes](#scopes). |

### Scopes

A secret's `scope` is a URI prefix. A location is opened with the secret whose
scope is the longest prefix of it, so a more specific secret overrides a
broader one:

```yaml
secrets:
  analytics:
    type: s3
    scope: s3://analytics/            # everything in the bucket...
    access_key_id: replace-me
    secret_access_key: replace-me
  analytics-archive:
    type: s3
    scope: s3://analytics/archive/    # ...except under archive/
    access_key_id: replace-me
    secret_access_key: replace-me
```

Scopes match whole path segments: `s3://analytics/arch` does not cover
`s3://analytics/archive/`. Trailing and doubled slashes make no difference.
A secret without a `scope` covers every location of its type.

Two secrets may not have the same scope, and a scope must use its secret's
scheme: an `s3` secret scoped to `gs://example-bucket/` is a startup error.

### Locations without a secret

When no secret covers a location:

- **S3** locations are read with anonymous, unsigned requests. This works for
  public buckets and for S3-compatible services that require no
  authentication.
- **GCS** locations use Application Default Credentials: the key file named by
  `GOOGLE_APPLICATION_CREDENTIALS`, then the credentials written by
  `gcloud auth application-default login`, then the identity of the Google
  Cloud instance the server runs on.

### Iceberg catalog secrets

An `iceberg` secret authenticates to an Iceberg REST catalog. It has no
scope: the Iceberg datastore names it with its `secret` key.

```yaml
secrets:
  lake-catalog:
    type: iceberg
    credential: client-id:client-secret
```

| Key | Default | Description |
| --- | --- | --- |
| `type` | Required | `iceberg`. |
| `token` | None | A bearer token sent with every catalog request. |
| `credential` | None | An OAuth2 client credential, written `client_id:client_secret`, exchanged for a token. |
| `oauth2_server_uri` | The catalog's token endpoint | The OAuth2 token endpoint, when it is not the catalog's own. |
| `oauth2_scope` | None | The OAuth2 scope to request with `credential`. |

Set exactly one of `token` and `credential`.

:::caution[Protect files that hold secrets]
S3 keys, Iceberg tokens, and Iceberg credentials are written in the file
itself. Make the configuration file and the metastore file readable only by
the user the server runs as.
:::

## Maintenance

A server keeps each pivotlake datastore up to date and tidy with three background
tasks. Iceberg datastores are only refreshed.

| Task | Runs | Setting |
| --- | --- | --- |
| [Refresh](#refresh) | Every 30 seconds | `datastore_refresh_interval` in the [configuration file](/docs/reference/configuration/#instance-settings) |
| [Compaction](#compaction) | 30 seconds after the previous round finishes | `compact` and the `compact_*` keys, per datastore |
| [Vacuum](#vacuum) | Every hour | `vacuum`, per datastore |

### Refresh

Every refresh interval, each datastore reloads the tables it holds and the
files each table has committed. Commits made by this server are visible
immediately. Commits made by another process to a shared datastore, or tables
added to an Iceberg catalog, become visible at the next refresh.

### Compaction

Many small inserts leave many small Parquet files. Compaction rewrites them
into files of about `compact_bytes`, and re-sorts files whose sort-key ranges
overlap within a partition, so that queries read fewer files and skip more of
them.

A file smaller than half of `compact_bytes` is a small file. Small files are
merged once their total size reaches `compact_merge_bytes`, or earlier once
there are `compact_min_files` of them and they are close enough in size to be
worth merging.

Each merge holds its input rows in memory until its output is written. Raise
`compact_parallelism` only on a server with memory to spare.

A compaction round goes through every table until none has anything left to
merge, however long that takes, and the next round starts 30 seconds after it
finishes. Each round first reloads the table it compacts from its log, so it
sees new files as soon as they are committed, without waiting for the next
[refresh](#refresh). A table another process creates in a shared datastore is
compacted once a refresh has loaded it.

To compact a table on demand, use [`COMPACT`](/docs/reference/statements/compact/).

### Vacuum

Compaction, `DROP TABLE`, and new commits leave files that the current
version of a table no longer references. Vacuum deletes them once they are
older than the table's retention window:

| Files | Retention |
| --- | --- |
| Data files no longer referenced, including a dropped table's | 4 hours |
| Iceberg snapshots, with their manifests and metadata files, superseded for longer than the window | 4 hours |
| Delta log files made redundant by a checkpoint | 24 hours |

A query that is still reading an older version of a table keeps working for
as long as that version's files are retained.

### Sharing a datastore between processes

Several servers, or a server and `pivot open` shells, can serve the same
datastore in a bucket. Each sees the others' commits at its next refresh.

:::caution[Run maintenance in one process per datastore]
Compaction and vacuum rewrite and delete files, so only one process should run
them for a given datastore. Set `compact: false` and `vacuum: false` on the
datastore in every other server. `pivot open` never runs them.
:::

A local directory cannot be shared: only one process can open it at a time.

## Configuration file and metastore file

Datastores and secrets can be defined in the configuration file, in the
[metastore file](/docs/reference/configuration/#metastore), or in both. The
server serves the entries of both files together. The same name defined in
both files is a startup error, and so is a scope claimed by two secrets across
the files.

## Related

- [Configuration file](/docs/reference/configuration/)
- [System tables](/docs/reference/system-tables/): inspect datastores, tables, and files.
- [`COMPACT`](/docs/reference/statements/compact/)
- [Command-line interface](/docs/reference/cli/): open a single datastore in a shell.
