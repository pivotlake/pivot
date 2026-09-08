---
title: "Datastores & storage credentials"
description: Connect local and object storage and configure table maintenance.
---

A datastore exposes a collection of schemas and tables. Define datastores
under `metastore.datastores` in the server's YAML configuration.

## Example

```yaml
metastore:
  datastores:
    local:
      kind: pivot
      location: ./pivot-data
      default: true
    lake:
      kind: pivot
      location: s3://example-bucket/pivot/
  secrets:
    lake-storage:
      type: s3
      scope: s3://example-bucket/
      region: us-east-1
      access_key_id: replace-me
      secret_access_key: replace-me
```

Replace the bucket and credential values with your own. Tables in `lake` can
be addressed as `lake.schema.table`.

## Datastores

Exactly one datastore must set `default: true`.

| Key | Default | Description |
| --- | --- | --- |
| `kind` | Required | Datastore implementation. `pivot` is the only supported value. |
| `location` | Required | Local path, `s3://` URI, or `gs://` URI. |
| `default` | `false` | Makes this the target of unqualified SQL names. Exactly one must be true. |
| `compact` | `true` | Runs background compaction. Enable it in only one process per shared datastore. |
| `compact_bytes` | `64m` | Compaction output target; files strictly below half this size are small-file candidates, and an individual row group may exceed it. |
| `compact_merge_bytes` | 1.3 times `compact_bytes` | Accumulated small-file bytes that immediately trigger a merge. |
| `compact_min_files` | `100` | File count at which the small-file balance fallback may merge. |
| `compact_parallelism` | `3` | Maximum number of disjoint compaction merges rewritten concurrently. Values below one run one merge at a time. Each merge in flight holds its decoded input rows in memory. |
| `vacuum` | `true` | Deletes expired unreferenced files and old log entries. Enable it in only one process per shared datastore. |

## Maintenance

Background compaction merges small files and re-sorts overlapping files.
`compact_bytes` controls the output target, and `compact_parallelism` limits
how many disjoint merges are rewritten at once. Each active merge holds its
decoded input rows in memory.

Enable compaction and vacuum in only one process per shared datastore. Set
`compact: false` and `vacuum: false` on the other processes.

Use [COMPACT](/docs/reference/statements/compact/) to request a compaction round
through SQL. Vacuum removes expired unreferenced files and old log entries;
[DROP TABLE](/docs/reference/statements/drop-table/) does not immediately
remove data files.

## Storage credentials

Secret names are user-defined. `scope` is an optional URI prefix; the most
specific secret covering a datastore location is selected.

| Secret type | Keys |
| --- | --- |
| `s3` | `type: s3`, optional `scope`, required `region`, `access_key_id`, and `secret_access_key`, plus optional `endpoint` for S3-compatible storage. |
| `gcs` | `type: gcs`, optional `scope`, and required `credentials_file`. Without a matching secret, GCS uses ambient Application Default Credentials. |

Two secrets cannot claim the same scope.

### S3 and compatible storage

An S3 location requires a matching S3 secret. Set `endpoint` for an
S3-compatible service, such as MinIO. A secret can cover multiple datastores.

### Google Cloud Storage

For an explicit credentials file, add a GCS secret under `metastore.secrets`:

```yaml
metastore:
  secrets:
    google:
      type: gcs
      scope: gs://example-bucket/
      credentials_file: /etc/pivot/gcs-key.json
```

Without a matching GCS secret, Pivot uses ambient Application Default
Credentials, including `GOOGLE_APPLICATION_CREDENTIALS`.

## Related

- [Server configuration](/docs/reference/configuration/)
- [CREATE TABLE](/docs/reference/statements/create-table/)
- [System tables](/docs/reference/system-tables/)
