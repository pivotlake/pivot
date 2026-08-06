# Configuration

Pivot reads one YAML file at startup. It has an optional `server` section and a
required `metastore` section.

```sh
pivotdb-server --config pivot.yaml
```

## Server settings

Every server setting has a default, so this section can be omitted.

```yaml
server:
  bind: 0.0.0.0:5432
  memory: 32g
  workers: 16
  http_bind: 127.0.0.1:8081
  disk_cache:
    dir: /var/cache/pivot
    size: 64g
    max_objects: 65536
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `bind` | `127.0.0.1:5432` | PostgreSQL endpoint |
| `memory` | 80% of physical memory | Buffer-pool budget |
| `workers` | Number of CPU cores | Dispatch worker threads |
| `http_bind` | Disabled | Dashboard and HTTP API endpoint |
| `disk_cache` | Disabled | Persistent cache for remote reads |

Set `RUST_LOG` to adjust logging:

```sh
RUST_LOG=server=debug,dispatch=info pivotdb-server --config pivot.yaml
```

## Datastores

Exactly one datastore must set `default: true`. Unqualified table names resolve
against it. Every other datastore is available as a database and can be queried
with a three-part name such as `archive.main.events`.

### Local storage

```yaml
metastore:
  refresh_interval: 30s
  datastores:
    local:
      kind: delta
      location: /var/lib/pivot
      default: true
```

### S3 storage

```yaml
metastore:
  datastores:
    lake:
      kind: delta
      location: s3://analytics/pivot/
      default: true
      region: us-east-1
      access_key_id: replace-me
      secret_access_key: replace-me
      # endpoint: http://localhost:9000
```

S3 credentials are stored inline. Restrict access to the configuration file.
Use `endpoint` for an S3-compatible service such as MinIO.

### Multiple datastores

```yaml
metastore:
  datastores:
    hot:
      kind: delta
      location: /var/lib/pivot/hot
      default: true
    archive:
      kind: delta
      location: s3://analytics/archive/
      region: us-east-1
      access_key_id: replace-me
      secret_access_key: replace-me
```

```sql
SELECT h.id, a.event_type
FROM hot.main.events AS h
JOIN archive.main.event_details AS a ON h.id = a.id;
```

## Compaction

Compaction is disabled by default and configured per datastore:

```yaml
metastore:
  datastores:
    local:
      kind: delta
      location: /var/lib/pivot
      default: true
      compact: true
      compact_bytes: 128m
      compact_min_files: 8
```

Only one Pivot process should compact a given datastore.

## Users

Without a `users` section, Pivot provides a trusted user named `pivot`. Defining
users replaces that default with an allowlist.

```yaml
metastore:
  users:
    local_reader:
      auth:
        method: trust
    analytics:
      auth:
        method: scram-sha-256
        verifier: "pivot-scram-sha-256$4096:base64-salt$base64-salted-password"
```

`trust` does not prove identity. Anyone who supplies that user name is accepted.
Use the precomputed SCRAM verifier form when a password is required.

The fully commented configuration is in
[`server/config.example.yaml`](https://github.com/Epsio-Labs/pivotdb/blob/main/server/config.example.yaml).
