---
title: "Users & authentication"
description: Configure trusted users and SCRAM password authentication.
---

Users are declared in the top-level `users` map of the server's YAML
configuration or added with
[CREATE USER](/docs/reference/statements/create-user/), which stores them in
the [metastore file](/docs/reference/configuration/#metastore).

## Example

Create a user with a password through SQL. Replace the example password:

```sql
CREATE USER analyst PASSWORD 'replace-with-your-password';
```

A PostgreSQL client can then connect using that login:

```sh
psql -h 127.0.0.1 -p 5432 -U analyst -W
```

## Authentication methods

| Authentication method | Configuration |
| --- | --- |
| Trust | `auth: { method: trust }`. No password is checked. |
| SCRAM-SHA-256 | `auth: { method: scram-sha-256, verifier: "pivot-scram-sha-256$..." }`. Store the precomputed verifier, not the password. |

The built-in `pivot` user uses trust authentication unless it is configured
explicitly.

## Trust authentication

Trust accepts the supplied login name without checking a password.

```yaml
users:
  local_analyst:
    auth:
      method: trust
```

This is a fragment to add to a configuration that also declares its
datastores.

## Password authentication

A YAML user with `method: scram-sha-256` must supply a precomputed `verifier`
in Pivot's `pivot-scram-sha-256$...` format. The YAML field takes the verifier,
not a plaintext password. Use `CREATE USER ... PASSWORD` when adding a user
from a password through SQL.

Trust configuration cannot contain a verifier, and SCRAM configuration cannot
omit one.

## Built-in user

The `pivot` user is always available. Configure it explicitly under `users`
to replace its default trust authentication with SCRAM.

## TLS

Authentication and transport encryption are configured separately. See
[TLS settings](/docs/reference/configuration/#tls) to offer encrypted
connections.

## Related

- [CREATE USER](/docs/reference/statements/create-user/)
- [Server configuration](/docs/reference/configuration/)
- [Datastores and storage credentials](/docs/reference/server/datastores/)
