---
title: "CREATE USER"
description: Add a login user with password or trust authentication.
sidebar:
  order: 6
---

`CREATE USER` adds a login user to the server.

## Example

Replace the example password before running:

```sql
CREATE USER analyst PASSWORD 'replace-with-your-password';
```

## Syntax

```sql
CREATE USER name [PASSWORD 'password'];
```

## Authentication

`name` identifies the login. Supplying `PASSWORD` configures password
authentication.

Omitting `PASSWORD` creates a trusted user: the server accepts the login name
without checking a password.

```sql
CREATE USER local_analyst;
```

For users declared in YAML, configure the authentication method under
`metastore.users`. Password authentication there uses a precomputed SCRAM
verifier, not the password itself.

## Related

- [Users and authentication](/docs/reference/server/authentication/)
- [Server TLS configuration](/docs/reference/configuration/#tls)
