---
title: "CREATE USER"
description: Add a login user with password or trust authentication.
sidebar:
  order: 6
---

`CREATE USER` adds a login user to the server. The user is written to the
[metastore file](/docs/reference/configuration/#metastore), so the statement
is refused when the configuration names no metastore, and it refuses a name
the configuration already defines.

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

## Related

- [Users and authentication](/docs/reference/server/authentication/)
- [Server TLS configuration](/docs/reference/configuration/#tls)
