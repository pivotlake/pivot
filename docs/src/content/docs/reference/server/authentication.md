---
title: "Users & authentication"
description: Define the users that can connect to a Pivot server, and how each one authenticates.
---

A Pivot server accepts only the users it knows. Each user either is trusted,
and logs in without a password, or must supply a password. Users are defined in
the top-level `users` map of the
[configuration file](/docs/reference/configuration/), or created with
[`CREATE USER`](/docs/reference/statements/create-user/).

## Example

```yaml
users:
  dashboard:
    auth:
      method: trust
  analyst:
    auth:
      method: password
      password: replace-with-your-password
```

`dashboard` can connect without a password. `analyst` must supply its
password:

```sh
psql -h 127.0.0.1 -p 5432 -U analyst
```

## Configuration

Each entry under `users` is keyed by the login name.

| Key | Default | Description |
| --- | --- | --- |
| `users.<name>.auth.method` | Required | `trust`, `password`, or `scram-sha-256`. See [Authentication methods](#authentication-methods). |
| `users.<name>.auth.password` | Required for `password` | The user's password, in plain text. Only allowed with `password`. |
| `users.<name>.auth.verifier` | Required for `scram-sha-256` | The password verifier, in Pivot's `pivot-scram-sha-256$...` format. Only allowed with `scram-sha-256`. |

A misspelled key anywhere under `users` is a startup error, so a typo cannot
silently leave a user undefined.

## Authentication methods

| Method | The user logs in with | The file holds |
| --- | --- | --- |
| `trust` | No password | Nothing |
| `password` | A password | The password, in plain text |
| `scram-sha-256` | A password | A verifier derived from the password |

`password` and `scram-sha-256` authenticate the same way. They differ only in
what the file holds.

### Trust

`trust` accepts the login name without checking a password. Use it for users
that connect only from a trusted network, or for local development.

:::caution
Anyone who can reach the server's port can log in as a trusted user. Keep
`server.bind` on loopback, or restrict the port with a firewall, when any user
is trusted.
:::

### Password

`password` takes the user's password as written:

```yaml
users:
  analyst:
    auth:
      method: password
      password: replace-with-your-password
```

The server derives a SCRAM-SHA-256 verifier from the password when it reads
the file, and keeps only the verifier.

Logins use SCRAM-SHA-256, the same exchange as PostgreSQL: the client proves
it knows the password without sending it, so the password never crosses the
network, even without TLS. Standard PostgreSQL clients and drivers support it
without extra settings.

### SCRAM-SHA-256

`scram-sha-256` takes a verifier instead of the password, so the file never
holds the password, and the password cannot be recovered from the verifier.
Logins work exactly as with `password`. The verifier has the form:

```text
pivot-scram-sha-256$4096:<base64 salt>$<base64 salted password>
```

The iteration count must be `4096`. A verifier copied from PostgreSQL's
`pg_authid` has a different format and is rejected. To get a verifier, create
the user with `CREATE USER` and copy it from the metastore file.

:::caution[Protect files that hold users]
Make the configuration file and the metastore file readable only by the user
the server runs as. A `password` is the password itself. A verifier cannot be
turned back into the password, but it should be protected just as carefully.
:::

## Creating users

[`CREATE USER`](/docs/reference/statements/create-user/) adds a user while the
server runs:

```sql
CREATE USER analyst PASSWORD 'replace-with-your-password';
CREATE USER dashboard;                     -- trust, no password
```

`CREATE USER` writes the user to the
[metastore file](/docs/reference/configuration/#metastore) with a
`scram-sha-256` verifier, so the server must be configured with one. The user
can log in immediately.

### Where users are defined

| Defined in | Added by | Changed by |
| --- | --- | --- |
| Configuration file | Editing the file | Editing the file and restarting the server |
| Metastore file | `CREATE USER` | Editing the file while the server is stopped |

The server serves the users of both files together. The same name defined in
both files is a startup error, and `CREATE USER` refuses a name the
configuration file defines, because the server never rewrites that file.

## Built-in user

A user named `pivot` is always available, so a new server can be reached
before any user is configured. It uses trust authentication.

To require a password for `pivot`, define it under `users` in either file with
the `password` or `scram-sha-256` method. That definition replaces the
built-in user entirely:

```yaml
users:
  pivot:
    auth:
      method: password
      password: replace-with-your-password
```

`CREATE USER pivot` is refused, because the name is already taken.

## Unknown users

A login with a name no file defines is refused. To a client, the refusal
looks the same as a wrong password, so failed logins do not reveal which user
names exist.

## Permissions

Every user has full access to every datastore the server serves. Pivot does
not have roles or privileges.

## TLS

Authentication works the same over plaintext and TLS connections. To encrypt
connections, configure [`server.tls`](/docs/reference/configuration/#tls).

Pivot does not support SCRAM channel binding (`SCRAM-SHA-256-PLUS`). A client
connecting with `channel_binding=require` is refused. Use the default,
`channel_binding=prefer`, or `disable`.

## Limitations

- Users cannot be changed or dropped with SQL: there is no `ALTER USER` or
  `DROP USER`. Edit the file that defines the user instead.
- The configuration file and the metastore file are read once, at startup.
  Hand edits to either take effect after a restart.

## Related

- [`CREATE USER`](/docs/reference/statements/create-user/)
- [Configuration file](/docs/reference/configuration/)
- [Datastores & storage credentials](/docs/reference/server/datastores/)
