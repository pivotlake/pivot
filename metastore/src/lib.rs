//! The provider-neutral **metastore** interface: the server's source of
//! configuration about which named datastores it serves and which users may
//! authenticate to it.
//!
//! [`Metastore`] is the trait the server holds; the datastores it returns are
//! resolved however the implementation likes. Concrete providers live in sibling
//! crates so a YAML file, PostgreSQL, or another backend can be selected without
//! coupling this interface to its configuration format or datastore implementation.
//!
//! A user's authentication method crosses that boundary as [`UserAuth`].
//! SCRAM credentials use [`ScramVerifier`], whose serialised form
//! ([`format_scram_verifier`] / [`parse_scram_verifier`]) lives here too so
//! every provider stores the same text.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use catalog::Datastore;
use dispatch::DataFlowDispatcher;
use std::collections::HashMap;
use std::sync::Arc;

/// The user provided by a metastore configuration that does not define any
/// users explicitly. It authenticates with [`UserAuth::Trust`].
pub const DEFAULT_USER_NAME: &str = "pivot";

/// The iteration count every stored [`ScramVerifier`] is derived with, and the
/// one the server announces to a client in the SCRAM server-first message. The
/// two must agree for the client's proof to verify, so it is fixed server-wide
/// rather than carried per user. 4096 is the RFC 5802 minimum and PostgreSQL's
/// own default.
pub const SCRAM_ITERATIONS: usize = 4096;

/// A user's stored SCRAM-SHA-256 credential: the salt the password was derived
/// with, and `SaltedPassword` itself, `Hi(Normalize(password), salt,
/// SCRAM_ITERATIONS)` as RFC 5802 defines it.
///
/// The cleartext password is not recoverable from either field, and neither
/// crosses the wire during authentication: the client proves knowledge of the
/// password against them instead.
#[derive(Clone)]
pub struct ScramVerifier {
    pub salt: Vec<u8>,
    pub salted_password: Vec<u8>,
}

/// Redacted: a verifier is a credential, and the derived form of a password an
/// offline attack still works against, so it must not reach a log or an error
/// message through a `{:?}` of some struct that happens to hold one.
impl std::fmt::Debug for ScramVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScramVerifier(redacted)")
    }
}

/// How a configured user authenticates to the PostgreSQL endpoint.
///
/// `Trust` deliberately proves nothing: anyone who supplies that user name is
/// accepted. It is explicit per user so omitting a password by mistake cannot
/// silently grant access.
#[derive(Clone, Debug)]
pub enum UserAuth {
    Trust,
    ScramSha256(ScramVerifier),
}

/// The server's configuration source. It defines the datastores to serve and
/// the users that may authenticate; it is the seam further server configuration
/// (secrets) would extend.
pub trait Metastore: Send + Sync {
    /// Open every configured datastore, keyed by name. Each is a
    /// [`Datastore`] over the datastore's object store; opening reads the
    /// tables' footers over `dispatcher`. The datastore named by
    /// [`default_datastore_name`](Self::default_datastore_name) must be present.
    fn open_datastores(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<HashMap<String, Arc<dyn Datastore>>>;

    /// The name of the default datastore: DuckDB's current database, the target
    /// of unqualified table names. Known from the parsed configuration, so this
    /// does no I/O.
    fn default_datastore_name(&self) -> &str;

    /// The authentication method for `username`, or `Ok(None)` when no such
    /// user is configured. The server calls this for every login rather than
    /// caching credentials at startup, so implementations may reflect users
    /// added or passwords changed while the server is running. A provider that
    /// consults a live backend returns `Err` when that backend cannot be
    /// reached, so an outage fails the login as a server error rather than
    /// masquerading as an unknown user.
    fn user_auth(&self, username: &str) -> Result<Option<UserAuth>>;
}

/// Provider errors cross the trait boundary without making this crate depend on
/// a concrete metastore implementation.
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;

/// The tag a serialised [`ScramVerifier`] starts with.
///
/// Deliberately not PostgreSQL's own `SCRAM-SHA-256`: that format's final field
/// is `StoredKey:ServerKey`, whereas this one carries `SaltedPassword`, so a
/// value copied from `pg_authid.rolpassword` would parse but never verify. A
/// distinct tag rejects it at startup instead.
const SCRAM_VERIFIER_TAG: &str = "pivot-scram-sha-256";

/// Render `verifier` as the single line a metastore stores for a user:
/// `pivot-scram-sha-256$<iterations>:<base64 salt>$<base64 salted password>`.
pub fn format_scram_verifier(verifier: &ScramVerifier) -> String {
    format!(
        "{SCRAM_VERIFIER_TAG}${SCRAM_ITERATIONS}:{}${}",
        BASE64.encode(&verifier.salt),
        BASE64.encode(&verifier.salted_password)
    )
}

/// Parse the stored form produced by [`format_scram_verifier`]. The error is a
/// message describing what is wrong with `text`, for a provider to attach its
/// own context (which file, which user) to.
pub fn parse_scram_verifier(text: &str) -> std::result::Result<ScramVerifier, String> {
    let (tag, fields) = text.split_once('$').unwrap_or((text, ""));
    if tag != SCRAM_VERIFIER_TAG {
        if tag.eq_ignore_ascii_case("SCRAM-SHA-256") {
            return Err(format!(
                "this is a PostgreSQL `{tag}` verifier, whose final field is \
                 `StoredKey:ServerKey`; expected PivotDB's `{SCRAM_VERIFIER_TAG}` \
                 salted-password format"
            ));
        }
        return Err(format!(
            "expected a `{SCRAM_VERIFIER_TAG}$<iterations>:<base64 salt>$\
             <base64 salted password>` verifier"
        ));
    }
    let (derivation, salted_password) = fields
        .split_once('$')
        .ok_or("expected `<iterations>:<base64 salt>$<base64 salted password>` after the tag")?;
    let (iterations, salt) = derivation
        .split_once(':')
        .ok_or("expected `<iterations>:<base64 salt>` before the salted password")?;

    let iterations: usize = iterations
        .parse()
        .map_err(|_| format!("iteration count `{iterations}` is not a number"))?;
    // The count is announced to the client, which derives its proof with it, so
    // one that disagrees with the stored salted password can only fail to log
    // in. Reject it here rather than at the failed login.
    if iterations != SCRAM_ITERATIONS {
        return Err(format!(
            "iteration count {iterations} differs from the {SCRAM_ITERATIONS} this server derives with"
        ));
    }
    let salt = BASE64
        .decode(salt)
        .map_err(|source| format!("salt is not valid base64: {source}"))?;
    let salted_password = BASE64
        .decode(salted_password)
        .map_err(|source| format!("salted password is not valid base64: {source}"))?;
    if salt.is_empty() {
        return Err("salt is empty".to_string());
    }
    // SHA-256 output width: `Hi` is an HMAC-SHA-256 cascade, so any other length
    // did not come from this scheme.
    if salted_password.len() != 32 {
        return Err(format!(
            "salted password is {} bytes, expected 32",
            salted_password.len()
        ));
    }

    Ok(ScramVerifier {
        salt,
        salted_password,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verifier() -> ScramVerifier {
        ScramVerifier {
            salt: vec![7; 16],
            salted_password: vec![9; 32],
        }
    }

    #[test]
    fn a_formatted_verifier_parses_back_unchanged() {
        let original = verifier();

        let parsed = parse_scram_verifier(&format_scram_verifier(&original)).unwrap();

        assert_eq!(parsed.salt, original.salt);
        assert_eq!(parsed.salted_password, original.salted_password);
    }

    #[test]
    fn a_postgres_verifier_is_rejected_by_name() {
        let stored = "SCRAM-SHA-256$4096:c2FsdA==$c3RvcmVk:c2VydmVy";

        let error = parse_scram_verifier(stored).unwrap_err();

        assert!(error.contains("PostgreSQL"), "{error}");
    }

    #[test]
    fn a_foreign_iteration_count_is_rejected() {
        let stored = format_scram_verifier(&verifier()).replace("$4096:", "$8192:");

        let error = parse_scram_verifier(&stored).unwrap_err();

        assert!(error.contains("differs from the 4096"), "{error}");
    }

    #[test]
    fn a_truncated_salted_password_is_rejected() {
        let stored = format!(
            "{SCRAM_VERIFIER_TAG}${SCRAM_ITERATIONS}:{}${}",
            BASE64.encode([7; 16]),
            BASE64.encode([9; 16])
        );

        let error = parse_scram_verifier(&stored).unwrap_err();

        assert!(error.contains("16 bytes, expected 32"), "{error}");
    }

    #[test]
    fn a_cleartext_password_is_rejected() {
        let error = parse_scram_verifier("hunter2").unwrap_err();

        assert!(
            error.contains("expected a `pivot-scram-sha-256$"),
            "{error}"
        );
    }
}
