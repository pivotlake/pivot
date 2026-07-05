//! [`CreateSecret`] - `CREATE SECRET name (TYPE s3, ...)`.
//!
//! Like `SET`, a secret statement is a catalog command, not a query: the
//! server intercepts it after planning (see
//! [`Plan::as_secret_command`](crate::Plan::as_secret_command)) and applies it
//! to the catalog directly, so it never compiles into a dataflow.

use crate::catalog::{CreateSecretRequest, secret_option_is_redacted};
use std::fmt;

/// CREATE SECRET with its bound options.
#[derive(Debug)]
pub struct CreateSecret {
    pub request: CreateSecretRequest,
}

impl fmt::Display for CreateSecret {
    /// Renders with sensitive option values replaced by `redacted` - plan
    /// trees end up in EXPLAIN output and logs, which must never carry a
    /// credential in clear text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let options = self
            .request
            .options
            .iter()
            .map(|(key, value)| {
                if secret_option_is_redacted(key) {
                    format!("{key}: redacted")
                } else {
                    format!("{key}: {value:?}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "CreateSecret({}, type: {}, scope: {:?}, options: {{{options}}})",
            self.request.name, self.request.secret_type, self.request.scope
        )
    }
}
