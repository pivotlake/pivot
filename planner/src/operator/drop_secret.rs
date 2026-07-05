//! [`DropSecret`] - `DROP SECRET name`.
//!
//! Like `SET`, a secret statement is a catalog command, not a query: the
//! server intercepts it after planning (see
//! [`Plan::as_secret_command`](crate::Plan::as_secret_command)) and applies it
//! to the catalog directly, so it never compiles into a dataflow.

use crate::catalog::DropSecretRequest;
use std::fmt;

/// DROP SECRET by name.
#[derive(Debug)]
pub struct DropSecret {
    pub request: DropSecretRequest,
}

impl fmt::Display for DropSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DropSecret({})", self.request.name)
    }
}
