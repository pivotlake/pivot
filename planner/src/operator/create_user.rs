//! [`CreateUser`] — the `CREATE USER <name> [PASSWORD '<password>']` statement.

use crate::catalog::{CatalogTransaction, CreateUserRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// `CREATE USER <name> [PASSWORD '<password>']`: add a login user.
///
/// A statement, not a query: it produces no rows. Like a `CREATE SCHEMA`, it
/// compiles through the transaction into a dataflow that only stages the
/// creation; the user is durably added when the transaction commits.
#[derive(Debug, PartialEq, Eq)]
pub struct CreateUser {
    pub name: String,
    /// The password, or `None` when the PASSWORD clause was omitted (the user
    /// authenticates by trust).
    pub password: Option<String>,
}

impl CreateUser {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let creation = transaction
            .bind_create_user(CreateUserRequest {
                name: self.name.clone(),
                password: self.password.clone(),
            })
            .map_err(Error::CreateUser)?;
        creation.compile(dispatcher).map_err(Error::CreateUser)
    }
}

impl fmt::Display for CreateUser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CreateUser({}", self.name)?;
        if self.password.is_some() {
            write!(f, " PASSWORD [redacted]")?;
        }
        write!(f, ")")
    }
}
