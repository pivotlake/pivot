//! [`DropUser`]: the `DROP USER <name>` statement.

use crate::catalog::{CatalogTransaction, DropUserRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// `DROP USER <name>`: remove a login user.
///
/// A statement, not a query: it produces no rows. Like a `CREATE USER`, it
/// compiles through the transaction into a dataflow that only stages the
/// drop; the user is durably removed when the transaction commits.
#[derive(Debug, PartialEq, Eq)]
pub struct DropUser {
    pub name: String,
}

impl DropUser {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let drop = transaction
            .bind_drop_user(DropUserRequest {
                name: self.name.clone(),
            })
            .map_err(Error::DropUser)?;
        drop.compile(dispatcher).map_err(Error::DropUser)
    }
}

impl fmt::Display for DropUser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DropUser({})", self.name)
    }
}
