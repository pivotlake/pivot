//! [`TransactionStatement`]: `BEGIN`/`COMMIT`/`ROLLBACK`.

use std::fmt;

/// `BEGIN`, `COMMIT` or `ROLLBACK`.
///
/// Pivot commits every statement individually, so these carry no state to
/// apply. They are accepted so PostgreSQL drivers that wrap statements in a
/// transaction by default can work. The server inspects the plan after
/// planning (see [`Plan::as_transaction_stmt`](crate::Plan::as_transaction_stmt)) and
/// answers with the statement's tag without compiling or executing anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionStatement {
    Begin,
    Commit,
    Rollback,
}

impl fmt::Display for TransactionStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionStatement::Begin => f.write_str("Begin"),
            TransactionStatement::Commit => f.write_str("Commit"),
            TransactionStatement::Rollback => f.write_str("Rollback"),
        }
    }
}
