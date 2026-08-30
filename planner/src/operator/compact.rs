//! [`Compact`] — the `COMPACT <table> [FINAL]` statement.

use std::fmt;

/// `COMPACT <table> [FINAL]`: merge the table's small files into target-sized
/// ones.
///
/// A statement, not a query: it produces no rows and isn't compiled into a
/// dataflow. The server inspects it after planning (see
/// [`Plan::as_compact`](crate::Plan::as_compact)) and runs the round on the
/// coordinator, because a round drives dataflows of its own and would deadlock
/// the pool if it ran on a worker.
#[derive(Debug, PartialEq, Eq)]
pub struct Compact {
    /// The datastore the statement named, or `None` for the default.
    pub datastore: Option<String>,
    /// The schema the statement named, or `None` for the default.
    pub schema: Option<String>,
    pub table: String,
    /// `COMPACT ... FINAL`: also merge candidates that do not meet the normal
    /// size, file-count, balance, or overlap guards.
    pub final_sweep: bool,
}

impl fmt::Display for Compact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Compact(")?;
        if let Some(datastore) = &self.datastore {
            write!(f, "{datastore}.")?;
        }
        if let Some(schema) = &self.schema {
            write!(f, "{schema}.")?;
        }
        write!(f, "{}", self.table)?;
        if self.final_sweep {
            write!(f, " FINAL")?;
        }
        write!(f, ")")
    }
}
