//! [`CreateTextIndex`] — `SELECT create_text_index('events', 'message')`.

use std::fmt::{self, Display};

/// `create_text_index(table, column)`: build the text index of `column` for
/// every data file of `table` that has none, so existing rows are covered and
/// a later call covers files written since.
///
/// Written as a function call but run as a statement: it produces no rows and
/// isn't compiled into a dataflow. The server inspects the plan (see
/// [`Plan::as_create_text_index`](crate::Plan::as_create_text_index)) and runs
/// the build on the coordinator, because the build drives dataflows of its own
/// and would deadlock the pool if it ran on a worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTextIndex {
    /// The schema the table name was qualified with, or `None` for the default.
    pub schema: Option<String>,
    pub table: String,
    pub column: String,
}

impl CreateTextIndex {
    /// `table` is the table's name, optionally qualified as `schema.table`.
    pub fn new(table: &str, column: String) -> Self {
        let (schema, table) = match table.rsplit_once('.') {
            Some((schema, table)) => (Some(schema.to_string()), table.to_string()),
            None => (None, table.to_string()),
        };
        Self {
            schema,
            table,
            column,
        }
    }
}

impl Display for CreateTextIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "create_text_index('")?;
        if let Some(schema) = &self.schema {
            write!(f, "{schema}.")?;
        }
        write!(f, "{}', '{}')", self.table, self.column)
    }
}
