//! [`SetVariable`] — `SET`/`RESET` of a session variable.

use duckdb_planner::operator as duckdb_operator;
use std::fmt;

/// `SET <name> = <value>` / `RESET <name>` (the latter arrives with no value).
///
/// A session knob, not a query: it produces no rows and isn't compiled into a
/// dataflow. The server inspects it after planning (see
/// [`Plan::as_set_variable`](crate::Plan::as_set_variable)) and acts on the names
/// it recognises. `value` is DuckDB's serialized constant (a boolean reads back
/// as `"true"`/`"false"`); `None` is a `RESET`.
#[derive(Debug)]
pub struct SetVariable {
    pub name: String,
    pub value: Option<String>,
}

impl TryFrom<duckdb_operator::SetVariable> for SetVariable {
    type Error = super::Error;

    fn try_from(set: duckdb_operator::SetVariable) -> Result<Self, Self::Error> {
        Ok(SetVariable {
            name: set.name,
            value: set.value,
        })
    }
}

impl fmt::Display for SetVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            Some(v) => write!(f, "Set({} = {v})", self.name),
            None => write!(f, "Reset({})", self.name),
        }
    }
}
