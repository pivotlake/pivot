//! Prepared-statement parameter holes.

use std::fmt;

use crate::types::Type;

/// A zero-based pointer into the values supplied by a protocol Bind message.
///
/// The type is fixed by DuckDB while the statement is parsed and optimized;
/// the value itself is deliberately absent from the cached plan.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub index: usize,
    pub return_type: Type,
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.index + 1)
    }
}
