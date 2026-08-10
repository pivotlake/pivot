//! [`Parameter`], a prepared-statement parameter placeholder (`$n`).

use crate::types::Type;
use std::fmt;

/// A prepared-statement parameter placeholder (`$n`), carrying the type the
/// binder inferred for it. It holds no value: each execution binds one, and
/// [`Expression::resolve_parameters`](crate::expression::Expression::resolve_parameters)
/// replaces the placeholder with that value as a constant before the
/// expression compiles.
#[derive(Debug, Clone)]
pub struct Parameter {
    /// The parameter's 1-based position: `$3` has index 3.
    pub index: usize,
    pub return_type: Type,
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.index)
    }
}
