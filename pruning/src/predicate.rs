use arrow_array::{ArrayRef, Scalar};

/// A column of the table, or the field at `path` inside it. Only VARIANT
/// columns have fields; a plain column has an empty path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColumnPath {
    /// Index in the full table schema, before scan projection.
    pub column_idx: usize,
    pub path: Vec<String>,
}

impl From<usize> for ColumnPath {
    fn from(column_idx: usize) -> Self {
        Self {
            column_idx,
            path: Vec::new(),
        }
    }
}

/// Comparison semantics match Arrow's kernels, including total float ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

/// `column <comparison> value`. The value is one typed constant; for a VARIANT
/// field its type is the type the query casts the field to.
#[derive(Clone, Debug)]
pub struct Predicate {
    pub column: ColumnPath,
    pub comparison: Comparison,
    pub value: Scalar<ArrayRef>,
}

impl Predicate {
    /// The distinct columns `predicates` read, in first-use order: the only
    /// columns whose statistics a query needs.
    pub fn columns(predicates: &[Predicate]) -> Vec<&ColumnPath> {
        let mut columns: Vec<&ColumnPath> = Vec::new();
        for predicate in predicates {
            if !columns.contains(&&predicate.column) {
                columns.push(&predicate.column);
            }
        }
        columns
    }
}
