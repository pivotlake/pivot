use arrow_arith::boolean::{and, or};
use arrow_array::{ArrayRef, BooleanArray, Scalar};
use arrow_schema::ArrowError;

use crate::ColumnBounds;

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

/// A logical column or VARIANT path compared with a typed constant. The planner
/// retains the original SQL filter: metadata pruning only proves exclusions.
#[derive(Clone, Debug)]
pub struct ColumnPredicate {
    /// Index in the full bound-table schema, before scan projection. For a
    /// VARIANT path this identifies the VARIANT column, not a physical leaf.
    pub column_idx: usize,
    /// Field names inside a VARIANT column. An empty path with `as_type` set
    /// refers to its root value; without `as_type` it is a plain column.
    pub path: Vec<String>,
    /// The SQL cast's physical output type for a variant path. Pruning against
    /// the raw typed leaf is sound only when this is a semantic identity.
    pub as_type: Option<arrow_schema::DataType>,
    pub compare_type: Comparison,
    pub value: Scalar<ArrayRef>,
}

/// A logical source in the full table schema, optionally a typed VARIANT path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnReference {
    pub column_idx: usize,
    pub path: Vec<String>,
    /// A VARIANT path's exact cast type. `None` denotes an ordinary column.
    pub as_type: Option<arrow_schema::DataType>,
}

impl From<usize> for ColumnReference {
    fn from(column_idx: usize) -> Self {
        Self {
            column_idx,
            path: Vec::new(),
            as_type: None,
        }
    }
}

impl ColumnReference {
    pub(crate) fn matches(&self, predicate: &ColumnPredicate) -> bool {
        self.column_idx == predicate.column_idx
            && self.path == predicate.path
            && self.as_type == predicate.as_type
    }
}

/// A projected predicate over one set of bounds, with no physical column index.
/// NOT must be pushed into comparisons before construction: negating a
/// "may match" result does not prove that the opposite predicate cannot match.
pub enum PruningPredicate {
    Always(bool),
    Compare {
        compare: Comparison,
        value: Scalar<ArrayRef>,
    },
    And(Vec<Self>),
    Or(Vec<Self>),
}

/// The source and transform whose values a partition's bounds describe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionExpression {
    pub source: ColumnReference,
    /// The type used to write these partitions, before schema evolution.
    pub source_type: arrow_schema::DataType,
    pub transform: crate::PartitionTransform,
}

impl PartitionExpression {
    /// Projection is inclusive: matching rows must always survive. Unsupported
    /// casts or transforms supply no proof that an object can be excluded.
    pub(crate) fn project(&self, predicate: &ColumnPredicate) -> PruningPredicate {
        self.transform
            .project(&self.source_type, predicate.compare_type, &predicate.value)
    }
}

impl PruningPredicate {
    /// Evaluate against one expression's bounds. Even a constant projection
    /// applies only where those bounds' coverage is valid (e.g. an old spec).
    pub fn may_match(&self, bounds: &ColumnBounds, len: usize) -> Result<BooleanArray, ArrowError> {
        bounds.validate(len)?;
        let result = self.evaluate(bounds, len)?;
        Ok(match &bounds.validity {
            Some(validity) => BooleanArray::new(result.values() | &!validity.inner(), None),
            None => result,
        })
    }

    fn evaluate(&self, bounds: &ColumnBounds, len: usize) -> Result<BooleanArray, ArrowError> {
        match self {
            Self::Always(value) => Ok(BooleanArray::from(vec![*value; len])),
            Self::Compare { compare, value } => bounds.compare(len, *compare, value),
            Self::And(predicates) | Self::Or(predicates) => {
                let conjunction = matches!(self, Self::And(_));
                let mut result = BooleanArray::from(vec![conjunction; len]);
                for child in predicates {
                    let next = child.evaluate(bounds, len)?;
                    result = if conjunction {
                        and(&result, &next)?
                    } else {
                        or(&result, &next)?
                    };
                }
                Ok(result)
            }
        }
    }
}
