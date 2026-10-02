use std::collections::BTreeMap;

use arrow_arith::boolean::and;
use arrow_array::BooleanArray;
use arrow_schema::{ArrowError, DataType};
use arrow_select::filter::FilterBuilder;

use crate::{ColumnBounds, ColumnPredicate, PartitionExpression, PruningPredicate};

/// Statistics for one logical table column. VARIANT paths stay under their
/// parent column and use typed Arrow arrays, without encoding VARIANT objects.
#[derive(Clone)]
pub enum ColumnStatistics {
    Primitive(Box<ColumnBounds>),
    /// An empty path describes a typed VARIANT root. Missing paths are unknown.
    Variant(BTreeMap<Vec<String>, VariantPathStatistics>),
}

/// Bounds usable for a VARIANT path's exact cast type. The bounds' validity
/// proves coverage of unshredded fallbacks separately for each object and path.
#[derive(Clone)]
pub struct VariantPathStatistics {
    pub data_type: DataType,
    pub bounds: ColumnBounds,
}

impl From<ColumnBounds> for ColumnStatistics {
    fn from(bounds: ColumnBounds) -> Self {
        Self::Primitive(Box::new(bounds))
    }
}

impl ColumnStatistics {
    pub fn primitive(&self) -> Option<&ColumnBounds> {
        match self {
            Self::Primitive(bounds) => Some(bounds),
            Self::Variant(_) => None,
        }
    }

    fn resolve(&self, predicate: &ColumnPredicate) -> Option<&ColumnBounds> {
        match self {
            Self::Primitive(bounds) if predicate.path.is_empty() && predicate.as_type.is_none() => {
                Some(bounds)
            }
            Self::Variant(paths) => {
                let path = paths.get(&predicate.path)?;
                (predicate.as_type.as_ref() == Some(&path.data_type)).then_some(&path.bounds)
            }
            _ => None,
        }
    }

    fn validate(&self, len: usize) -> Result<(), ArrowError> {
        match self {
            Self::Primitive(bounds) => bounds.validate(len),
            Self::Variant(paths) => {
                for path in paths.values() {
                    path.bounds.validate(len)?;
                    if [&path.bounds.lower, &path.bounds.upper]
                        .into_iter()
                        .flatten()
                        .any(|array| array.data_type() != &path.data_type)
                    {
                        return Err(ArrowError::InvalidArgumentError(
                            "VARIANT bounds must have their path's declared type".into(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    fn map_bounds(
        &self,
        map: &mut impl FnMut(&ColumnBounds) -> Result<ColumnBounds, ArrowError>,
    ) -> Result<Self, ArrowError> {
        Ok(match self {
            Self::Primitive(bounds) => Self::from(map(bounds)?),
            Self::Variant(paths) => Self::Variant(
                paths
                    .iter()
                    .map(|(path, statistics)| {
                        Ok((
                            path.clone(),
                            VariantPathStatistics {
                                data_type: statistics.data_type.clone(),
                                bounds: map(&statistics.bounds)?,
                            },
                        ))
                    })
                    .collect::<Result<_, ArrowError>>()?,
            ),
        })
    }
}

/// One partition expression's bounds across the batch. Validity marks the rows
/// written with this expression, so historical specs can share the same batch.
#[derive(Clone)]
pub struct PartitionStatistics {
    pub expression: PartitionExpression,
    pub bounds: ColumnBounds,
}

/// Arrow bounds in object order: one row per manifest, file or row
/// group. Column statistics are keyed by the full table's logical column index;
/// partition statistics carry their source and transform separately.
/// Cloning shares the Arrow buffers. Storage adapters retain object descriptors
/// and physical Parquet leaf order.
#[derive(Clone)]
pub struct StatisticsBatch {
    len: usize,
    column_stats: BTreeMap<usize, ColumnStatistics>,
    partition_stats: Vec<PartitionStatistics>,
}

impl StatisticsBatch {
    pub fn new(
        len: usize,
        column_stats: BTreeMap<usize, ColumnStatistics>,
        partition_stats: Vec<PartitionStatistics>,
    ) -> Result<Self, ArrowError> {
        for column in column_stats.values() {
            column.validate(len)?;
        }
        for partition in &partition_stats {
            partition.bounds.validate(len)?;
        }
        Ok(Self {
            len,
            column_stats,
            partition_stats,
        })
    }

    pub fn column_stats(&self) -> &BTreeMap<usize, ColumnStatistics> {
        &self.column_stats
    }

    pub fn partition_stats(&self) -> &[PartitionStatistics] {
        &self.partition_stats
    }

    /// Select statistics rows in their original order. Build the Arrow filter
    /// once and reuse it for all columns, VARIANT paths and partition bounds.
    pub fn filter(&self, keep: &BooleanArray) -> Result<Self, ArrowError> {
        if keep.len() != self.len {
            return Err(ArrowError::InvalidArgumentError(
                "selection must have one entry per object".into(),
            ));
        }
        let filter = FilterBuilder::new(keep).optimize().build();
        self.map_bounds(filter.count(), |bounds| bounds.filter(&filter))
    }

    /// Slice all bounds through shared Arrow buffers, preserving their grouping.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset <= self.len && len <= self.len - offset);
        self.map_bounds(len, |bounds| Ok(bounds.slice(offset, len)))
            .expect("slicing bounds is infallible")
    }

    fn map_bounds(
        &self,
        len: usize,
        mut map: impl FnMut(&ColumnBounds) -> Result<ColumnBounds, ArrowError>,
    ) -> Result<Self, ArrowError> {
        let column_stats = self
            .column_stats
            .iter()
            .map(|(&column, statistics)| Ok((column, statistics.map_bounds(&mut map)?)))
            .collect::<Result<_, ArrowError>>()?;
        let partition_stats = self
            .partition_stats
            .iter()
            .map(|partition| {
                Ok(PartitionStatistics {
                    expression: partition.expression.clone(),
                    bounds: map(&partition.bounds)?,
                })
            })
            .collect::<Result<_, ArrowError>>()?;
        Ok(Self {
            len,
            column_stats,
            partition_stats,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Add an adapter's proof that the listed primitive columns contain no NaNs
    /// in any object. VARIANT columns require their own per-path proofs.
    pub fn mark_columns_nan_free(&mut self, columns: &[usize]) {
        for column in columns {
            if let Some(ColumnStatistics::Primitive(bounds)) = self.column_stats.get_mut(column) {
                bounds.nan_free = Some(BooleanArray::from(vec![true; self.len]));
            }
        }
    }

    /// Match logical columns and VARIANT paths, project partition predicates,
    /// and return a non-null mask: false proves an object cannot match.
    pub fn prune(&self, predicates: &[ColumnPredicate]) -> Result<BooleanArray, ArrowError> {
        let mut masks = self
            .project(predicates)
            .map(|(bounds, predicate)| predicate.may_match(bounds, self.len));
        let Some(first) = masks.next() else {
            return Ok(BooleanArray::from(vec![true; self.len]));
        };
        masks.try_fold(first?, |keep, next| and(&keep, &next?))
    }

    /// Evaluate a live predicate for one object, slicing only matching bounds.
    pub fn prune_row(
        &self,
        row: usize,
        predicates: &[ColumnPredicate],
    ) -> Result<bool, ArrowError> {
        if row >= self.len {
            return Err(ArrowError::InvalidArgumentError(
                "statistics row is out of bounds".into(),
            ));
        }
        for (bounds, predicate) in self.project(predicates) {
            if !predicate.may_match(&bounds.slice(row, 1), 1)?.value(0) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn project<'a>(
        &'a self,
        predicates: &'a [ColumnPredicate],
    ) -> impl Iterator<Item = (&'a ColumnBounds, PruningPredicate)> + 'a {
        predicates.iter().flat_map(move |predicate| {
            let column = self
                .column_stats
                .get(&predicate.column_idx)
                .and_then(|column| column.resolve(predicate))
                .map(|bounds| {
                    (
                        bounds,
                        PruningPredicate::Compare {
                            compare: predicate.compare_type,
                            value: predicate.value.clone(),
                        },
                    )
                });
            let partitions = self
                .partition_stats
                .iter()
                .filter(|partition| partition.expression.source.matches(predicate))
                .map(|partition| (&partition.bounds, partition.expression.project(predicate)));
            column
                .into_iter()
                .chain(partitions)
                .filter(|(_, predicate)| !matches!(predicate, PruningPredicate::Always(true)))
        })
    }
}
