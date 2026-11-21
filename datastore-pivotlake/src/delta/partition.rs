//! A Delta table's partition, named the way the table's log names it: by
//! the string the log writes for each partition column's value, which
//! nothing outside this module reads.
//!
//! A Delta log records a file's partition as a map from each partition
//! column's name to its value as a string, or to a null, with no spec to
//! number partitions by. A partition is named by those strings in the
//! table's partition column order: a NULL is told apart from every string,
//! the empty one included, and a column the tuple lacks is told apart from
//! a NULL and from any value. Partitions order by their strings, an order
//! that is fixed and says nothing of how their values order.
//!
//! A file's tuple reaches here from the writer that committed the file or
//! from a read of the log, which need not hold a value as the same Arrow
//! type, and another writer's log need not spell a value as this one
//! does. Each string is therefore written here from the typed value, cast
//! to its column's own type, by the function that writes the log's own
//! strings, so a partition is named alike wherever its files' tuples came
//! from.
//!
//! A file recorded without a tuple and one whose tuple lacks every column
//! name one partition: the log records both as an empty map, so a file
//! read back from it cannot tell them apart.

use std::any::Any;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use arrow_schema::DataType;
use planner::catalog::Column;

use super::log::format_partition_string;
use crate::file_entry::PartitionValues;
use crate::table_format::{TablePartition, downcast_partition};
use crate::{Error, Result};

/// One partition of a Delta table: equal to another, ordered against it
/// and hashed by what its tuple holds for each partition column.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DeltaPartition {
    /// What the tuple holds for each partition column, in the table's
    /// partition column order.
    values: Vec<PartitionColumnValue>,
}

/// What a file's partition tuple holds for one partition column.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PartitionColumnValue {
    /// The tuple does not name the column.
    Absent,
    /// The tuple holds a NULL for the column.
    Null,
    /// The tuple holds a value, kept as the string the log writes for it.
    Value(String),
}

impl TablePartition for DeltaPartition {
    fn compare(&self, other: &dyn TablePartition) -> Ordering {
        self.cmp(downcast_partition::<Self>(other))
    }

    fn hash_into(&self, mut hasher: &mut dyn Hasher) {
        self.hash(&mut hasher);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One partition column: its name and the Arrow type its values are
/// written as.
struct PartitionColumn {
    name: String,
    data_type: DataType,
}

/// Names the partitions of one table's files.
pub(crate) struct PartitionNamer {
    partition_columns: Vec<PartitionColumn>,
}

impl PartitionNamer {
    /// The namer for a table of `columns` partitioned by `partition_by`.
    pub(crate) fn new(columns: &[Column], partition_by: &[String]) -> Result<Self> {
        let partition_columns = partition_by
            .iter()
            .map(|name| {
                let column = columns
                    .iter()
                    .find(|column| &column.name == name)
                    .ok_or_else(|| Error::UnknownPartitionColumn {
                        column: name.clone(),
                    })?;
                Ok(PartitionColumn {
                    name: name.clone(),
                    data_type: planner::types::physical_arrow_type(&column.col_type),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { partition_columns })
    }

    /// The partition a file recorded with the tuple `partition` sits in.
    pub(crate) fn name_partition(
        &self,
        partition: Option<&PartitionValues>,
    ) -> Result<DeltaPartition> {
        let values = self
            .partition_columns
            .iter()
            .map(|column| {
                let Some(value) = partition.and_then(|partition| partition.get(&column.name))
                else {
                    return Ok(PartitionColumnValue::Absent);
                };
                let (array, _) = arrow_array::Datum::get(value);
                let array = if array.data_type() == &column.data_type {
                    array.slice(0, 1)
                } else {
                    arrow_cast::cast(&array.slice(0, 1), &column.data_type)?
                };
                if array.is_null(0) {
                    return Ok(PartitionColumnValue::Null);
                }
                let string = format_partition_string(&column.name, array.as_ref())?;
                Ok(PartitionColumnValue::Value(string))
            })
            .collect::<Result<_>>()?;
        Ok(DeltaPartition { values })
    }
}
