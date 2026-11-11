//! A Delta file's partition as bytes equal for exactly the files of one
//! partition: what its table names the partition by.
//!
//! A Delta log records a file's partition as a map of typed values, with no
//! spec to number partitions by, so the bytes are derived from the values:
//! each partition column's value, a NULL included, and a column the tuple
//! lacks told apart from any value. The bytes of two partitions say nothing
//! of how their values order.
//!
//! A file's tuple reaches here from the writer that committed the file or
//! from a read of the log, which need not hold a value as the same Arrow
//! type. Each value is encoded as its column's own type, so a partition is
//! named alike wherever its files' tuples came from.
//!
//! A file recorded without a tuple and one whose tuple lacks every column
//! name one partition: the log records both as an empty map, so a file
//! read back from it cannot tell them apart.

use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;
use planner::catalog::Column;

use crate::file_entry::PartitionValues;
use crate::{Error, Result};

/// One partition column: its name, the Arrow type its values are encoded
/// as, and the converter over that type.
struct PartitionColumn {
    name: String,
    data_type: DataType,
    converter: RowConverter,
}

/// Encodes the partition tuples of one table's files.
pub(crate) struct PartitionKeyEncoder {
    partition_columns: Vec<PartitionColumn>,
}

impl PartitionKeyEncoder {
    /// The encoder for a table of `columns` partitioned by `partition_by`.
    pub(crate) fn new(columns: &[Column], partition_by: &[String]) -> Result<Self> {
        let partition_columns = partition_by
            .iter()
            .map(|name| {
                let column = columns
                    .iter()
                    .find(|column| &column.name == name)
                    .ok_or_else(|| Error::UnknownSpecColumn {
                        option: "partition_by".to_string(),
                        column: name.clone(),
                    })?;
                let data_type = planner::types::physical_arrow_type(&column.col_type);
                let converter = RowConverter::new(vec![SortField::new(data_type.clone())])?;
                Ok(PartitionColumn {
                    name: name.clone(),
                    data_type,
                    converter,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { partition_columns })
    }

    /// The bytes of the partition a file recorded with the tuple
    /// `partition` sits in.
    pub(crate) fn encode(&self, partition: Option<&PartitionValues>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for column in &self.partition_columns {
            let Some(value) = partition.and_then(|partition| partition.get(&column.name)) else {
                bytes.push(0);
                continue;
            };
            let (array, _) = arrow_array::Datum::get(value);
            let array = if array.data_type() == &column.data_type {
                array.slice(0, 1)
            } else {
                arrow_cast::cast(&array.slice(0, 1), &column.data_type)?
            };
            let rows = column.converter.convert_columns(&[array])?;
            let row = rows.row(0);
            bytes.push(1);
            bytes.extend_from_slice(&(row.as_ref().len() as u32).to_le_bytes());
            bytes.extend_from_slice(row.as_ref());
        }
        Ok(bytes)
    }
}
