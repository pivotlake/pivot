//! One live data file of a Delta table as its table holds it: the facts
//! every format knows of a file, beside the entry the log recorded for it
//! and the row groups of its footer.

use std::sync::Arc;

use arrow_array::ArrayRef;
use parquet_engine::RowGroupMetadata;
use planner::catalog::Column;
use table_file_index::{FileFacts, FileIndex, IndexedRecord, SortKeyEncoder};

use super::partition_key::PartitionKeyEncoder;
use crate::Result;
use crate::file_entry::{FileEntry, FileStats};
use crate::table::TableFile;

/// One live data file of a Delta table version.
pub(crate) struct DeltaFile {
    facts: FileFacts,
    /// The file as the log recorded it: its partition tuple and its
    /// statistics.
    pub entry: FileEntry,
    /// The file's row groups, in the file's own order.
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl DeltaFile {
    /// The rows the file's row groups hold.
    pub(crate) fn count_rows(&self) -> u64 {
        self.row_groups
            .iter()
            .map(|row_group| row_group.num_rows.max(0) as u64)
            .sum()
    }
}

impl IndexedRecord for DeltaFile {
    fn facts(&self) -> &FileFacts {
        &self.facts
    }

    /// Zero: nothing budgets a Delta record, so none is measured.
    fn estimate_heap_bytes(&self) -> u64 {
        0
    }
}

/// Files of a Delta table version, in the order a partition's files stand
/// in.
pub(crate) type DeltaIndex = FileIndex<DeltaFile>;

/// Builds the records of one table's files: names each file's partition and
/// places it on the table's sort key.
pub(crate) struct RecordBuilder {
    /// The sort columns the table declares, in sort order.
    sort_columns: Vec<String>,
    sort_keys: SortKeyEncoder,
    partition_keys: PartitionKeyEncoder,
}

impl RecordBuilder {
    /// The builder for a table of `columns`, partitioned by `partition_by`
    /// and sorted by `sort_by`.
    pub(crate) fn new(
        columns: &[Column],
        partition_by: &[String],
        sort_by: &[String],
    ) -> Result<Self> {
        let (sort_columns, sort_types) = sort_by
            .iter()
            .filter_map(|name| {
                let column = columns.iter().find(|column| &column.name == name)?;
                let arrow_type = planner::types::physical_arrow_type(&column.col_type);
                Some((name.clone(), arrow_type))
            })
            .unzip();
        Ok(Self {
            sort_columns,
            sort_keys: SortKeyEncoder::new(sort_types)?,
            partition_keys: PartitionKeyEncoder::new(columns, partition_by)?,
        })
    }

    /// The record of `file`, which the commit standing at `commit_sequence`
    /// among the table's added, beside the bytes its partition is named by.
    pub(crate) fn build(
        &self,
        file: TableFile,
        commit_sequence: u64,
    ) -> Result<(Vec<u8>, DeltaFile)> {
        let TableFile { entry, row_groups } = file;
        let partition = self.partition_keys.encode(entry.partition.as_ref())?;
        let stats = entry.stats.as_deref();
        let find_bound =
            |bounds: fn(&FileStats) -> &std::collections::HashMap<String, ArrayRef>,
             position: usize| {
                stats.and_then(|stats| bounds(stats).get(&self.sort_columns[position]).cloned())
            };
        let sort_range = self.sort_keys.encode_range(
            |position| find_bound(|stats| &stats.min_values, position),
            |position| find_bound(|stats| &stats.max_values, position),
        )?;
        let rows = stats
            .and_then(|stats| stats.num_records)
            .and_then(|rows| u64::try_from(rows).ok());
        let facts = FileFacts::new(
            entry.file.clone(),
            rows,
            sort_range.minimum,
            sort_range.maximum,
            commit_sequence,
        );
        let record = DeltaFile {
            facts,
            entry,
            row_groups,
        };
        Ok((partition, record))
    }
}
