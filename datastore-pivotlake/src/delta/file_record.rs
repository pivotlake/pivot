//! One live data file of a Delta table as its partition's index holds it:
//! the facts every format knows of a file, beside the partition it sits in
//! and the entry the log recorded for it.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use parquet_engine::{measure_allocation, measure_shared_allocation};
use planner::catalog::Column;
use table_file_index::{FileFacts, FileIndex, IndexedRecord, SortKeyEncoder};

use super::partition_key::PartitionKeyEncoder;
use crate::Result;
use crate::file_entry::{FileEntry, FileStats};

/// One live data file of a Delta table version.
pub(crate) struct DeltaFile {
    facts: FileFacts,
    /// The partition the file sits in, as its index names it.
    partition: Arc<[u8]>,
    /// The file as the log recorded it: its partition tuple and its
    /// statistics.
    pub entry: FileEntry,
    /// An estimate of the memory the record holds, itself included.
    heap_bytes: u64,
}

impl IndexedRecord for DeltaFile {
    fn facts(&self) -> &FileFacts {
        &self.facts
    }

    fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

impl DeltaFile {
    /// The partition the file sits in, as its index names it. A file is
    /// grouped by it before any index of its partition exists.
    pub(crate) fn partition(&self) -> &Arc<[u8]> {
        &self.partition
    }
}

/// The live data files of one partition of a Delta table version.
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

    /// The bytes the index names the partition of a file recorded with the
    /// tuple of `entry` by.
    pub(crate) fn encode_partition(&mut self, entry: &FileEntry) -> Result<Arc<[u8]>> {
        self.partition_keys.encode(entry.partition.as_ref())
    }

    /// The record of `entry`, a file the commit standing at
    /// `commit_sequence` among the table's added.
    pub(crate) fn build(&mut self, entry: FileEntry, commit_sequence: u64) -> Result<DeltaFile> {
        let partition = self.encode_partition(&entry)?;
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
        let heap_bytes = estimate_record_heap_bytes(&facts, &entry);
        Ok(DeltaFile {
            facts,
            partition,
            entry,
            heap_bytes,
        })
    }
}

/// An estimate of the memory the record of `entry` holds, itself included:
/// its facts, its path, its partition tuple and its statistics, each at the
/// size an allocation of it takes, with what the index's own maps spend on
/// one file.
fn estimate_record_heap_bytes(facts: &FileFacts, entry: &FileEntry) -> u64 {
    /// What a hash map spends on one entry beside its key and value.
    const MAP_ENTRY_BYTES: usize = 48;

    let measure_name = |name: &String| measure_allocation(name.len()) + MAP_ENTRY_BYTES;
    let partition_bytes: usize = entry.partition.as_ref().map_or(0, |partition| {
        partition
            .iter()
            .map(|(name, value)| {
                let (array, _) = arrow_array::Datum::get(value);
                measure_name(name) + array.get_array_memory_size()
            })
            .sum()
    });
    let stats_bytes: usize = entry.stats.as_ref().map_or(0, |stats| {
        let measure_bounds = |bounds: &std::collections::HashMap<String, ArrayRef>| -> usize {
            bounds
                .iter()
                .map(|(name, bound)| measure_name(name) + bound.get_array_memory_size())
                .sum()
        };
        measure_shared_allocation(size_of::<FileStats>())
            + measure_bounds(&stats.min_values)
            + measure_bounds(&stats.max_values)
            + stats.null_counts.keys().map(measure_name).sum::<usize>()
    });
    let total = measure_shared_allocation(size_of::<DeltaFile>())
        + measure_allocation(entry.file.path.as_str().len())
        + partition_bytes
        + stats_bytes;
    total as u64 + facts.estimate_heap_bytes()
}
