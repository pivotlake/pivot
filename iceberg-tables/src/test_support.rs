//! What tests take of a table that nothing else does: its files listed
//! whole, a metadata store that keeps its versions in memory, and an object
//! store that fails on request. The tests of the crates built on this one
//! reach it through the `test-support` feature.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::spec::{DataFile, Datum, Manifest};
use object_storage::FileRef;
use parquet_engine::FileStats;

use crate::Result;
use crate::files::data_files::{DataFileEntry, literal_to_scalar, read_partition_values};
use crate::files::partition::Partition;
use crate::files::sort_bounded_file::holds_primitive_field;
use crate::table::{MANIFESTS_PER_READ, Table};

mod faulty_store;
mod memory_metadata_store;

pub use faulty_store::{Faults, FaultyStore, HeldRead, Requests};
pub use memory_metadata_store::MemoryMetadataStore;

impl Table {
    /// Every partition this version has a file in, in key order. A
    /// manifest whose manifest-list entry names its one partition is not
    /// read; the partitions any other holds are read off the part of it
    /// that lists them.
    pub fn partitions(&self) -> Result<Vec<Arc<Partition>>> {
        self.list_partitions()
    }

    /// Every live data file of this version as a commit records it, with
    /// the statistics of every declared column, by partition in key order,
    /// and a partition's manifest by manifest, as the manifests list them. Every data manifest is read
    /// and parsed, whatever the cache holds of it.
    pub fn entries(&self) -> Result<Vec<DataFileEntry>> {
        let listed = self.manifest_list().listed_data_manifests();
        let mut recorded: Vec<(Arc<[u8]>, DataFileEntry)> = Vec::new();
        for manifests in listed.chunks(MANIFESTS_PER_READ) {
            let parsed = self.read_manifests(manifests.iter().map(|manifest| manifest.file()))?;
            for manifest in &parsed {
                self.read_live_entries(manifest, &mut recorded)?;
            }
        }
        recorded.sort_by(|(left, _), (right, _)| left.cmp(right));
        Ok(recorded.into_iter().map(|(_, entry)| entry).collect())
    }

    /// Add to `recorded`, beside the key of its partition, each live entry
    /// of `manifest`, a parsed data manifest of the table.
    fn read_live_entries(
        &self,
        manifest: &Manifest,
        recorded: &mut Vec<(Arc<[u8]>, DataFileEntry)>,
    ) -> Result<()> {
        let schema_id = manifest.metadata().schema_id();
        let spec_id = manifest.metadata().partition_spec().spec_id();
        for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
            let data = entry.data_file();
            let path = self.location().file_path_for_uri(data.file_path())?;
            let partition = Partition::new(spec_id, data.partition().clone());
            let entry = DataFileEntry {
                file: FileRef {
                    path: path.clone(),
                    size: data.file_size_in_bytes(),
                },
                partition: read_partition_values(
                    self.metadata(),
                    self.columns(),
                    &partition,
                    &path,
                )?,
                stats: Some(Arc::new(self.read_file_stats(data, schema_id))),
                sort_order_id: data.sort_order_id(),
            };
            recorded.push((Arc::clone(partition.key()), entry));
        }
        Ok(())
    }

    /// The statistics `data`, an entry of a manifest written under the
    /// schema `schema_id`, records for the declared columns, as the
    /// datastore keys them: by column name, as typed scalars of the
    /// column's Pivot type.
    fn read_file_stats(&self, data: &DataFile, schema_id: i32) -> FileStats {
        let schema = self.metadata().current_schema();
        let stored_schema = self.metadata().schema_by_id(schema_id);
        let mut stats = FileStats {
            num_records: i64::try_from(data.record_count()).ok(),
            min_values: Default::default(),
            max_values: Default::default(),
            null_counts: Default::default(),
        };
        for column in self.columns() {
            let Some(field_id) = schema.field_id_by_name(&column.name) else {
                continue;
            };
            // A schema that does not have the field as a primitive records
            // no bound of it, nor does one the table no longer lists.
            let is_bounded =
                stored_schema.is_some_and(|stored| holds_primitive_field(stored, field_id));
            let read = |bounds: &HashMap<i32, Datum>| {
                let bound = bounds.get(&field_id).filter(|_| is_bounded)?;
                let scalar = literal_to_scalar(bound.literal(), &column.col_type)?;
                Some(scalar.into_inner())
            };
            if let (Some(min), Some(max)) = (read(data.lower_bounds()), read(data.upper_bounds())) {
                stats.min_values.insert(column.name.clone(), min);
                stats.max_values.insert(column.name.clone(), max);
            }
            let null_count = data.null_value_counts().get(&field_id);
            if let Some(nulls) = null_count.and_then(|nulls| i64::try_from(*nulls).ok()) {
                stats.null_counts.insert(column.name.clone(), nulls);
            }
        }
        stats
    }
}
