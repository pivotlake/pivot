//! Parts of manifests that list no file, stored in a cache and found in it
//! by where they are, the parts a cache holds of one of a table's
//! manifests, and a cache emptied of every table: what a test of the cache
//! takes of it.

use std::sync::Arc;

use arrow_array::{Int32Array, StringArray};
use arrow_buffer::BooleanBuffer;
use iceberg::spec::{Literal, Struct};
use parquet_engine::measure_shared_allocation;

use crate::files::manifest_parts::{CachedColumnBounds, CachedPartitionFiles, ManifestPartitions};
use crate::files::partition::Partition;
use crate::table::Table;
use crate::table_metadata_cache::held_entries::HeldEntries;
use crate::table_metadata_cache::{CachedPartKey, ManifestPart, PartitionPart, TableMetadataCache};
use iceberg::spec::FormatVersion;
use uuid::Uuid;

/// The partition of the one string `value`, under the spec 0: a partition
/// for a test of the cache to name parts by.
pub fn build_text_partition(value: &str) -> Arc<Partition> {
    let tuple = Struct::from_iter([Some(Literal::string(value))]);
    Arc::new(Partition::new(0, tuple))
}

/// The files of a partition a manifest lists no file of.
fn build_empty_files() -> CachedPartitionFiles {
    CachedPartitionFiles {
        schema_id: 0,
        paths: StringArray::from_iter_values(std::iter::empty::<&str>()),
        sizes: Vec::new(),
        record_counts: Vec::new(),
        commit_sequences: Vec::new(),
        sort_order_ids: Int32Array::from(Vec::<Option<i32>>::new()),
        stats_field_ids: Vec::new(),
        null_free: BooleanBuffer::new_unset(0),
        heap_bytes: measure_shared_allocation(size_of::<CachedPartitionFiles>()) as u64,
    }
}

/// What a manifest records of a column in no file.
fn build_empty_bounds() -> CachedColumnBounds {
    CachedColumnBounds {
        bounds: None,
        only_nulls: BooleanBuffer::new_unset(0),
        heap_bytes: measure_shared_allocation(size_of::<CachedColumnBounds>()) as u64,
    }
}

/// Hold in the table `table_id` a part that lists no file at `key`, of
/// a manifest read under `format_version`, as used now.
pub fn store_empty_part(
    cache: &TableMetadataCache,
    table_id: Uuid,
    format_version: FormatVersion,
    key: &CachedPartKey,
) {
    let manifest = &key.manifest;
    match &key.part {
        ManifestPart::PartitionList => {
            let partitions = Arc::new(ManifestPartitions::new(Vec::new()));
            cache.store_manifest_partitions(table_id, format_version, manifest, partitions);
        }
        ManifestPart::ForPartition { partition, part } => match part {
            PartitionPart::Files => {
                let files = Arc::new(build_empty_files());
                cache.store_manifest_files(table_id, format_version, manifest, partition, files);
            }
            PartitionPart::ColumnBounds { field_id } => {
                let bounds = vec![(*field_id, Arc::new(build_empty_bounds()))];
                cache.store_column_bounds(table_id, format_version, manifest, partition, bounds);
            }
        },
    }
}

/// Hold in the table `table_id` a part that lists no file at `key`, of
/// a manifest read under `format_version`, the files of a
/// partition or what a manifest records of a column in them, as a part
/// nobody asked for. Whether a part is held there afterwards.
pub fn store_unasked_empty_part(
    cache: &TableMetadataCache,
    table_id: Uuid,
    format_version: FormatVersion,
    key: &CachedPartKey,
) -> bool {
    let manifest = &key.manifest;
    match &key.part {
        ManifestPart::PartitionList => {
            panic!("the partitions of a manifest are never stored unasked")
        }
        ManifestPart::ForPartition { partition, part } => match part {
            PartitionPart::Files => {
                let files = Arc::new(build_empty_files());
                cache.store_unasked_manifest_files(
                    table_id,
                    format_version,
                    manifest,
                    partition,
                    files,
                )
            }
            PartitionPart::ColumnBounds { field_id } => {
                let bounds = vec![(*field_id, Arc::new(build_empty_bounds()))];
                cache.store_unasked_column_bounds(
                    table_id,
                    format_version,
                    manifest,
                    partition,
                    bounds,
                )
            }
        },
    }
}

/// Whether the table `table_id` holds a part at `key`, of a manifest
/// read under `format_version`, which is counted as used now when it does.
pub fn find_part(
    cache: &TableMetadataCache,
    table_id: Uuid,
    format_version: FormatVersion,
    key: &CachedPartKey,
) -> bool {
    let manifest = &key.manifest;
    match &key.part {
        ManifestPart::PartitionList => {
            find_partitions_parts(cache, table_id, format_version, &[manifest])[0]
        }
        ManifestPart::ForPartition { partition, part } => match part {
            PartitionPart::Files => {
                let found = cache.find_manifest_files(
                    table_id,
                    format_version,
                    [(manifest, partition)].into_iter(),
                );
                found[0].is_some()
            }
            PartitionPart::ColumnBounds { field_id } => {
                let asked = [(manifest, partition, *field_id)].into_iter();
                cache.find_column_bounds(table_id, format_version, asked)[0].is_some()
            }
        },
    }
}

/// Whether the table `table_id` holds the partitions of each of the
/// manifests at `manifests`, read under `format_version`, in the order given, asked of the cache at once: each found
/// is counted as used now, in that order.
pub fn find_partitions_parts(
    cache: &TableMetadataCache,
    table_id: Uuid,
    format_version: FormatVersion,
    manifests: &[&Arc<str>],
) -> Vec<bool> {
    let found = cache.find_manifest_partitions(table_id, format_version, manifests.iter().copied());
    found.iter().map(Option::is_some).collect()
}

impl TableMetadataCache {
    /// The parts the table `table_id` holds of the manifest at `manifest`,
    /// read under `format_version`, in the order
    /// of the parts ([`ManifestPart`]): the manifest's partitions, then
    /// partition by partition in their order, a partition's
    /// files before its columns by field id. None is counted as used.
    pub fn list_manifest_parts(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &str,
    ) -> Vec<CachedPartKey> {
        let entries = self.lock_entries();
        let held_table = entries.tables.get(&table_id);
        let held_manifests = held_table.and_then(|held| held.find_manifests(format_version));
        let Some(held_manifest) = held_manifests.and_then(|held| held.get(manifest)) else {
            return Vec::new();
        };
        let mut parts = Vec::from_iter(
            held_manifest
                .partition_list_part
                .iter()
                .map(|_| ManifestPart::PartitionList),
        );
        for (partition, held_partition) in &held_manifest.partitions {
            let files = held_partition.files.iter().map(|_| PartitionPart::Files);
            let bounds = held_partition.bounds.iter().map(|held| {
                let field_id = held.field_id;
                PartitionPart::ColumnBounds { field_id }
            });
            parts.extend(files.chain(bounds).map(|part| {
                let partition = Arc::clone(partition);
                ManifestPart::ForPartition { partition, part }
            }));
        }
        parts.sort();
        let path: Arc<str> = Arc::from(manifest);
        let parts = parts.into_iter();
        parts
            .map(|part| CachedPartKey {
                manifest: Arc::clone(&path),
                part,
            })
            .collect()
    }

    /// Let go of every entry of every table.
    pub fn remove_all_tables(&self) {
        let mut entries = self.lock_entries();
        let capacity_bytes = entries.capacity_bytes;
        let left = std::mem::replace(&mut *entries, HeldEntries::new(capacity_bytes));
        drop(entries);
        drop(left);
    }
}

impl Table {
    /// Where each part the table's cache holds of the manifest at
    /// `manifest` is, the manifest by the path a manifest list of the table
    /// gives it, in the order of the parts. None is counted as used.
    pub fn list_cached_manifest_parts(&self, manifest: &str) -> Vec<CachedPartKey> {
        self.cache_slot.cache.list_manifest_parts(
            self.cache_slot.table_id,
            self.format_version(),
            manifest,
        )
    }
}
