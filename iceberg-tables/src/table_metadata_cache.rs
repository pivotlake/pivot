//! The table metadata cache: the parts of tables' data manifests and the
//! footers of their data files, as this process holds them between uses.
//! It is one cache for every table it is handed to, within one budget of
//! memory.
//!
//! Nothing here makes a cache or keeps one: whoever owns the tables makes
//! it once ([`TableMetadataCache::new`]) with the budget it gives it, and
//! hands it to each table as it loads or creates it, with the id the table's
//! entries are held under ([`TableCacheSlot`]). A server makes one for the
//! tables of all its datastores, so they share one budget; two tables
//! handed different caches share nothing.
//!
//! Each of the two costs reads and a decode to get again:
//!
//! - A data file's footer: the file's row groups ([`RowGroupMetadata`]) as
//!   the footer read produced them, each reading through the open of the
//!   file that read made, with no row deleted. A data file never changes,
//!   so its row groups serve every version that lists the file, shared as
//!   they are: a scan's selection takes them with nothing opened, read or
//!   copied. Row-level deletes belong to a version, so a scan records them
//!   on copies of its own and the row groups held here never carry any.
//!   A local file held here is held open, so the footer of a file the
//!   table deletes is dropped with it
//!   ([`TableMetadataCache::remove_footers`]).
//! - A part of one of a table's data manifests
//!   (`files::manifest_parts`): the partitions the
//!   manifest holds, the files of one partition in it, or what it records
//!   of one column in those files. A manifest never changes, so neither
//!   does a part of it. A parse of a manifest yields parts nobody asked for
//!   beside those that were: the first are stored as used now, the others
//!   as the first the cache lets go of, and only while it has room for them
//!   (`TableMetadataCache::store_unasked_manifest_files`,
//!   `TableMetadataCache::store_unasked_column_bounds`), so what a parse
//!   happens to yield never costs the cache a part that is in use.
//!
//! Each kind is held as what it is. An entry of either is found first by
//! its table: one id, which whoever owns the table gives it when it loads
//! it, so a table made anew where another was shares nothing with it. A
//! footer is then found by what it was parsed under ([`FooterSchema`]) and
//! by the location of its file, and a table holds one footer per data file
//! and way of parsing it. A part is found by the format version its
//! manifest is read under, by its manifest, and by which part of it it is
//! ([`ManifestPart`]), a level at a time: a table's manifests by their
//! path, a manifest's partitions each by itself, and under a partition its
//! files and its columns by field id. What a manifest holds is therefore
//! let go of as the one thing it stands under
//! ([`TableMetadataCache::remove_manifests`]), which is what whoever
//! deletes a manifest calls, and so is what a table holds.
//!
//! Whatever asks one cache under equal names finds the same entries, and
//! nothing leaves the cache because a table object went away: an entry
//! stays until the budget lets go of it, until whoever holds its table
//! removes it ([`TableMetadataCache::remove_table`]), or until the last
//! handle to the cache is dropped. The cache knows a table only
//! while it holds an entry of it, and a manifest or a partition of one
//! only while it holds a part of it.
//!
//! The budget is of estimated memory: each entry's
//! ([`measure_footer_entry`] for a footer), with the cache's own record of
//! it, and what the cache keeps once of each table, each way its footers
//! were parsed, each format version its manifests are read under, each
//! manifest and each partition of one it holds an entry of. Footers and parts stand in one
//! order of last use, and past the budget the cache lets go of the entries
//! used longest ago, of whichever kind and whichever table, one at a time,
//! keeping the one just stored. A scan marks the parts it selected through
//! as used once it has taken its files' footers, so the footers go before
//! the parts. While the footers a scan read are stored, a part it selected
//! through that has become the entry used longest ago is counted as used
//! instead of leaving, once ([`TableMetadataCache::store_footers`]), so
//! what leaves for a scan's footers is whatever was used before the scan,
//! then its footers, and only then its parts. The budget bounds what the
//! cache holds, not what the process does: a scan holds the parts it is
//! selecting through and the row groups it took for as long as it needs
//! them, cached or not, and the cache only drops its own hold on an entry.
//! What the cache lets go of is dropped once the cache is unlocked: a
//! footer that goes closes its file.

use std::sync::{Arc, Mutex, MutexGuard};

use iceberg::spec::FormatVersion;
use parquet_engine::RowGroupMetadata;
use uuid::Uuid;

use crate::files::manifest_parts::{CachedColumnBounds, CachedPartitionFiles, ManifestPartitions};
use crate::files::partition::Partition;

pub(crate) mod held_entries;
mod manifest_part;
mod use_order;

pub use held_entries::measure_footer_entry;
pub use manifest_part::{CachedPartKey, ManifestPart, PartitionPart};

use held_entries::{HeldEntries, LeftEntries, SparedParts};
use manifest_part::ManifestPartValue;

/// The budget whoever makes the cache of a process's tables gives it when
/// nothing says otherwise.
pub const DEFAULT_TABLE_METADATA_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

/// Where one table's entries are: the cache that holds them, and the id it
/// holds them under. Whoever loads or creates a table hands it one, and
/// every version derived from that table keeps it.
#[derive(Clone)]
pub struct TableCacheSlot {
    /// The cache the table's footers and the parts of its manifests are
    /// kept in.
    pub cache: Arc<TableMetadataCache>,
    /// The id the cache holds the table's entries under: one per table,
    /// the same for every load of it, so a table made anew where another
    /// was shares nothing with it.
    pub table_id: Uuid,
}

/// What a footer of a table's data file is parsed under: a footer parsed
/// under another is not the same entry. Both fields are compared whole,
/// and the value is cloned by sharing its string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FooterSchema {
    /// The id of the schema the footer is parsed against.
    pub schema_id: i32,
    /// The table property recording the Pivot types the schema's fields
    /// are read as, when the table has it.
    pub logical_types: Option<Arc<str>>,
}

/// The footers and the manifest parts of every table handed this cache,
/// within a budget.
pub struct TableMetadataCache {
    /// Every entry held, footers and parts, in one order of last use.
    entries: Mutex<HeldEntries>,
}

impl std::fmt::Debug for TableMetadataCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TableMetadataCache")
            .field("free_bytes", &self.free_bytes())
            .finish_non_exhaustive()
    }
}

impl TableMetadataCache {
    /// A cache that holds entries of up to `capacity_bytes` of estimated
    /// memory between them.
    pub fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            entries: Mutex::new(HeldEntries::new(capacity_bytes)),
        })
    }

    pub(crate) fn lock_entries(&self) -> MutexGuard<'_, HeldEntries> {
        self.entries.lock().unwrap()
    }

    /// The row groups the table `table_id` holds for each of the files at
    /// `paths`, parsed under `schema`, in the order given, shared with the
    /// cache, each found counted as used now, so it is the last the cache
    /// lets go of.
    pub fn find_footers<'p>(
        &self,
        table_id: Uuid,
        schema: &FooterSchema,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let mut entries = self.lock_entries();
        entries.find_footers(table_id, schema, paths)
    }

    /// The row groups the table `table_id` holds for each of the files at
    /// `paths`, parsed under `schema`, in the order given, shared with the
    /// cache, which is left as it was: every footer found stays where it
    /// stood in the order of last use.
    pub fn peek_footers<'p>(
        &self,
        table_id: Uuid,
        schema: &FooterSchema,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<Vec<Arc<RowGroupMetadata>>>> {
        let entries = self.lock_entries();
        entries.peek_footers(table_id, schema, paths)
    }

    /// Hold in the table `table_id` each of `footers`, a file's row groups
    /// as its footer read produced them under `schema`, with no row
    /// deleted, for the file at the path they come with, replacing those
    /// already held for it so parsed. After each, while the cache holds
    /// more than its budget, let go of the entry used longest ago, but
    /// never of the entry used last.
    ///
    /// The parts at `spared_parts`, of the table's manifests read under
    /// `format_version`, are those the files were selected through, none
    /// for files no scan selected. One of them
    /// that has become the entry used longest ago is counted as used now
    /// instead of leaving, once, so the footers stored leave before the
    /// parts, the one just stored among them.
    ///
    /// Every footer is checked and measured before the cache is locked, and
    /// what leaves for the budget is dropped once it is unlocked, so the
    /// lock is held for the cache's own bookkeeping alone.
    ///
    /// # Panics
    ///
    /// When a row group of any of `footers` carries deleted rows; none of
    /// them is stored then.
    pub fn store_footers(
        &self,
        table_id: Uuid,
        schema: &FooterSchema,
        footers: Vec<(Arc<str>, Vec<Arc<RowGroupMetadata>>)>,
        format_version: FormatVersion,
        spared_parts: &[CachedPartKey],
    ) {
        let measured: Vec<(Arc<str>, Vec<Arc<RowGroupMetadata>>, u64)> = footers
            .into_iter()
            .map(|(path, row_groups)| {
                assert!(
                    row_groups
                        .iter()
                        .all(|row_group| row_group.deleted_rows.is_none()),
                    "a footer is held without any version's deleted rows"
                );
                let heap_bytes = measure_footer_entry(&path, &row_groups);
                (path, row_groups, heap_bytes)
            })
            .collect();
        if measured.is_empty() {
            return;
        }
        let mut left = LeftEntries::default();
        let mut entries = self.lock_entries();
        // The parts not spared yet: one spared leaves like any other entry
        // when it comes round again.
        let mut spared = SparedParts::new(table_id, format_version, spared_parts);
        for (path, row_groups, heap_bytes) in measured {
            entries.store_footer(
                table_id,
                schema,
                path,
                row_groups,
                heap_bytes,
                &mut spared,
                &mut left,
            );
        }
        drop(entries);
        drop(left);
    }

    /// The partitions each of the manifests at `manifests` holds, as the
    /// table `table_id` holds them of manifests read under
    /// `format_version`, in the order given, shared with the cache, each
    /// found counted as used now.
    pub(crate) fn find_manifest_partitions<'m>(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifests: impl Iterator<Item = &'m Arc<str>>,
    ) -> Vec<Option<Arc<ManifestPartitions>>> {
        let asked = manifests.map(|manifest| (manifest, ManifestPart::PartitionList));
        let found = self.find_parts(table_id, format_version, asked).into_iter();
        found
            .map(|value| match value? {
                ManifestPartValue::PartitionList(partitions) => Some(partitions),
                _ => unreachable!("the list of a manifest's partitions is held as one"),
            })
            .collect()
    }

    /// The files of each of `asked`, a partition in the
    /// manifest at a path, as the table `table_id` holds them of manifests
    /// read under `format_version`, in the order given, shared with the
    /// cache, each found counted as used now.
    pub(crate) fn find_manifest_files<'a>(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        asked: impl Iterator<Item = (&'a Arc<str>, &'a Arc<Partition>)>,
    ) -> Vec<Option<Arc<CachedPartitionFiles>>> {
        let asked = asked.map(|(manifest, partition)| {
            let part = PartitionPart::Files;
            let partition = Arc::clone(partition);
            (manifest, ManifestPart::ForPartition { partition, part })
        });
        let found = self.find_parts(table_id, format_version, asked).into_iter();
        found
            .map(|value| match value? {
                ManifestPartValue::Files(files) => Some(files),
                _ => unreachable!("the files of a partition are held as files"),
            })
            .collect()
    }

    /// What the table `table_id` holds of each of `asked`, a field by its
    /// id in the files of a partition in the manifest at a path, read
    /// under `format_version`, in the order given, shared with the cache,
    /// each found counted as used now.
    pub(crate) fn find_column_bounds<'a>(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        asked: impl Iterator<Item = (&'a Arc<str>, &'a Arc<Partition>, i32)>,
    ) -> Vec<Option<Arc<CachedColumnBounds>>> {
        let asked = asked.map(|(manifest, partition, field_id)| {
            let part = PartitionPart::ColumnBounds { field_id };
            let partition = Arc::clone(partition);
            (manifest, ManifestPart::ForPartition { partition, part })
        });
        let found = self.find_parts(table_id, format_version, asked).into_iter();
        found
            .map(|value| match value? {
                ManifestPartValue::ColumnBounds(bounds) => Some(bounds),
                _ => unreachable!("what a manifest records of a column is held as bounds"),
            })
            .collect()
    }

    /// The part the table `table_id` holds of each of `asked`, a part of
    /// the manifest at the path it comes with, read under
    /// `format_version`, in the order given, each found counted as used
    /// now.
    fn find_parts<'m>(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        asked: impl Iterator<Item = (&'m Arc<str>, ManifestPart)>,
    ) -> Vec<Option<ManifestPartValue>> {
        let mut entries = self.lock_entries();
        entries.find_parts(table_id, format_version, asked)
    }

    /// Hold in the table `table_id` the partitions the manifest at
    /// `manifest` holds, read under `format_version`, as used now,
    /// replacing those already held for it. While the cache then
    /// holds more than its budget, let go of the entry used longest ago,
    /// but never of the part just stored.
    pub(crate) fn store_manifest_partitions(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &Arc<str>,
        partitions: Arc<ManifestPartitions>,
    ) {
        let key = CachedPartKey::of_partitions(manifest);
        let value = ManifestPartValue::PartitionList(partitions);
        self.store_parts(table_id, format_version, vec![(key, value)]);
    }

    /// Hold in the table `table_id` the files of the partition `partition`
    /// in the manifest at `manifest`, read under `format_version`, as
    /// [`store_manifest_partitions`](Self::store_manifest_partitions)
    /// holds a manifest's partitions.
    pub(crate) fn store_manifest_files(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &Arc<str>,
        partition: &Arc<Partition>,
        files: Arc<CachedPartitionFiles>,
    ) {
        let key = CachedPartKey::of_files(manifest, partition);
        self.store_parts(
            table_id,
            format_version,
            vec![(key, ManifestPartValue::Files(files))],
        );
    }

    /// Hold in the table `table_id` each of `bounds`, what the manifest at
    /// `manifest`, read under `format_version`, records of the field whose
    /// id it comes with in the files of the partition `partition`, each as
    /// [`store_manifest_partitions`](Self::store_manifest_partitions)
    /// holds a manifest's partitions.
    pub(crate) fn store_column_bounds(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &Arc<str>,
        partition: &Arc<Partition>,
        bounds: Vec<(i32, Arc<CachedColumnBounds>)>,
    ) {
        self.store_parts(
            table_id,
            format_version,
            locate_bounds(manifest, partition, bounds),
        );
    }

    /// Hold in the table `table_id` the files of the partition `partition`
    /// in the manifest at `manifest`, read under `format_version`, which
    /// nobody asked for, as the first entry
    /// the cache lets go of, and only when the budget has room for them:
    /// nothing leaves the cache for them, and files already held for the
    /// partition stay as they are. Whether they are held afterwards.
    pub(crate) fn store_unasked_manifest_files(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &Arc<str>,
        partition: &Arc<Partition>,
        files: Arc<CachedPartitionFiles>,
    ) -> bool {
        let key = CachedPartKey::of_files(manifest, partition);
        self.store_unasked_parts(
            table_id,
            format_version,
            vec![(key, ManifestPartValue::Files(files))],
        )
    }

    /// Hold in the table `table_id` each of `bounds`, what the manifest at
    /// `manifest`, read under `format_version`, records of the field whose
    /// id it comes with in the files of the partition `partition`, which
    /// nobody asked for, each as
    /// [`store_unasked_manifest_files`](Self::store_unasked_manifest_files)
    /// holds a partition's files. Whether every one of them is held
    /// afterwards.
    pub(crate) fn store_unasked_column_bounds(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &Arc<str>,
        partition: &Arc<Partition>,
        bounds: Vec<(i32, Arc<CachedColumnBounds>)>,
    ) -> bool {
        self.store_unasked_parts(
            table_id,
            format_version,
            locate_bounds(manifest, partition, bounds),
        )
    }

    /// Hold in the table `table_id` each of `parts` at the key it
    /// comes with, of a manifest read under `format_version`, as used now. What leaves the cache for the budget is dropped once the
    /// cache is unlocked: a footer that goes closes its file.
    fn store_parts(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        parts: Vec<(CachedPartKey, ManifestPartValue)>,
    ) {
        let mut left = LeftEntries::default();
        let mut entries = self.lock_entries();
        for (key, value) in parts {
            entries.store_part(table_id, format_version, &key, value, &mut left);
        }
        drop(entries);
        drop(left);
    }

    /// Hold in the table `table_id` each of `parts`, which nobody asked
    /// for, at the key it comes with, of a manifest read under
    /// `format_version`, while the budget has room. Whether every one
    /// of them is held afterwards.
    fn store_unasked_parts(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        parts: Vec<(CachedPartKey, ManifestPartValue)>,
    ) -> bool {
        let mut entries = self.lock_entries();
        let mut is_every_part_held = true;
        for (key, value) in parts {
            is_every_part_held &= entries.store_unasked_part(table_id, format_version, &key, value);
        }
        is_every_part_held
    }

    /// Count the parts the table `table_id` holds at `parts`, of manifests
    /// read under `format_version`, as used now. One the cache let go of
    /// stays let go of.
    pub fn mark_manifest_parts_used(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        parts: &[CachedPartKey],
    ) {
        let mut entries = self.lock_entries();
        entries.mark_parts_used(table_id, format_version, parts);
    }

    /// How much of its budget the cache has left: what it can still hold
    /// without letting go of an entry.
    pub fn free_bytes(&self) -> u64 {
        self.lock_entries().free_bytes()
    }

    /// Let go of the footers the table `table_id` holds for the files at
    /// `paths`, parsed under `schema`, and with each of the cache's hold on
    /// the open of its file, which is dropped once the cache is unlocked.
    pub fn remove_footers<'p>(
        &self,
        table_id: Uuid,
        schema: &FooterSchema,
        paths: impl Iterator<Item = &'p str>,
    ) {
        let mut left = LeftEntries::default();
        let mut entries = self.lock_entries();
        for path in paths {
            entries.remove_footer(table_id, schema, path, &mut left);
        }
        entries.compact_sparse_slots();
        drop(entries);
        drop(left);
    }

    /// Let go of every part the table `table_id` holds of each of the
    /// manifests at `manifests`, read under `format_version`, a manifest's
    /// as the one thing they stand under: what whoever deletes a table's
    /// manifests calls for them. A path the cache holds no part of, a
    /// manifest's or any other object's, is left at that.
    pub fn remove_manifests<'m>(
        &self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifests: impl Iterator<Item = &'m str>,
    ) {
        let mut left = LeftEntries::default();
        let mut entries = self.lock_entries();
        for manifest in manifests {
            entries.remove_manifest(table_id, format_version, manifest, &mut left);
        }
        entries.compact_sparse_slots();
        drop(entries);
        drop(left);
    }

    /// Let go of every entry of the table `table_id`, whatever its footers
    /// were parsed under and its manifests read under: what whoever
    /// deletes a table's directory calls first, and whoever is the last to
    /// use a table of its own. A footer held holds its file open, and a
    /// deleted local file keeps its blocks for as long as it is open.
    pub fn remove_table(&self, table_id: Uuid) {
        let mut left = LeftEntries::default();
        let mut entries = self.lock_entries();
        entries.remove_table(table_id, &mut left);
        entries.compact_sparse_slots();
        drop(entries);
        drop(left);
    }
}

/// Each of `bounds` with its key: what the manifest at `manifest`
/// records of the field whose id it comes with in the files of the
/// partition `partition`.
fn locate_bounds(
    manifest: &Arc<str>,
    partition: &Arc<Partition>,
    bounds: Vec<(i32, Arc<CachedColumnBounds>)>,
) -> Vec<(CachedPartKey, ManifestPartValue)> {
    let bounds = bounds.into_iter();
    bounds
        .map(|(field_id, bounds)| {
            let key = CachedPartKey::of_bounds(manifest, partition, field_id);
            (key, ManifestPartValue::ColumnBounds(bounds))
        })
        .collect()
}
