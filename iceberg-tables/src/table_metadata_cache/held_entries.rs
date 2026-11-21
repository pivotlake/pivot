//! What the cache holds, behind its lock: every table's footers and the
//! parts of its manifests, within one budget and in one order of last use.
//!
//! The entries are held under what names them, level by level
//! ([`HeldTable`], [`HeldManifest`]): a table by its id, its footers by
//! what they were parsed under ([`FooterSchema`]) and then by the location
//! of their file, its manifests by the format version they are read under
//! and then by their path, a manifest's list of partitions beside its
//! partitions each by itself, and under a partition ([`HeldPartition`])
//! its files beside what the manifest records of each column
//! ([`HeldColumnBounds`]), by field id. So an entry is found by one lookup
//! a level, and what a manifest or a table holds leaves as the one node it
//! stands under. A level holds a name exactly while something is held
//! under it.
//!
//! Every entry stands in the order of last use ([`UseOrder`]) with its
//! key, and past the budget the entry used longest ago is taken out
//! from under the names its key gives. The budget counts each entry
//! ([`measure_footer_entry`], [`measure_part_entry`]) and what the cache
//! keeps once for a table, for each group of its footers or of its
//! manifests, for a manifest and for a partition of one
//! ([`TABLE_RECORD_BYTES`], [`measure_footer_group_record`],
//! [`MANIFEST_GROUP_RECORD_BYTES`], [`measure_manifest_record`],
//! [`measure_partition_record`]) while it holds an entry under them.

use std::borrow::Borrow;
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use hashbrown::{Equivalent, HashMap};
use iceberg::spec::FormatVersion;
use parquet_engine::{RowGroupMetadata, estimate_footer_heap_bytes, measure_shared_allocation};
use uuid::Uuid;

use super::FooterSchema;
use super::manifest_part::{CachedPartKey, ManifestPart, ManifestPartValue, PartitionPart};
use super::use_order::{EntryKey, OrderedEntry, UseOrder};
use crate::files::partition::Partition;

/// A file's row groups, in file order, none with a deleted row.
pub(super) type FooterRowGroups = Vec<Arc<RowGroupMetadata>>;

/// What is held under each of the names of one level.
type NodeMap<Name, Node> = HashMap<Name, Node, ahash::RandomState>;

struct HeldFooter {
    row_groups: FooterRowGroups,
    /// The slot the entry stands in in the order of last use.
    slot: usize,
}

pub(crate) struct HeldPart {
    value: ManifestPartValue,
    /// The slot the entry stands in in the order of last use.
    slot: usize,
}

/// The footers held of the files of one table that were parsed alike, by
/// the location of their file: never none.
type HeldFooters = NodeMap<Arc<str>, HeldFooter>;

/// The manifests of one table, read under one format version, that a part
/// is held of, by path: never none.
pub(crate) type HeldManifests = NodeMap<Arc<str>, HeldManifest>;

/// What is held of one table: never nothing.
#[derive(Default)]
pub(crate) struct HeldTable {
    /// The table's footers, grouped by what they were parsed under, in an
    /// allocation of exactly the groups' number: a table is read under few
    /// schemas at a time, so a group is found by comparing.
    footers: Vec<(Arc<FooterSchema>, HeldFooters)>,
    /// The manifests a part is held of, grouped by the format version they
    /// are read under, in an allocation of exactly the groups' number.
    manifests: Vec<(FormatVersion, HeldManifests)>,
}

impl HeldTable {
    fn is_empty(&self) -> bool {
        self.footers.is_empty() && self.manifests.is_empty()
    }

    /// The footers held that were parsed under `schema`.
    fn find_footers(&self, schema: &FooterSchema) -> Option<&HeldFooters> {
        let position = find_group(&self.footers, schema)?;
        Some(&self.footers[position].1)
    }

    /// The manifests read under `format_version` that a part is held of.
    pub(crate) fn find_manifests(&self, format_version: FormatVersion) -> Option<&HeldManifests> {
        let position = find_group(&self.manifests, &format_version)?;
        Some(&self.manifests[position].1)
    }
}

/// Where the group named `name` stands among `groups`.
fn find_group<Asked, Name, Node>(groups: &[(Name, Node)], name: &Asked) -> Option<usize>
where
    Asked: PartialEq + ?Sized,
    Name: Borrow<Asked>,
{
    let mut names = groups.iter().map(|(held_name, _)| held_name);
    names.position(|held_name| Borrow::<Asked>::borrow(held_name) == name)
}

/// The parts held of one manifest: never none.
#[derive(Default)]
pub(crate) struct HeldManifest {
    /// The list of the partitions the manifest holds.
    pub(crate) partition_list_part: Option<HeldPart>,
    /// What is held of each partition the manifest holds a part of.
    pub(crate) partitions: NodeMap<Arc<Partition>, HeldPartition>,
}

impl HeldManifest {
    fn is_empty(&self) -> bool {
        self.partition_list_part.is_none() && self.partitions.is_empty()
    }

    fn find_part(&self, part: &ManifestPart) -> Option<&HeldPart> {
        match part {
            ManifestPart::PartitionList => self.partition_list_part.as_ref(),
            ManifestPart::ForPartition { partition, part } => {
                self.partitions.get(&**partition)?.find_part(*part)
            }
        }
    }

    fn find_part_mut(&mut self, part: &ManifestPart) -> Option<&mut HeldPart> {
        match part {
            ManifestPart::PartitionList => self.partition_list_part.as_mut(),
            ManifestPart::ForPartition { partition, part } => {
                self.partitions.get_mut(&**partition)?.find_part_mut(*part)
            }
        }
    }
}

/// What is held of one partition in one manifest: never nothing.
#[derive(Default)]
pub(crate) struct HeldPartition {
    /// The partition's files in the manifest.
    pub(crate) files: Option<HeldPart>,
    /// What the manifest records of each column in those files, in the
    /// order of the fields' ids, in an allocation of exactly their number.
    pub(crate) bounds: Vec<HeldColumnBounds>,
}

/// What one manifest records of one column in the files of one partition.
pub(crate) struct HeldColumnBounds {
    pub(crate) field_id: i32,
    part: HeldPart,
}

impl HeldPartition {
    fn is_empty(&self) -> bool {
        self.files.is_none() && self.bounds.is_empty()
    }

    fn find_part(&self, part: PartitionPart) -> Option<&HeldPart> {
        match part {
            PartitionPart::Files => self.files.as_ref(),
            PartitionPart::ColumnBounds { field_id } => {
                let position = self.find_bounds(field_id).ok()?;
                Some(&self.bounds[position].part)
            }
        }
    }

    fn find_part_mut(&mut self, part: PartitionPart) -> Option<&mut HeldPart> {
        match part {
            PartitionPart::Files => self.files.as_mut(),
            PartitionPart::ColumnBounds { field_id } => {
                let position = self.find_bounds(field_id).ok()?;
                Some(&mut self.bounds[position].part)
            }
        }
    }

    /// Where the bounds of the field `field_id` stand among those held,
    /// or where they would.
    fn find_bounds(&self, field_id: i32) -> Result<usize, usize> {
        self.bounds
            .binary_search_by_key(&field_id, |held| held.field_id)
    }

    /// Hold `held` as `part`, and hand back the part it took the place of.
    fn hold_part(&mut self, part: PartitionPart, held: HeldPart) -> Option<HeldPart> {
        match part {
            PartitionPart::Files => self.files.replace(held),
            PartitionPart::ColumnBounds { field_id } => match self.find_bounds(field_id) {
                Ok(position) => Some(std::mem::replace(&mut self.bounds[position].part, held)),
                Err(position) => {
                    self.bounds.reserve_exact(1);
                    let held_bounds = HeldColumnBounds {
                        field_id,
                        part: held,
                    };
                    self.bounds.insert(position, held_bounds);
                    None
                }
            },
        }
    }

    /// Take `part` out, the bounds left in an allocation of exactly their
    /// number.
    fn detach_part(&mut self, part: PartitionPart) -> Option<HeldPart> {
        match part {
            PartitionPart::Files => self.files.take(),
            PartitionPart::ColumnBounds { field_id } => {
                let position = self.find_bounds(field_id).ok()?;
                let held_bounds = self.bounds.remove(position);
                self.bounds.shrink_to_fit();
                Some(held_bounds.part)
            }
        }
    }

    /// Every part held of the partition: its files, then its columns in
    /// the order of their field ids.
    fn into_parts(self) -> impl Iterator<Item = HeldPart> {
        let of_columns = self.bounds.into_iter().map(|held| held.part);
        self.files.into_iter().chain(of_columns)
    }
}

/// The parts of one table's manifests as they are asked for, a manifest at
/// a time: the manifest asked last is found again by its path's allocation
/// alone.
struct AskedManifests<'held, 'asked> {
    manifests: Option<&'held HeldManifests>,
    latest: Option<(&'asked Arc<str>, Option<&'held HeldManifest>)>,
}

impl<'held, 'asked> AskedManifests<'held, 'asked> {
    fn new(manifests: Option<&'held HeldManifests>) -> Self {
        Self {
            manifests,
            latest: None,
        }
    }

    fn find_part(
        &mut self,
        manifest: &'asked Arc<str>,
        part: &ManifestPart,
    ) -> Option<&'held HeldPart> {
        let held_manifest = match self.latest {
            Some((asked_last, found)) if Arc::ptr_eq(asked_last, manifest) => found,
            _ => {
                let found = self.manifests?.get(&**manifest);
                self.latest = Some((manifest, found));
                found
            }
        };
        held_manifest?.find_part(part)
    }
}

/// The parts a store of footers spares, each once: those of one table's
/// manifests, read under one format version.
pub(super) struct SparedParts<'asked> {
    table_id: Uuid,
    format_version: FormatVersion,
    keys: HashSet<&'asked CachedPartKey, ahash::RandomState>,
}

impl<'asked> SparedParts<'asked> {
    pub(super) fn new(
        table_id: Uuid,
        format_version: FormatVersion,
        keys: &'asked [CachedPartKey],
    ) -> Self {
        Self {
            table_id,
            format_version,
            keys: keys.iter().collect(),
        }
    }

    /// Whether the part at `key` of the manifests of the table
    /// `table_id` read under `format_version` is spared, which it then is
    /// no more.
    fn remove(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        key: &CachedPartKey,
    ) -> bool {
        self.table_id == table_id && self.format_version == format_version && self.keys.remove(key)
    }
}

/// What a level spends on holding one name with a node of `entry_bytes`
/// between them: the two and the byte they are found by, counted as if the
/// level had just grown to twice the room, when the least of it is in use.
const fn measure_node_entry(entry_bytes: usize) -> usize {
    ((entry_bytes + 1) * 16).div_ceil(7)
}

/// The fixed overhead the cache adds for one footer, besides the footer
/// itself and the location of its file: its slot in the order of last use,
/// and its place among the footers of its table parsed as it was.
const FOOTER_OVERHEAD_BYTES: usize = UseOrder::SLOT_BYTES + size_of::<(Arc<str>, HeldFooter)>();

/// What the footer of the file at `path`, read as `row_groups`, counts for
/// in the budget: the row groups' estimated memory and the cache's own
/// record of the entry, the path with it.
pub fn measure_footer_entry(path: &str, row_groups: &[Arc<RowGroupMetadata>]) -> u64 {
    let overhead_bytes = FOOTER_OVERHEAD_BYTES + measure_shared_allocation(path.len());
    estimate_footer_heap_bytes(row_groups) + overhead_bytes as u64
}

/// What `value`, held as the part `part` of a manifest, counts for in the
/// budget: the part's estimated memory, its slot in the order of last use
/// and, for what is recorded of a column, its place among the columns of
/// its partition. A manifest's list of partitions and a partition's files
/// stand in the node they are held under, which its record counts.
fn measure_part_entry(part: &ManifestPart, value: &ManifestPartValue) -> u64 {
    let place_bytes = match part {
        ManifestPart::PartitionList => 0,
        ManifestPart::ForPartition { part, .. } => match part {
            PartitionPart::Files => 0,
            PartitionPart::ColumnBounds { .. } => size_of::<HeldColumnBounds>(),
        },
    };
    value.estimate_heap_bytes() + (UseOrder::SLOT_BYTES + place_bytes) as u64
}

/// What the cache keeps of a table while it holds an entry of it, whatever
/// the number of entries: its id and the node its entries stand under.
const TABLE_RECORD_BYTES: u64 = measure_node_entry(size_of::<(Uuid, HeldTable)>()) as u64;

/// What the cache keeps of the manifests of a table read under one format
/// version while it holds a part of one, whatever the number of parts: the
/// version and the node the manifests stand under.
const MANIFEST_GROUP_RECORD_BYTES: u64 = size_of::<(FormatVersion, HeldManifests)>() as u64;

/// What the cache keeps of the footers of a table parsed under `schema`
/// while it holds one, whatever their number: what they were parsed under,
/// shared with each of them in the order of last use, the string it holds
/// outside itself, and the node the footers stand under.
fn measure_footer_group_record(schema: &FooterSchema) -> u64 {
    let logical_types = schema.logical_types.as_ref();
    let logical_types_bytes = logical_types.map_or(0, |logical_types| {
        measure_shared_allocation(logical_types.len())
    });
    let record_bytes = size_of::<(Arc<FooterSchema>, HeldFooters)>()
        + measure_shared_allocation(size_of::<FooterSchema>())
        + logical_types_bytes;
    record_bytes as u64
}

/// What the cache keeps of the manifest at `manifest` while it holds a
/// part of it, whatever the number of parts: the path and the node its
/// parts stand under.
fn measure_manifest_record(manifest: &str) -> u64 {
    let record_bytes = measure_node_entry(size_of::<(Arc<str>, HeldManifest)>())
        + measure_shared_allocation(manifest.len());
    record_bytes as u64
}

/// What the cache keeps of `partition` while it holds a part of it in one
/// manifest, whatever the number of parts: the partition and the node its
/// parts stand under, the place of its files with it. The partition is
/// counted whole, as if nothing else held it: the manifest's list of
/// partitions, which counts it too, may leave the cache before the node
/// does.
fn measure_partition_record(partition: &Partition) -> u64 {
    let node_bytes = measure_node_entry(size_of::<(Arc<Partition>, HeldPartition)>());
    (node_bytes + partition.estimate_heap_bytes()) as u64
}

/// The node `nodes` holds under `name`, with the name as the level holds
/// it: an empty node is made under what `share_name` hands over, and
/// `record_bytes` added to `held_bytes` for it, when none is held yet.
fn hold_node<'nodes, Asked, Name, Node>(
    nodes: &'nodes mut NodeMap<Name, Node>,
    name: &Asked,
    share_name: impl FnOnce() -> Name,
    record_bytes: u64,
    held_bytes: &mut u64,
) -> (&'nodes Name, &'nodes mut Node)
where
    Asked: Hash + Equivalent<Name> + ?Sized,
    Name: Hash + Eq,
    Node: Default,
{
    if !nodes.contains_key(name) {
        nodes.insert(share_name(), Node::default());
        *held_bytes += record_bytes;
    }
    let held = nodes.get_key_value_mut(name);
    held.expect("a node just made is held")
}

/// The group `groups` holds under `name`, with the name as the level holds
/// it: an empty group is made under what `share_name` hands over, and
/// `record_bytes` added to `held_bytes` for it, when none is held yet.
fn hold_group<'groups, Asked, Name, Node>(
    groups: &'groups mut Vec<(Name, Node)>,
    name: &Asked,
    share_name: impl FnOnce() -> Name,
    record_bytes: u64,
    held_bytes: &mut u64,
) -> (&'groups Name, &'groups mut Node)
where
    Asked: PartialEq + ?Sized,
    Name: Borrow<Asked>,
    Node: Default,
{
    let position = find_group(groups, name).unwrap_or_else(|| {
        groups.reserve_exact(1);
        groups.push((share_name(), Node::default()));
        *held_bytes += record_bytes;
        groups.len() - 1
    });
    let (name, node) = &mut groups[position];
    (name, node)
}

/// Take the group at `position` out of `groups`, which keep an allocation
/// of exactly their number.
fn remove_group<Name, Node>(groups: &mut Vec<(Name, Node)>, position: usize) {
    groups.swap_remove(position);
    groups.shrink_to_fit();
}

/// What left the cache, for whoever took it out to drop once the cache is
/// unlocked: a footer that goes closes its file.
#[derive(Default)]
pub(super) struct LeftEntries {
    footers: Vec<FooterRowGroups>,
    parts: Vec<ManifestPartValue>,
}

/// The footers and the parts of manifests held, within a budget.
pub(crate) struct HeldEntries {
    pub(crate) capacity_bytes: u64,
    /// What the entries held and the records of the names they stand under
    /// count for between them.
    held_bytes: u64,
    /// The tables an entry is held of, a footer or a part, by table id.
    pub(crate) tables: NodeMap<Uuid, HeldTable>,
    order: UseOrder,
}

impl HeldEntries {
    /// Stores that hold entries of up to `capacity_bytes` of estimated
    /// memory between them.
    pub(crate) fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            held_bytes: 0,
            tables: NodeMap::default(),
            order: UseOrder::new(),
        }
    }

    /// How much of the budget is left.
    pub(super) fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.held_bytes)
    }

    /// The row groups the table `table_id` holds for each of the files at
    /// `paths`, parsed under `schema`, in the order given, each found
    /// counted as used now.
    pub(super) fn find_footers<'p>(
        &mut self,
        table_id: Uuid,
        schema: &FooterSchema,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<FooterRowGroups>> {
        let held_table = self.tables.get(&table_id);
        let held_footers = held_table.and_then(|held| held.find_footers(schema));
        paths
            .map(|path| {
                let held = held_footers?.get(path)?;
                self.order.mark_used(held.slot);
                Some(held.row_groups.clone())
            })
            .collect()
    }

    /// The row groups the table `table_id` holds for each of the files at
    /// `paths`, parsed under `schema`, in the order given, each found left
    /// where it stands in the order of last use.
    pub(super) fn peek_footers<'p>(
        &self,
        table_id: Uuid,
        schema: &FooterSchema,
        paths: impl Iterator<Item = &'p str>,
    ) -> Vec<Option<FooterRowGroups>> {
        let held_table = self.tables.get(&table_id);
        let held_footers = held_table.and_then(|held| held.find_footers(schema));
        paths
            .map(|path| Some(held_footers?.get(path)?.row_groups.clone()))
            .collect()
    }

    /// The part the table `table_id` holds of each of `asked`, a part of
    /// the manifest at the path it comes with, read under `format_version`,
    /// in the order given, each found counted as used now.
    pub(super) fn find_parts<'m>(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        asked: impl Iterator<Item = (&'m Arc<str>, ManifestPart)>,
    ) -> Vec<Option<ManifestPartValue>> {
        let held_table = self.tables.get(&table_id);
        let held_manifests = held_table.and_then(|held| held.find_manifests(format_version));
        let mut manifests = AskedManifests::new(held_manifests);
        asked
            .map(|(manifest, part)| {
                let held = manifests.find_part(manifest, &part)?;
                self.order.mark_used(held.slot);
                Some(held.value.clone())
            })
            .collect()
    }

    /// Count the parts the table `table_id` holds at `keys`, of
    /// manifests read under `format_version`, as used now.
    pub(super) fn mark_parts_used(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        keys: &[CachedPartKey],
    ) {
        let held_table = self.tables.get(&table_id);
        let held_manifests = held_table.and_then(|held| held.find_manifests(format_version));
        let mut manifests = AskedManifests::new(held_manifests);
        for key in keys {
            if let Some(held) = manifests.find_part(&key.manifest, &key.part) {
                self.order.mark_used(held.slot);
            }
        }
    }

    /// Hold `row_groups` in the table `table_id` for the file at `path`,
    /// parsed under `schema`, for `weight_bytes` of the budget, as used
    /// now, in place of the footer already held for the file so parsed.
    /// While more than the budget is held, the entries used longest ago
    /// leave into `left`, except that each of `spared` is counted as used
    /// now instead, once.
    #[allow(clippy::too_many_arguments)] // a footer's whole key, its weight, and where what leaves goes
    pub(super) fn store_footer(
        &mut self,
        table_id: Uuid,
        schema: &FooterSchema,
        path: Arc<str>,
        row_groups: FooterRowGroups,
        weight_bytes: u64,
        spared: &mut SparedParts<'_>,
        left: &mut LeftEntries,
    ) {
        let (_, held_table) = hold_node(
            &mut self.tables,
            &table_id,
            || table_id,
            TABLE_RECORD_BYTES,
            &mut self.held_bytes,
        );
        let (schema, held_footers) = hold_group(
            &mut held_table.footers,
            schema,
            || Arc::new(schema.clone()),
            measure_footer_group_record(schema),
            &mut self.held_bytes,
        );
        // The footer replaced leaves with the path it was held under, so
        // the path held is the one the order of last use has.
        if let Some(replaced) = held_footers.remove(&*path) {
            let entry = self.order.remove(replaced.slot);
            self.held_bytes -= entry.weight_bytes;
            left.footers.push(replaced.row_groups);
        }
        let key = EntryKey::Footer {
            table_id,
            schema: Arc::clone(schema),
            path: Arc::clone(&path),
        };
        let slot = self.order.push_newest(OrderedEntry { key, weight_bytes });
        held_footers.insert(path, HeldFooter { row_groups, slot });
        self.held_bytes += weight_bytes;
        self.evict_past_capacity(Some(spared), left);
    }

    /// Hold `value` as the part at `key` of the manifests of the table
    /// `table_id` read under `format_version`, as used now, in place of the
    /// part already held there. While more than the budget is held, the
    /// entries used longest ago leave into `left`.
    pub(super) fn store_part(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        key: &CachedPartKey,
        value: ManifestPartValue,
        left: &mut LeftEntries,
    ) {
        let push = UseOrder::push_newest;
        let replaced = self.hold_part(table_id, format_version, key, value, push);
        left.parts.extend(replaced);
        self.evict_past_capacity(None, left);
    }

    /// Hold `value` as the part at `key` of the manifests of the table
    /// `table_id` read under `format_version`, as the entry used longest
    /// ago, when the budget has room for it and for what the cache keeps
    /// of a table, a group of manifests, a manifest or a partition it
    /// holds nothing of yet: nothing leaves for it. A part already held
    /// there stays as it is, where it stood in the order of last use.
    /// Whether a part is held there afterwards.
    pub(super) fn store_unasked_part(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        key: &CachedPartKey,
        value: ManifestPartValue,
    ) -> bool {
        let held_table = self.tables.get(&table_id);
        let held_manifests = held_table.and_then(|held| held.find_manifests(format_version));
        let held_manifest = held_manifests.and_then(|held| held.get(&*key.manifest));
        if held_manifest.is_some_and(|held| held.find_part(&key.part).is_some()) {
            return true;
        }
        let mut added_bytes = measure_part_entry(&key.part, &value);
        if held_table.is_none() {
            added_bytes += TABLE_RECORD_BYTES;
        }
        if held_manifests.is_none() {
            added_bytes += MANIFEST_GROUP_RECORD_BYTES;
        }
        if held_manifest.is_none() {
            added_bytes += measure_manifest_record(&key.manifest);
        }
        if let ManifestPart::ForPartition { partition, .. } = &key.part
            && !held_manifest.is_some_and(|held| held.partitions.contains_key(&**partition))
        {
            added_bytes += measure_partition_record(partition);
        }
        if self.held_bytes.saturating_add(added_bytes) > self.capacity_bytes {
            return false;
        }
        let push = UseOrder::push_oldest;
        self.hold_part(table_id, format_version, key, value, push);
        true
    }

    /// Let go of the footer the table `table_id` holds for the file at
    /// `path`, parsed under `schema`, into `left`.
    pub(super) fn remove_footer(
        &mut self,
        table_id: Uuid,
        schema: &FooterSchema,
        path: &str,
        left: &mut LeftEntries,
    ) {
        if let Some(held) = self.detach_footer(table_id, schema, path) {
            self.release_slot(held.slot);
            left.footers.push(held.row_groups);
        }
    }

    /// Let go of every part of the manifest at `manifest` held for the
    /// table `table_id` as read under `format_version`, into `left`.
    pub(super) fn remove_manifest(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        manifest: &str,
        left: &mut LeftEntries,
    ) {
        let Some(held_table) = self.tables.get_mut(&table_id) else {
            return;
        };
        let Some(position) = find_group(&held_table.manifests, &format_version) else {
            return;
        };
        let held_manifests = &mut held_table.manifests[position].1;
        let Some(held_manifest) = held_manifests.remove(manifest) else {
            return;
        };
        if held_manifests.is_empty() {
            remove_group(&mut held_table.manifests, position);
            self.held_bytes -= MANIFEST_GROUP_RECORD_BYTES;
        }
        let is_table_empty = held_table.is_empty();
        self.release_manifest(manifest, held_manifest, left);
        if is_table_empty {
            self.forget_table(table_id);
        }
    }

    /// Let go of every footer and every part held for the table
    /// `table_id`, into `left`.
    pub(super) fn remove_table(&mut self, table_id: Uuid, left: &mut LeftEntries) {
        let Some(held_table) = self.forget_table(table_id) else {
            return;
        };
        for (schema, held_footers) in held_table.footers {
            self.held_bytes -= measure_footer_group_record(&schema);
            for (_, held) in held_footers {
                self.release_slot(held.slot);
                left.footers.push(held.row_groups);
            }
        }
        for (_, held_manifests) in held_table.manifests {
            self.held_bytes -= MANIFEST_GROUP_RECORD_BYTES;
            for (manifest, held_manifest) in held_manifests {
                self.release_manifest(&manifest, held_manifest, left);
            }
        }
    }

    /// Give back the room the entries that left took in the order of last
    /// use, once few enough of its slots hold an entry, and with it the
    /// room every level keeps spare.
    pub(super) fn compact_sparse_slots(&mut self) {
        let Self { order, tables, .. } = self;
        let has_moved = order.compact_sparse_slots(|key, slot| {
            let held_slot = match key {
                EntryKey::Footer {
                    table_id,
                    schema,
                    path,
                } => {
                    let held_table = tables.get_mut(table_id);
                    let held_footers = held_table.and_then(|held| {
                        let position = find_group(&held.footers, &**schema)?;
                        Some(&mut held.footers[position].1)
                    });
                    let held = held_footers.and_then(|held| held.get_mut(&**path));
                    held.map(|held| &mut held.slot)
                }
                EntryKey::Part {
                    table_id,
                    format_version,
                    key,
                } => {
                    let held_table = tables.get_mut(table_id);
                    let held_manifests = held_table.and_then(|held| {
                        let position = find_group(&held.manifests, format_version)?;
                        Some(&mut held.manifests[position].1)
                    });
                    let held = held_manifests
                        .and_then(|held| held.get_mut(&*key.manifest))
                        .and_then(|held| held.find_part_mut(&key.part));
                    held.map(|held| &mut held.slot)
                }
            };
            *held_slot.expect("an entry in the order is held under its key") = slot;
        });
        if !has_moved {
            return;
        }
        self.tables.shrink_to_fit();
        for held_table in self.tables.values_mut() {
            for (_, held_footers) in &mut held_table.footers {
                held_footers.shrink_to_fit();
            }
            for (_, held_manifests) in &mut held_table.manifests {
                held_manifests.shrink_to_fit();
                for held_manifest in held_manifests.values_mut() {
                    held_manifest.partitions.shrink_to_fit();
                }
            }
        }
    }

    /// Stand `value` at `key` of the manifests of the table `table_id`
    /// read under `format_version`, under the names the levels hold or new
    /// ones, and in the order of last use where `push` puts it, and count
    /// it and each new name in the budget. Hands back the part it took the
    /// place of, taken out of the order and the budget.
    fn hold_part(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        key: &CachedPartKey,
        value: ManifestPartValue,
        push: fn(&mut UseOrder, OrderedEntry) -> usize,
    ) -> Option<ManifestPartValue> {
        let Self {
            tables,
            order,
            held_bytes,
            ..
        } = self;
        let weight_bytes = measure_part_entry(&key.part, &value);
        let (_, held_table) = hold_node(
            tables,
            &table_id,
            || table_id,
            TABLE_RECORD_BYTES,
            held_bytes,
        );
        let (_, held_manifests) = hold_group(
            &mut held_table.manifests,
            &format_version,
            || format_version,
            MANIFEST_GROUP_RECORD_BYTES,
            held_bytes,
        );
        let (manifest, held_manifest) = hold_node(
            held_manifests,
            &*key.manifest,
            || Arc::clone(&key.manifest),
            measure_manifest_record(&key.manifest),
            held_bytes,
        );
        // The entry's key shares the names the levels hold, so each
        // name is one allocation however many entries stand under it.
        let mut stand_in_order = |part: ManifestPart| {
            let key = EntryKey::Part {
                table_id,
                format_version,
                key: CachedPartKey {
                    manifest: Arc::clone(manifest),
                    part,
                },
            };
            push(order, OrderedEntry { key, weight_bytes })
        };
        let replaced = match &key.part {
            ManifestPart::PartitionList => {
                let slot = stand_in_order(ManifestPart::PartitionList);
                held_manifest
                    .partition_list_part
                    .replace(HeldPart { value, slot })
            }
            ManifestPart::ForPartition { partition, part } => {
                let (partition, held_partition) = hold_node(
                    &mut held_manifest.partitions,
                    &**partition,
                    || Arc::clone(partition),
                    measure_partition_record(partition),
                    held_bytes,
                );
                let slot = stand_in_order(ManifestPart::ForPartition {
                    partition: Arc::clone(partition),
                    part: *part,
                });
                held_partition.hold_part(*part, HeldPart { value, slot })
            }
        };
        *held_bytes += weight_bytes;
        let replaced = replaced?;
        self.release_slot(replaced.slot);
        Some(replaced.value)
    }

    /// Take the entry standing in `slot` out of the order of last use and
    /// out of the budget, and hand back its key.
    fn release_slot(&mut self, slot: usize) -> EntryKey {
        let entry = self.order.remove(slot);
        self.held_bytes -= entry.weight_bytes;
        entry.key
    }

    /// Take every part of `held_manifest`, which stood under `manifest`,
    /// out of the order of last use and out of the budget, with the
    /// records of the manifest and of its partitions, into `left`.
    fn release_manifest(
        &mut self,
        manifest: &str,
        held_manifest: HeldManifest,
        left: &mut LeftEntries,
    ) {
        let partitions = held_manifest.partitions.keys();
        let partition_record_bytes: u64 = partitions
            .map(|partition| measure_partition_record(partition))
            .sum();
        self.held_bytes -= measure_manifest_record(manifest) + partition_record_bytes;
        let held_partitions = held_manifest.partitions.into_values();
        let of_partitions = held_partitions.flat_map(HeldPartition::into_parts);
        for held in held_manifest
            .partition_list_part
            .into_iter()
            .chain(of_partitions)
        {
            self.release_slot(held.slot);
            left.parts.push(held.value);
        }
    }

    /// Take the table `table_id` out from the tables, with everything
    /// still under it, and its record out of the budget.
    fn forget_table(&mut self, table_id: Uuid) -> Option<HeldTable> {
        let held_table = self.tables.remove(&table_id)?;
        self.held_bytes -= TABLE_RECORD_BYTES;
        Some(held_table)
    }

    /// Take the footer the table `table_id` holds for the file at `path`,
    /// parsed under `schema`, out from under its names, and with it each
    /// name it was the last entry under, whose record the budget gets
    /// back.
    fn detach_footer(
        &mut self,
        table_id: Uuid,
        schema: &FooterSchema,
        path: &str,
    ) -> Option<HeldFooter> {
        let held_table = self.tables.get_mut(&table_id)?;
        let position = find_group(&held_table.footers, schema)?;
        let held_footers = &mut held_table.footers[position].1;
        let held = held_footers.remove(path)?;
        if held_footers.is_empty() {
            remove_group(&mut held_table.footers, position);
            self.held_bytes -= measure_footer_group_record(schema);
            if held_table.is_empty() {
                self.forget_table(table_id);
            }
        }
        Some(held)
    }

    /// Take the part at `key` of the manifests of the table `table_id`
    /// read under `format_version` out from under its names, and with it
    /// each name it was the last entry under, whose record the budget gets
    /// back.
    fn detach_part(
        &mut self,
        table_id: Uuid,
        format_version: FormatVersion,
        key: &CachedPartKey,
    ) -> Option<HeldPart> {
        let held_table = self.tables.get_mut(&table_id)?;
        let group_position = find_group(&held_table.manifests, &format_version)?;
        let held_manifests = &mut held_table.manifests[group_position].1;
        let held_manifest = held_manifests.get_mut(&*key.manifest)?;
        let held = match &key.part {
            ManifestPart::PartitionList => held_manifest.partition_list_part.take()?,
            ManifestPart::ForPartition { partition, part } => {
                let held_partition = held_manifest.partitions.get_mut(&**partition)?;
                let held = held_partition.detach_part(*part)?;
                if held_partition.is_empty() {
                    held_manifest.partitions.remove(&**partition);
                    self.held_bytes -= measure_partition_record(partition);
                }
                held
            }
        };
        if held_manifest.is_empty() {
            held_manifests.remove(&*key.manifest);
            self.held_bytes -= measure_manifest_record(&key.manifest);
            if held_manifests.is_empty() {
                remove_group(&mut held_table.manifests, group_position);
                self.held_bytes -= MANIFEST_GROUP_RECORD_BYTES;
                if held_table.is_empty() {
                    self.forget_table(table_id);
                }
            }
        }
        Some(held)
    }

    /// While more than the budget is held, let go of the entry used
    /// longest ago, into `left`, but never of the one used last; a part
    /// among `spared` is counted as used now instead of leaving, and is
    /// spared no more.
    fn evict_past_capacity(
        &mut self,
        mut spared: Option<&mut SparedParts<'_>>,
        left: &mut LeftEntries,
    ) {
        while self.held_bytes > self.capacity_bytes && self.order.count_entries() > 1 {
            let oldest = self.order.find_oldest();
            let (slot, key) = oldest.expect("more than one entry is held");
            if let (
                EntryKey::Part {
                    table_id,
                    format_version,
                    key,
                },
                Some(spared),
            ) = (key, &mut spared)
                && spared.remove(*table_id, *format_version, key)
            {
                self.order.mark_used(slot);
                continue;
            }
            match self.release_slot(slot) {
                EntryKey::Part {
                    table_id,
                    format_version,
                    key,
                } => {
                    let held = self.detach_part(table_id, format_version, &key);
                    let held = held.expect("a part in the order is held under its key");
                    left.parts.push(held.value);
                }
                EntryKey::Footer {
                    table_id,
                    schema,
                    path,
                } => {
                    let held = self.detach_footer(table_id, &schema, &path);
                    let held = held.expect("a footer in the order is held by its table");
                    left.footers.push(held.row_groups);
                }
            }
        }
        self.compact_sparse_slots();
    }
}
