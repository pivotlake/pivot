//! The files of a table as the compacter keeps them between its rounds: a
//! partition at a time, its small files and its large ones each in the
//! order a partition's files stand in, with what the rounds measured of the
//! large ones.
//!
//! The lists are the compacter's own. A table's format hands its files out
//! in groups that never change ([`FileGroupId`]): a manifest of an Iceberg
//! table, the whole of one version of a Delta table. The lists remember
//! which groups they hold, and are brought to a later version by the groups
//! the two differ in ([`TableFileLists::apply_file_groups`]): the files of the
//! groups that left go out, those of the groups that joined come in, and a
//! file that leaves with one group and comes back with another, as every
//! file a rewritten manifest carries over does, stays as it was, measured.
//! Nothing is read of a version whose groups the lists already hold, and
//! nothing of the versions between the one they hold and the one they are
//! brought to.
//!
//! A merge the compacter committed itself is applied without that
//! comparison when the table's format says how the commit's groups carry
//! forward ([`TableFileLists::carry_merge`]): the merge's inputs go out and
//! its outputs come in, under the groups of the version it committed.
//!
//! A partition left without a file is dropped there and then, with what was
//! measured of it. A partition a file joined or left is marked as changed
//! until a round finds nothing left to do in it, so a routine round
//! examines the partitions that changed and no other.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use object_storage::{FileRef, ObjectPath};

use super::overlap::{OverlapCache, RowGroupRanges, locate_record, read_row_group_ranges};
use super::sort_key::compare_file_order;
use super::{CompactionPass, is_small_file};
use crate::table_format::{CarriedFileGroups, CompactionFile, FileGroupId, PartitionGroupFiles};
use crate::{CatalogTable, Result};

/// A file as a partition's lists hold it: it stands by the lower bound of
/// the table's sort key in it, then by when it was committed, then by its
/// path, which only makes the order total.
struct FileInOrder(Arc<dyn CompactionFile>);

impl PartialEq for FileInOrder {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for FileInOrder {}

impl PartialOrd for FileInOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FileInOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_file_order(&*self.0, &*other.0)
    }
}

/// What the rounds keep of one partition of a table.
#[derive(Default)]
pub(super) struct PartitionFileLists {
    /// The files below half the target size, which a small-file merge
    /// takes from the start of.
    small: BTreeSet<FileInOrder>,
    /// The files at least half the target size, which a layout merge may
    /// rewrite.
    large: BTreeSet<FileInOrder>,
    /// What the rounds measured of the large files' layout, and which of
    /// them they are yet to measure.
    overlaps: OverlapCache,
    /// Whether a file joined or left since a round last found nothing to do
    /// in the partition.
    is_changed: bool,
}

impl PartitionFileLists {
    fn is_empty(&self) -> bool {
        self.small.is_empty() && self.large.is_empty()
    }

    /// The list `file` stands in, by its size against `target_bytes`.
    fn find_list(
        &mut self,
        file: &Arc<dyn CompactionFile>,
        target_bytes: u64,
    ) -> &mut BTreeSet<FileInOrder> {
        if is_small_file(file.file().size, target_bytes) {
            &mut self.small
        } else {
            &mut self.large
        }
    }

    /// List `files`, which joined the partition. The lists of a partition
    /// that held none are ordered in one go.
    fn extend(&mut self, files: impl Iterator<Item = Arc<dyn CompactionFile>>, target_bytes: u64) {
        let (small, large): (Vec<FileInOrder>, Vec<FileInOrder>) = files
            .map(FileInOrder)
            .partition(|FileInOrder(file)| is_small_file(file.file().size, target_bytes));
        if small.is_empty() && large.is_empty() {
            return;
        }
        self.is_changed = true;
        for FileInOrder(file) in &large {
            self.overlaps.add_unmeasured_file(file);
        }
        for (list, joining) in [(&mut self.small, small), (&mut self.large, large)] {
            if list.is_empty() {
                *list = BTreeSet::from_iter(joining);
            } else {
                list.extend(joining);
            }
        }
    }

    /// Take `file`, which left the partition, off its list, and forget what
    /// was measured of it. Hands back the record the list held of the file;
    /// `None` for a file it did not hold.
    fn remove(
        &mut self,
        file: &Arc<dyn CompactionFile>,
        target_bytes: u64,
    ) -> Option<Arc<dyn CompactionFile>> {
        self.is_changed = true;
        let list = self.find_list(file, target_bytes);
        let held = list.take(&FileInOrder(Arc::clone(file)));
        let held = held.map(|FileInOrder(held)| held);
        self.overlaps.forget_file(held.as_ref().unwrap_or(file));
        held
    }

    /// Hold `kept` in place of `listed`, another record of the same file,
    /// as the group that left held it: the file stays, with what was
    /// measured of it. A file the two records put in different lists, by
    /// the sizes they record, leaves the one and joins the other, as a
    /// file nothing was measured of.
    fn replace(
        &mut self,
        listed: &Arc<dyn CompactionFile>,
        kept: Arc<dyn CompactionFile>,
        target_bytes: u64,
    ) {
        let is_small =
            |file: &Arc<dyn CompactionFile>| is_small_file(file.file().size, target_bytes);
        if is_small(listed) != is_small(&kept) {
            self.remove(listed, target_bytes);
            self.extend(std::iter::once(kept), target_bytes);
            return;
        }
        self.find_list(listed, target_bytes)
            .remove(&FileInOrder(Arc::clone(listed)));
        self.overlaps.replace_record(listed, &kept);
        self.find_list(&kept, target_bytes)
            .insert(FileInOrder(kept));
    }

    /// The small files no merge in flight is rewriting, in order.
    pub(super) fn list_small_files<'f>(
        &'f self,
        reserved: &'f HashSet<ObjectPath>,
    ) -> impl Iterator<Item = &'f Arc<dyn CompactionFile>> + 'f {
        self.small
            .iter()
            .map(|FileInOrder(file)| file)
            .filter(|file| !reserved.contains(&file.file().path))
    }

    /// The files the next layout merge rewrites together, large files of
    /// this partition of `table`, as a round running `pass` selects them
    /// ([`OverlapCache::select_group`]); `None` when no merge should start,
    /// as in a table without a sort key. The large files that joined since
    /// the last selection are measured first, against the files they can
    /// interleave with ([`OverlapCache::observe`]).
    pub(super) fn select_layout_group(
        &mut self,
        table: &CatalogTable,
        reserved: &HashSet<ObjectPath>,
        pass: CompactionPass,
        clique_size: usize,
    ) -> Result<Option<Vec<Arc<dyn CompactionFile>>>> {
        let sort_by = table.sort_by();
        if sort_by.is_empty() {
            return Ok(None);
        }
        // The footers are read for the call alone: only the bounds of
        // their row groups on the sort columns leave it.
        let read_ranges = |files: &[FileRef]| -> Result<Vec<RowGroupRanges>> {
            let footers = table.row_groups(files)?;
            let footers = footers.iter();
            Ok(footers
                .map(|row_groups| read_row_group_ranges(row_groups, sort_by))
                .collect())
        };
        let large = self.large.iter().map(|FileInOrder(file)| file);
        self.overlaps.observe(large, &read_ranges, sort_by, pass)?;
        let Some(group) = self.overlaps.select_group(reserved, pass, clique_size) else {
            return Ok(None);
        };
        let files = group.iter();
        Ok(Some(
            files
                .map(|path| Arc::clone(self.overlaps.find_paired_file(path)))
                .collect(),
        ))
    }

    /// Record that a round found nothing left to do in the partition: what
    /// only a final sweep measured is let go of, and the partition counts
    /// as unchanged until a file joins or leaves it.
    pub(super) fn settle(&mut self) {
        self.overlaps.forget_pairs_below_bar();
        self.is_changed = false;
    }
}

/// What the rounds over one table keep of it from one to the next: its
/// files, a partition at a time, and the groups they came in.
#[derive(Default)]
pub(super) struct TableFileLists {
    /// The columns the table was sorted by when the files were listed: the
    /// order they stand in is that of their bounds on these.
    sort_by: Vec<String>,
    /// The groups whose files the lists hold, each with the files it
    /// brought, as the lists hold them.
    groups: HashMap<FileGroupId, Vec<PartitionGroupFiles>>,
    /// The files of each partition that has one, by the partition's key.
    partitions: BTreeMap<Vec<u8>, PartitionFileLists>,
}

impl TableFileLists {
    /// Bring the lists to the version `table` is at, by the groups of files
    /// that version has and the lists do not hold, which are loaded, and
    /// the groups the lists hold and that version does not have. A file a
    /// group that left listed and a group that joined lists in the same
    /// partition stays where it stands: told by its record when both
    /// groups hold the same one, else by its path. Nothing is read when the
    /// version's groups are the ones held. `target_bytes` is the compaction
    /// target, half of which tells a small file from a large one.
    ///
    /// A table sorted by other columns than the lists were built under has
    /// its files listed anew, in the order of the new key.
    pub(super) fn apply_file_groups(
        &mut self,
        table: &CatalogTable,
        target_bytes: u64,
    ) -> Result<()> {
        if self.sort_by != table.sort_by() {
            *self = Self {
                sort_by: table.sort_by().to_vec(),
                ..Self::default()
            };
        }
        let listed = table.list_file_groups();
        let joined: Vec<FileGroupId> = listed
            .iter()
            .filter(|group| !self.groups.contains_key(*group))
            .cloned()
            .collect();
        let listed: HashSet<&FileGroupId> = listed.iter().collect();
        let has_group_left = self.groups.keys().any(|group| !listed.contains(group));
        if joined.is_empty() && !has_group_left {
            return Ok(());
        }
        let loaded = table.load_file_groups(&joined)?;
        let left = self.groups.extract_if(|group, _| !listed.contains(group));
        let left: Vec<PartitionGroupFiles> = left.flat_map(|(_, files)| files).collect();

        // The records of the groups that joined whose files stay listed as
        // they stand: first those a group that left holds as well.
        let left_records: HashSet<usize> = left
            .iter()
            .flat_map(|partition_files| &partition_files.files)
            .map(locate_record)
            .collect();
        let mut staying: HashSet<usize> = HashSet::new();
        // Every other file a group that joined lists, by its path, for the
        // files of the groups that left to be looked up among.
        let mut brought: HashMap<&ObjectPath, BroughtFile<'_>> = HashMap::new();
        for partition_files in loaded.iter().flatten() {
            for file in &partition_files.files {
                if left_records.contains(&locate_record(file)) {
                    staying.insert(locate_record(file));
                } else if !left.is_empty() {
                    let brought_file = BroughtFile {
                        partition: &partition_files.partition,
                        file,
                    };
                    brought.insert(&file.file().path, brought_file);
                }
            }
        }
        for partition_files in &left {
            // A partition stays for as long as a file of it is listed, so
            // only a file two groups listed at once, gone with the first of
            // them to leave, has a group name a partition that is not here.
            let Some(lists) = self.partitions.get_mut(&partition_files.partition) else {
                continue;
            };
            for file in &partition_files.files {
                if staying.contains(&locate_record(file)) {
                    continue;
                }
                match brought.get(&file.file().path) {
                    Some(kept) if kept.partition == partition_files.partition => {
                        lists.replace(file, Arc::clone(kept.file), target_bytes);
                        staying.insert(locate_record(kept.file));
                    }
                    _ => {
                        lists.remove(file, target_bytes);
                    }
                }
            }
        }
        for partition_files in loaded.iter().flatten() {
            let mut joining = partition_files
                .files
                .iter()
                .filter(|file| !staying.contains(&locate_record(file)))
                .cloned()
                .peekable();
            if joining.peek().is_some() {
                let lists = self.partitions.entry(partition_files.partition.clone());
                lists.or_default().extend(joining, target_bytes);
            }
        }
        self.partitions.retain(|_, lists| !lists.is_empty());
        self.groups.extend(joined.into_iter().zip(loaded));
        Ok(())
    }

    /// Bring the lists to the version a merge the compacter committed
    /// made, by what the merge changed alone: `inputs`, files of
    /// `partition`, leave their lists and the groups that listed them, and
    /// the files the commit added join, each as the committed version holds
    /// it, under the groups `carried` names. Of every other file, only the
    /// places the replaced groups hold their records at are passed over.
    ///
    /// Nothing is done unless the lists hold every group the commit
    /// replaced and none it wrote: they then stand at another version than
    /// the one the commit was made on, and are brought to a later one by
    /// its groups ([`apply_file_groups`](Self::apply_file_groups)).
    ///
    /// An input the lists do not hold shows them out of step with the
    /// table: its path is handed back as the error, and the lists are no
    /// longer to be used.
    pub(super) fn carry_merge(
        &mut self,
        partition: &[u8],
        inputs: &[Arc<dyn CompactionFile>],
        carried: CarriedFileGroups,
        target_bytes: u64,
    ) -> std::result::Result<(), ObjectPath> {
        let stands_where_committed = carried.replaced.iter().all(|(earlier, committed)| {
            self.groups.contains_key(earlier) && !self.groups.contains_key(committed)
        });
        if !stands_where_committed {
            return Ok(());
        }
        // Where the lists held the record of each input: what the groups
        // that listed them tell their records by.
        let mut left: Vec<usize> = Vec::with_capacity(inputs.len());
        let mut lists = self.partitions.get_mut(partition);
        for input in inputs {
            let lists = lists.as_deref_mut();
            match lists.and_then(|lists| lists.remove(input, target_bytes)) {
                Some(held) => left.push(locate_record(&held)),
                None => return Err(input.file().path.clone()),
            }
        }
        left.sort_unstable();
        for (earlier, committed) in carried.replaced {
            let listed = self.groups.remove(&earlier);
            let mut listed = listed.expect("the lists hold every group the commit replaced");
            for partition_files in &mut listed {
                if partition_files.partition == partition {
                    let files = &mut partition_files.files;
                    files.retain(|file| left.binary_search(&locate_record(file)).is_err());
                }
            }
            self.groups.insert(committed, listed);
        }
        let added_group = self.groups.entry(carried.added_in).or_default();
        for added in carried.added {
            let lists = self.partitions.entry(added.partition.clone()).or_default();
            lists.extend(added.files.iter().cloned(), target_bytes);
            let listed = added_group
                .iter_mut()
                .find(|listed| listed.partition == added.partition);
            match listed {
                Some(listed) => {
                    listed.files.reserve_exact(added.files.len());
                    listed.files.extend(added.files);
                }
                None => added_group.push(added),
            }
        }
        let is_left_empty = self
            .partitions
            .get(partition)
            .is_some_and(PartitionFileLists::is_empty);
        if is_left_empty {
            self.partitions.remove(partition);
        }
        Ok(())
    }

    /// The partitions a round running `pass` over `table` examines, by
    /// their keys, in the order of those keys: those a file joined or left
    /// since a round last found nothing to do in them, for a routine
    /// round; every partition, for a final one. The lists stand at the
    /// version `table` is at. A partition whose files the table's format
    /// does not rewrite is left out, and counts as unchanged until a file
    /// joins or leaves it.
    pub(super) fn list_round_partitions(
        &mut self,
        table: &CatalogTable,
        pass: CompactionPass,
    ) -> Result<Vec<Vec<u8>>> {
        let is_examined =
            |lists: &PartitionFileLists| pass == CompactionPass::Final || lists.is_changed;
        let examined = self.partitions.iter();
        let examined: Vec<Vec<u8>> = examined
            .filter(|(_, lists)| is_examined(lists))
            .map(|(partition, _)| partition.clone())
            .collect();
        let rewritable = table.select_rewritable_partitions(examined)?;
        // The format hands back those it rewrites in the order given, which
        // tells the ones it left out.
        let mut kept = rewritable.iter().peekable();
        let examined = self.partitions.iter_mut();
        for (partition, lists) in examined.filter(|(_, lists)| is_examined(lists)) {
            if kept.next_if_eq(&partition).is_none() {
                lists.is_changed = false;
            }
        }
        Ok(rewritable)
    }

    /// The files of `partition`; `None` for a partition the table has no
    /// file in.
    pub(super) fn find_partition(&mut self, partition: &[u8]) -> Option<&mut PartitionFileLists> {
        self.partitions.get_mut(partition)
    }
}

/// A file a group that joined a table's lists brought.
struct BroughtFile<'g> {
    /// The key of the partition the group lists the file in.
    partition: &'g [u8],
    file: &'g Arc<dyn CompactionFile>,
}
