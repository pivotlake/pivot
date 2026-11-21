//! The files of a table as the compacter keeps them between its rounds: a
//! partition at a time, its small files and its large ones each in the
//! order a partition's files stand in, with what the rounds measured of the
//! large ones.
//!
//! The lists are the compacter's own. A table's format hands its files out
//! in manifests that never change ([`ManifestId`]): a manifest of an
//! Iceberg table, the whole of one version of a Delta table. The lists
//! remember which manifests they hold, and are brought to a later version
//! by the manifests the two differ in
//! ([`TableFileLists::apply_manifests`]): the files of the manifests
//! that left go out, those of the manifests that joined come in, and a
//! file that leaves with one manifest and comes back with another, as
//! every file a rewritten Iceberg manifest carries over does, stays as it
//! was, measured. Nothing is read of a version whose manifests the lists
//! already hold, and nothing of the versions between the one they hold and
//! the one they are brought to.
//!
//! A file is told by its path, which no other file of its table has: the
//! file a manifest that left and a manifest that joined both list, as the
//! versions of a Delta table do every file a commit left in place, is the
//! one the two list at one path in one partition.
//!
//! A partition left without a file is dropped there and then, with what was
//! measured of it. A partition is settled once a round finds nothing left to
//! do in it, and stays so until a file joins or leaves it, so a routine
//! round examines the partitions that are not settled and no other.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use object_storage::ObjectPath;

use super::overlap::OverlapCache;
use super::sort_key::compare_file_order;
use super::{CompactionPass, is_small_file};
use crate::table_format::{CompactionFile, ManifestId, PartitionFiles, PartitionHandle};
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
    /// takes a run of neighbours of.
    small: BTreeSet<FileInOrder>,
    /// The files at least half the target size, which a layout merge may
    /// rewrite.
    large: BTreeSet<FileInOrder>,
    /// What the rounds measured of the large files' layout, and which of
    /// them they are yet to measure.
    overlaps: OverlapCache,
    /// Whether a round found nothing left to do in the partition and no
    /// file has joined or left it since.
    is_settled: bool,
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
        self.is_settled = false;
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
    /// was measured of it.
    fn remove(&mut self, file: &Arc<dyn CompactionFile>, target_bytes: u64) {
        self.is_settled = false;
        let list = self.find_list(file, target_bytes);
        list.remove(&FileInOrder(Arc::clone(file)));
        self.overlaps.forget_file(file);
    }

    /// Hold `kept` in place of `listed`, another record of the same file,
    /// as the manifest that left held it: the file stays, with what was
    /// measured of it, and the partition is as settled as it was. A file
    /// the two records put in different lists, by the sizes they record,
    /// leaves the one and joins the other, as a file nothing was measured
    /// of.
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
        self.overlaps.replace_record(&kept);
        let list = self.find_list(&kept, target_bytes);
        // Two records that agree on the file's bounds and commit stand at
        // one place, where the one takes the other's; else `listed` is
        // still at a place of its own.
        if list.replace(FileInOrder(kept)).is_none() {
            list.remove(&FileInOrder(Arc::clone(listed)));
        }
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
        if table.sort_by().is_empty() {
            return Ok(None);
        }
        let large = self.large.iter().map(|FileInOrder(file)| file);
        self.overlaps.observe(large, table, pass)?;
        Ok(self.overlaps.select_group(reserved, pass, clique_size))
    }

    /// Record that a round found nothing left to do in the partition: what
    /// only a final sweep measured is let go of, and the partition is
    /// settled until a file joins or leaves it.
    pub(super) fn settle(&mut self) {
        self.overlaps.forget_pairs_below_bar();
        self.is_settled = true;
    }
}

/// The compacter's record of one table, kept between rounds: the table's
/// files by partition, and the manifests they were loaded from.
#[derive(Default)]
pub(super) struct TableFileLists {
    /// The columns the table was sorted by when the files were listed: the
    /// order they stand in is that of their bounds on these.
    sort_by: Vec<String>,
    /// The manifests whose files the lists hold, each with the files it
    /// brought, as the lists hold them.
    manifests: HashMap<ManifestId, Vec<PartitionFiles>>,
    /// The files of each partition that has one, in the order the table's
    /// format gives its partitions.
    partitions: BTreeMap<PartitionHandle, PartitionFileLists>,
}

impl TableFileLists {
    /// Bring the lists to the version `table` is at, by the manifests that
    /// version has and the lists do not hold, which are loaded, and the
    /// manifests the lists hold and that version does not have. A file a
    /// manifest that left listed and a manifest that joined lists in the
    /// same partition, told by its path, stays where it stands. Nothing is
    /// read when the version's manifests are the ones held. `target_bytes`
    /// is the compaction target, half of which tells a small file from a
    /// large one.
    ///
    /// A table sorted by other columns than the lists were built under has
    /// its files listed anew, in the order of the new key.
    pub(super) fn apply_manifests(
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
        let listed = table.list_manifest_ids();
        let joined: Vec<ManifestId> = listed
            .iter()
            .filter(|manifest| !self.manifests.contains_key(*manifest))
            .cloned()
            .collect();
        let listed: HashSet<&ManifestId> = listed.iter().collect();
        let has_manifest_left = self
            .manifests
            .keys()
            .any(|manifest| !listed.contains(manifest));
        if joined.is_empty() && !has_manifest_left {
            return Ok(());
        }
        let loaded = table.load_manifest_files(&joined)?;
        let left = self
            .manifests
            .extract_if(|manifest, _| !listed.contains(manifest));
        let left: Vec<PartitionFiles> = left.flat_map(|(_, files)| files).collect();

        // Every file a manifest that joined lists, by its path, for the
        // files of the manifests that left to be looked up among.
        let mut brought: HashMap<&ObjectPath, BroughtFile<'_>> = HashMap::new();
        if !left.is_empty() {
            for partition_files in loaded.iter().flatten() {
                for file in &partition_files.files {
                    let brought_file = BroughtFile {
                        partition: &partition_files.partition,
                        file,
                    };
                    brought.insert(&file.file().path, brought_file);
                }
            }
        }
        // The files that stay listed as they stand, by their paths.
        let mut staying: HashSet<&ObjectPath> = HashSet::new();
        for partition_files in &left {
            // A partition stays for as long as a file of it is listed, so
            // only a file two manifests listed at once, gone with the first
            // of them to leave, has a manifest name a partition that is not
            // here.
            let Some(lists) = self.partitions.get_mut(&partition_files.partition) else {
                continue;
            };
            for file in &partition_files.files {
                let path = &file.file().path;
                if staying.contains(path) {
                    continue;
                }
                match brought.get(path) {
                    Some(kept) if kept.partition == &partition_files.partition => {
                        lists.replace(file, Arc::clone(kept.file), target_bytes);
                        staying.insert(path);
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
                .filter(|file| !staying.contains(&file.file().path))
                .cloned()
                .peekable();
            if joining.peek().is_some() {
                let lists = self.partitions.entry(partition_files.partition.clone());
                lists.or_default().extend(joining, target_bytes);
            }
        }
        self.partitions.retain(|_, lists| !lists.is_empty());
        self.manifests.extend(joined.into_iter().zip(loaded));
        Ok(())
    }

    /// The partitions a round running `pass` examines, in the order of
    /// their keys: those that are not settled, for a routine round; every
    /// partition, for a final one.
    pub(super) fn list_round_partitions(&self, pass: CompactionPass) -> Vec<PartitionHandle> {
        let examined = self
            .partitions
            .iter()
            .filter(|(_, lists)| pass == CompactionPass::Final || !lists.is_settled);
        examined.map(|(partition, _)| partition.clone()).collect()
    }

    /// The files of `partition`; `None` for a partition the table has no
    /// file in.
    pub(super) fn find_partition(
        &mut self,
        partition: &PartitionHandle,
    ) -> Option<&mut PartitionFileLists> {
        self.partitions.get_mut(partition)
    }
}

/// A file a manifest that joined a table's lists brought.
struct BroughtFile<'m> {
    /// The partition the manifest lists the file in.
    partition: &'m PartitionHandle,
    file: &'m Arc<dyn CompactionFile>,
}
