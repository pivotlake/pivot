//! What a plan reads of a table version's data manifests, through the parts
//! of them the cache holds ([`manifest_parts`](crate::files::manifest_parts)).
//!
//! A version is its manifest list, and a manifest never changes, so nothing
//! here is brought from one version to the next: a manifest another version
//! lists as well answers from the same parts.
//!
//! A plan reads three kinds of part, each found in the cache by a name of
//! its own: the partitions a manifest holds, the files of one partition in
//! one manifest, and what the manifest records of one column in those
//! files, of which a plan names the columns it reads. A part the cache does
//! not hold costs a parse of its manifest, which yields every part of the
//! manifest: those the plan asked for are stored as used now, and the
//! others as the first the cache lets go of, while it has room for them.
//!
//! A scan asks by partition, for the partitions its filters keep
//! ([`ManifestReader::load_slices`]); maintenance asks by manifest, for
//! everything a manifest lists ([`ManifestReader::load_manifests`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::file_metadata_cache::FileMetadataCache;
use object_storage::load_objects;

use super::{MANIFESTS_PER_READ, Table};
use crate::Result;
use crate::files::manifest_parts::{
    ColumnBounds, GroupedManifest, GroupedPartition, ManifestFiles, ManifestPartitions,
};
use crate::files::manifests::{EntryPartitions, ListedManifest};
use crate::files::partition::Partition;

/// What a part's name starts with, which tells the kinds of part apart.
const PARTITIONS_PART: u8 = 0;
const FILES_PART: u8 = 1;
const BOUNDS_PART: u8 = 2;

/// The name of the partitions the manifest at `manifest_path` holds.
fn name_partitions(manifest_path: &str) -> Arc<[u8]> {
    let mut name = Vec::with_capacity(1 + manifest_path.len());
    name.push(PARTITIONS_PART);
    name.extend_from_slice(manifest_path.as_bytes());
    Arc::from(name)
}

/// The start of the name of a part of the partition `partition_key`: the
/// part's kind, then the key behind its length, so no two keys run into what
/// follows them alike.
fn name_partition_part(kind: u8, partition_key: &[u8], rest_len: usize) -> Vec<u8> {
    let mut name = Vec::with_capacity(1 + size_of::<u32>() + partition_key.len() + rest_len);
    name.push(kind);
    name.extend_from_slice(&(partition_key.len() as u32).to_be_bytes());
    name.extend_from_slice(partition_key);
    name
}

/// The name of the files of the partition `partition_key` in the manifest
/// at `manifest_path`.
fn name_files(partition_key: &[u8], manifest_path: &str) -> Arc<[u8]> {
    let mut name = name_partition_part(FILES_PART, partition_key, manifest_path.len());
    name.extend_from_slice(manifest_path.as_bytes());
    Arc::from(name)
}

/// The name of what the manifest at `manifest_path` records of the field
/// `field_id` in the files of the partition `partition_key`.
fn name_bounds(partition_key: &[u8], manifest_path: &str, field_id: i32) -> Arc<[u8]> {
    let rest_len = size_of::<i32>() + manifest_path.len();
    let mut name = name_partition_part(BOUNDS_PART, partition_key, rest_len);
    name.extend_from_slice(&field_id.to_be_bytes());
    name.extend_from_slice(manifest_path.as_bytes());
    Arc::from(name)
}

/// The files of one partition in one manifest, with what a plan reads of
/// their columns.
pub(crate) struct PartitionSlice {
    pub(crate) partition: Arc<Partition>,
    pub(crate) files: Arc<ManifestFiles>,
    /// The ids of the fields whose bounds the plan reads, in order.
    field_ids: Arc<[i32]>,
    /// What the manifest records of each of those fields, in their order;
    /// `None` for one the manifest records nothing of in the files.
    bounds: Vec<Option<Arc<ColumnBounds>>>,
}

impl PartitionSlice {
    /// The files `files` of `partition` in one manifest, with `bounds`,
    /// what the manifest records of each of the fields `field_ids`, which
    /// are in order.
    pub(crate) fn new(
        partition: Arc<Partition>,
        files: Arc<ManifestFiles>,
        field_ids: Arc<[i32]>,
        bounds: Vec<Option<Arc<ColumnBounds>>>,
    ) -> Self {
        Self {
            partition,
            files,
            field_ids,
            bounds,
        }
    }

    /// What the manifest records of the field `field_id` in the files;
    /// `None` for a field the plan does not read the bounds of, and for one
    /// the manifest records nothing of.
    pub(crate) fn find_bounds(&self, field_id: i32) -> Option<&ColumnBounds> {
        let position = self.field_ids.binary_search(&field_id).ok()?;
        self.bounds[position].as_deref()
    }
}

/// One manifest asked for the files of one partition.
type AskedSlice<'a> = (&'a ListedManifest, &'a Arc<Partition>);

/// Whether a plan wants the files of a partition a manifest it parses
/// holds.
pub(super) type WantsPartition<'w> = dyn FnMut(&Partition) -> Result<bool> + 'w;

/// What one manifest answered of each partition it was asked for, by
/// partition key: `None` for a partition the manifest holds no file of.
type ManifestSlices = HashMap<Arc<[u8]>, Option<Arc<PartitionSlice>>>;

/// What one plan reads of a table version's data manifests, through the
/// cache: it holds every part it found or built until the plan is done,
/// whatever the cache keeps of them.
pub(crate) struct ManifestReader<'t> {
    table: &'t Table,
    cache: Arc<FileMetadataCache>,
    /// The ids of the fields whose bounds the plan reads, in order.
    field_ids: Arc<[i32]>,
    /// What each manifest answered of each partition it was asked for, by
    /// manifest path.
    slices: HashMap<Arc<str>, ManifestSlices>,
    /// The partitions each manifest the plan asked holds, by manifest
    /// path.
    partitions: HashMap<Arc<str>, Arc<ManifestPartitions>>,
    /// The names of the parts the plan read.
    pub(super) read_parts: Vec<Arc<[u8]>>,
}

impl<'t> ManifestReader<'t> {
    /// A reader for a plan over `table` that reads the bounds of the
    /// fields `field_ids`.
    pub(crate) fn new(table: &'t Table, mut field_ids: Vec<i32>) -> Self {
        field_ids.sort_unstable();
        field_ids.dedup();
        Self {
            table,
            cache: FileMetadataCache::process_wide(),
            field_ids: Arc::from(field_ids),
            slices: HashMap::new(),
            partitions: HashMap::new(),
            read_parts: Vec::new(),
        }
    }

    /// Whether the plan already knows what `manifest` holds of the
    /// partition `key`.
    fn knows_slice(&self, manifest: &ListedManifest, key: &[u8]) -> bool {
        self.slices
            .get(manifest.path())
            .is_some_and(|slices| slices.contains_key(key))
    }

    fn keep_slice(
        &mut self,
        manifest: &ListedManifest,
        key: &Arc<[u8]>,
        slice: Option<Arc<PartitionSlice>>,
    ) {
        self.slices
            .entry(Arc::clone(manifest.path()))
            .or_default()
            .insert(Arc::clone(key), slice);
    }

    /// Look in the cache for the partitions each of `manifests` holds, and
    /// hand back the manifests it does not hold them for, each once.
    fn find_partitions<'a>(&mut self, manifests: &[&'a ListedManifest]) -> Vec<&'a ListedManifest> {
        let mut asked_paths: HashSet<&str> = HashSet::new();
        let unknown: Vec<&ListedManifest> = manifests
            .iter()
            .copied()
            .filter(|manifest| !self.partitions.contains_key(manifest.path()))
            .filter(|manifest| asked_paths.insert(manifest.path()))
            .collect();
        let names: Vec<Arc<[u8]>> = unknown
            .iter()
            .map(|manifest| name_partitions(manifest.path()))
            .collect();
        let found = self.cache.find_metadata_parts::<ManifestPartitions>(
            &self.table.manifest_cache_key,
            names.iter().map(|name| &**name),
        );
        let mut missing = Vec::new();
        for ((manifest, name), found) in unknown.into_iter().zip(names).zip(found) {
            match found {
                Some(partitions) => {
                    self.partitions
                        .insert(Arc::clone(manifest.path()), partitions);
                    self.read_parts.push(name);
                }
                None => missing.push(manifest),
            }
        }
        missing
    }

    /// Look in the cache for what each of `asked` holds, the files of a
    /// partition in a manifest with the bounds the plan reads of them, and
    /// hand back those it cannot tell without parsing the manifest. A
    /// manifest the cache holds the partitions of answers for a partition
    /// it holds no file of.
    fn find_slices<'a>(&mut self, asked: &[AskedSlice<'a>]) -> Vec<AskedSlice<'a>> {
        let table = self.table;
        let cache_key = &table.manifest_cache_key;
        let unknown: Vec<AskedSlice<'a>> = asked
            .iter()
            .copied()
            .filter(|(manifest, partition)| !self.knows_slice(manifest, partition.key()))
            .collect();
        let file_names: Vec<Arc<[u8]>> = unknown
            .iter()
            .map(|(manifest, partition)| name_files(partition.key(), manifest.path()))
            .collect();
        let found_files = self
            .cache
            .find_metadata_parts::<ManifestFiles>(cache_key, file_names.iter().map(|name| &**name));

        // The bounds of the columns the plan reads, for the files found.
        let mut bound_names: Vec<Arc<[u8]>> = Vec::new();
        for ((manifest, partition), files) in unknown.iter().zip(&found_files) {
            let Some(files) = files else {
                continue;
            };
            for field_id in self.field_ids.iter() {
                if files.find_stats_field(*field_id).is_some() {
                    bound_names.push(name_bounds(partition.key(), manifest.path(), *field_id));
                }
            }
        }
        let mut found_bounds = self
            .cache
            .find_metadata_parts::<ColumnBounds>(cache_key, bound_names.iter().map(|name| &**name))
            .into_iter();
        let mut bound_names = bound_names.into_iter();

        let without_files: Vec<&ListedManifest> = unknown
            .iter()
            .zip(&found_files)
            .filter(|(_, files)| files.is_none())
            .map(|((manifest, _), _)| *manifest)
            .collect();
        self.find_partitions(&without_files);

        let mut unresolved = Vec::new();
        for (((manifest, partition), files_name), files) in
            unknown.into_iter().zip(file_names).zip(found_files)
        {
            let Some(files) = files else {
                let holds_none = self
                    .partitions
                    .get(manifest.path())
                    .is_some_and(|partitions| !partitions.holds(partition.key()));
                if holds_none {
                    self.keep_slice(manifest, partition.key(), None);
                } else {
                    unresolved.push((manifest, partition));
                }
                continue;
            };
            let mut bounds = Vec::with_capacity(self.field_ids.len());
            let mut is_every_column_found = true;
            for field_id in self.field_ids.iter() {
                if files.find_stats_field(*field_id).is_none() {
                    bounds.push(None);
                    continue;
                }
                let name = bound_names.next().expect("a name was built for the column");
                let found = found_bounds
                    .next()
                    .expect("the cache answers for every name");
                is_every_column_found &= found.is_some();
                if found.is_some() {
                    self.read_parts.push(name);
                }
                bounds.push(found);
            }
            if is_every_column_found {
                self.read_parts.push(files_name);
                let field_ids = Arc::clone(&self.field_ids);
                let slice = PartitionSlice::new(Arc::clone(partition), files, field_ids, bounds);
                self.keep_slice(manifest, partition.key(), Some(Arc::new(slice)));
            } else {
                unresolved.push((manifest, partition));
            }
        }
        unresolved
    }

    /// Read and parse `manifests`, a few at a time, each parsed and cut
    /// into its parts before the next is parsed. The partitions each holds
    /// are kept, and stored in the cache. Of the partitions `wants`
    /// accepts, asked once for each partition of a manifest, the files and
    /// the bounds of the columns the plan reads are kept, and stored as
    /// used now. Every other part of a manifest is stored as the
    /// first the cache lets go of, while the cache has room for it. The
    /// caller holds the table's build lock.
    fn parse_manifests(
        &mut self,
        manifests: &[&ListedManifest],
        wants: &mut WantsPartition<'_>,
    ) -> Result<()> {
        let table = self.table;
        for manifests in manifests.chunks(MANIFESTS_PER_READ) {
            let objects = manifests
                .iter()
                .map(|manifest| {
                    let file = manifest.file();
                    let length = file.manifest_length.max(0) as u64;
                    table.location.data_file(&file.manifest_path, length)
                })
                .collect::<Result<Vec<_>>>()?;
            let loaded = load_objects(&table.dispatcher, &objects)?;
            for (manifest, object) in manifests.iter().zip(loaded) {
                let parsed = table.parse_manifest(object.file.path.as_str(), &object.bytes)?;
                drop(object);
                let grouped = GroupedManifest::group(&table.name.to_string(), parsed)?;
                self.keep_parts(manifest, &grouped, wants)?;
            }
        }
        Ok(())
    }

    /// Keep what the plan reads of `grouped`, the parsed manifest `listed`
    /// names, and store every part of it in the cache: see
    /// [`parse_manifests`](Self::parse_manifests).
    fn keep_parts(
        &mut self,
        listed: &ListedManifest,
        grouped: &GroupedManifest,
        wants: &mut WantsPartition<'_>,
    ) -> Result<()> {
        let table = self.table;
        let cache_key = &table.manifest_cache_key;
        let manifest_path = listed.path();
        let partitions = Arc::new(grouped.list_partitions());
        let partitions_name = name_partitions(manifest_path);
        self.cache.store_metadata_parts(
            cache_key,
            vec![(Arc::clone(&partitions_name), Arc::clone(&partitions))],
        );
        self.read_parts.push(partitions_name);
        self.partitions
            .insert(Arc::clone(manifest_path), partitions);

        let build_files = |partition: &GroupedPartition| {
            grouped.build_files(partition, listed.file(), table.metadata(), &table.location)
        };
        let mut unwanted = Vec::new();
        for partition in grouped.partitions() {
            let key = partition.partition().key();
            if !wants(partition.partition())? {
                unwanted.push(partition);
                continue;
            }
            let files = Arc::new(build_files(partition)?);
            // Every column when the cache has room for one nobody asked
            // for, else those the plan reads.
            let built_field_ids: Vec<i32> = if self.cache.free_bytes() > 0 {
                files.stats_field_ids().to_vec()
            } else {
                let stats_field_ids = files.stats_field_ids().iter().copied();
                stats_field_ids
                    .filter(|field_id| self.field_ids.binary_search(field_id).is_ok())
                    .collect()
            };
            let mut bounds = vec![None; self.field_ids.len()];
            let mut read_bounds = Vec::new();
            let mut unread_bounds = Vec::new();
            let columns = partition.build_columns(&built_field_ids);
            for (field_id, column) in built_field_ids.into_iter().zip(columns) {
                let column = Arc::new(column);
                let name = name_bounds(key, manifest_path, field_id);
                match self.field_ids.binary_search(&field_id) {
                    Ok(position) => {
                        self.read_parts.push(Arc::clone(&name));
                        read_bounds.push((name, Arc::clone(&column)));
                        bounds[position] = Some(column);
                    }
                    Err(_) => unread_bounds.push((name, column)),
                }
            }
            let files_name = name_files(key, manifest_path);
            self.read_parts.push(Arc::clone(&files_name));
            self.cache
                .store_metadata_parts(cache_key, vec![(files_name, Arc::clone(&files))]);
            self.cache.store_metadata_parts(cache_key, read_bounds);
            self.cache
                .store_unasked_metadata_parts(cache_key, unread_bounds);
            let field_ids = Arc::clone(&self.field_ids);
            let slice =
                PartitionSlice::new(Arc::clone(partition.partition()), files, field_ids, bounds);
            self.keep_slice(listed, key, Some(Arc::new(slice)));
        }

        // The partitions nobody asked for, until the cache has no room for
        // one. What is wrong with an entry of one is reported to whoever
        // asks for the partition.
        for partition in unwanted {
            let key = partition.partition().key();
            if self.cache.free_bytes() == 0 {
                break;
            }
            let Ok(files) = build_files(partition) else {
                continue;
            };
            let columns = partition.build_columns(files.stats_field_ids());
            let bounds = files
                .stats_field_ids()
                .iter()
                .zip(columns)
                .map(|(field_id, column)| {
                    let name = name_bounds(key, manifest_path, *field_id);
                    (name, Arc::new(column))
                })
                .collect();
            let files = vec![(name_files(key, manifest_path), Arc::new(files))];
            let is_held = self.cache.store_unasked_metadata_parts(cache_key, files)
                && self.cache.store_unasked_metadata_parts(cache_key, bounds);
            if !is_held {
                break;
            }
        }
        Ok(())
    }

    /// The partitions each of `manifests` holds a live file of, in the
    /// order given, each manifest's in key order. A manifest whose
    /// manifest-list entry names its one partition, or counts no live file,
    /// is not asked. Of the others, one the cache does not hold the
    /// partitions of is read and parsed, under the table's build lock and
    /// once the cache was asked again, which keeps the files of the
    /// partitions `wants` accepts for the plan.
    fn read_partitions(
        &mut self,
        manifests: &[&ListedManifest],
        wants: &mut WantsPartition<'_>,
    ) -> Result<Vec<Vec<Arc<Partition>>>> {
        // The partitions each manifest holds, as its manifest-list entry
        // names them; `None` for a manifest only its own entries tell of.
        let mut unnamed: Vec<&ListedManifest> = Vec::new();
        let named: Vec<Option<Vec<Arc<Partition>>>> = manifests
            .iter()
            .map(|manifest| match manifest.partitions() {
                EntryPartitions::None => Some(Vec::new()),
                EntryPartitions::One(partition) => Some(vec![Arc::clone(partition)]),
                EntryPartitions::Unknown => {
                    unnamed.push(manifest);
                    None
                }
            })
            .collect();
        let missing = self.find_partitions(&unnamed);
        if !missing.is_empty() {
            let cache = Arc::clone(&self.cache);
            let _building = cache.lock_table(&self.table.manifest_cache_key);
            let missing = self.find_partitions(&missing);
            self.parse_manifests(&missing, wants)?;
        }
        Ok(manifests
            .iter()
            .zip(named)
            .map(|(manifest, named)| match named {
                Some(partitions) => partitions,
                None => self.partitions[manifest.path()].iter().cloned().collect(),
            })
            .collect())
    }

    /// The partitions `manifests` hold a live file of, each once, in key
    /// order ([`read_partitions`](Self::read_partitions)).
    pub(super) fn list_partitions(
        &mut self,
        manifests: &[&ListedManifest],
        wants: &mut WantsPartition<'_>,
    ) -> Result<Vec<Arc<Partition>>> {
        let partitions = self.read_partitions(manifests, wants)?;
        let mut partitions: Vec<Arc<Partition>> = partitions.into_iter().flatten().collect();
        partitions.sort_by(|left, right| left.key().cmp(right.key()));
        partitions.dedup_by(|later, earlier| later.key() == earlier.key());
        Ok(partitions)
    }

    /// Find out what each of `asked` holds, the files of a partition in a
    /// manifest with the bounds the plan reads of them. A manifest is read
    /// and parsed when the cache does not hold what the plan reads of it,
    /// once for all the partitions asked of it, and keeps for the plan the
    /// files of the partitions `wants` accepts.
    fn resolve_slices(
        &mut self,
        asked: &[AskedSlice<'_>],
        wants: &mut WantsPartition<'_>,
    ) -> Result<()> {
        let table = self.table;
        let unresolved = self.find_slices(asked);
        if unresolved.is_empty() {
            return Ok(());
        }
        let cache = Arc::clone(&self.cache);
        let _building = cache.lock_table(&table.manifest_cache_key);
        let unresolved = self.find_slices(&unresolved);
        let mut unparsed_paths: HashSet<&str> = HashSet::new();
        let unparsed: Vec<&ListedManifest> = unresolved
            .iter()
            .map(|(manifest, _)| *manifest)
            .filter(|manifest| unparsed_paths.insert(manifest.path()))
            .collect();
        self.parse_manifests(&unparsed, wants)?;
        // A manifest just parsed holds no file of a partition it kept
        // nothing for.
        for (manifest, partition) in unresolved {
            if !self.knows_slice(manifest, partition.key()) {
                self.keep_slice(manifest, partition.key(), None);
            }
        }
        Ok(())
    }

    /// The files of each of `partitions` with the bounds the plan reads of
    /// them, in the order given: a slice for each data manifest of the
    /// version that holds a file of the partition, in manifest-list order.
    /// A manifest is read and parsed when the cache does not hold what the
    /// plan reads of it, once for all the partitions.
    pub(crate) fn load_slices(
        &mut self,
        partitions: &[Arc<Partition>],
    ) -> Result<Vec<Vec<Arc<PartitionSlice>>>> {
        let listed = self.table.manifest_list.listed_data_manifests();
        // Each partition with the manifests whose manifest-list entries do
        // not rule it out, partition after partition, and how many of those
        // each partition has.
        let mut asked: Vec<AskedSlice<'_>> = Vec::new();
        let mut asked_per_partition = Vec::with_capacity(partitions.len());
        for partition in partitions {
            let asked_before = asked.len();
            let holders = listed.iter();
            let holders = holders.filter(|manifest| manifest.may_hold_partition(partition));
            asked.extend(holders.map(|manifest| (manifest, partition)));
            asked_per_partition.push(asked.len() - asked_before);
        }
        let wanted: HashSet<&[u8]> = partitions
            .iter()
            .map(|partition| partition.key().as_ref())
            .collect();
        self.resolve_slices(&asked, &mut |partition| {
            Ok(wanted.contains(partition.key().as_ref()))
        })?;
        let mut asked = asked.into_iter();
        Ok(asked_per_partition
            .into_iter()
            .map(|asked_of_partition| {
                let asked = asked.by_ref().take(asked_of_partition);
                asked
                    .filter_map(|(manifest, partition)| {
                        let slices = self.slices.get(manifest.path())?;
                        slices.get(partition.key())?.clone()
                    })
                    .collect()
            })
            .collect())
    }

    /// What each of `manifests` lists, in the order given: the files of
    /// every partition the manifest holds a live file of, in the order of
    /// the partitions' keys, with the bounds the plan reads of them. A manifest whose manifest-list entry counts no live file is
    /// not read; any other is read and parsed when the cache does not hold
    /// what the plan reads of it.
    pub(super) fn load_manifests(
        &mut self,
        manifests: &[&ListedManifest],
    ) -> Result<Vec<Vec<Arc<PartitionSlice>>>> {
        let held = self.read_partitions(manifests, &mut |_| Ok(true))?;
        let asked: Vec<AskedSlice<'_>> = manifests
            .iter()
            .zip(&held)
            .flat_map(|(manifest, partitions)| {
                partitions
                    .iter()
                    .map(move |partition| (*manifest, partition))
            })
            .collect();
        self.resolve_slices(&asked, &mut |_| Ok(true))?;
        Ok(manifests
            .iter()
            .zip(&held)
            .map(|(manifest, partitions)| {
                partitions
                    .iter()
                    .filter_map(|partition| {
                        let slices = self.slices.get(manifest.path())?;
                        slices.get(partition.key())?.clone()
                    })
                    .collect()
            })
            .collect())
    }
}
