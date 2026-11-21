//! What a plan reads of a table version's data manifests, through the parts
//! of them the cache holds ([`manifest_parts`](crate::files::manifest_parts)).
//!
//! A version is its manifest list, and a manifest never changes, so nothing
//! here is brought from one version to the next: a manifest another version
//! lists as well answers from the same parts.
//!
//! A plan reads three kinds of part, each asked of the cache as what it
//! is: the partitions a manifest holds, by the manifest's path; the files
//! of one partition in one manifest, by the path and the partition;
//! and what the manifest records of one column in those files, by the
//! path, the partition and the column's field id, of which a plan names the
//! columns it reads. A part the cache does not hold costs a parse of its
//! manifest, which yields every part of the manifest: those the plan asked
//! for are stored as used now, and the others as the first the cache lets
//! go of, while it has room for them. A plan remembers where each part it
//! read is ([`CachedPartKey`]), for the cache to keep those parts
//! while the plan's footers are stored and to count them as used after.
//!
//! A plan finds its files in two steps. It walks the manifests it looks at
//! once and matches each, by the partitions it holds, with the partitions
//! the plan wants ([`ManifestReader::match_partitions`]); it then reads the
//! files and the bounds of the pairs that matched, and of no other
//! ([`ManifestReader::load_slices`]). Maintenance and a description ask
//! by manifest instead, for everything a manifest lists
//! ([`ManifestReader::load_manifests`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::table_metadata_cache::{CachedPartKey, TableMetadataCache};
use object_storage::{FileRef, ObjectPath, load_objects};

use super::{MANIFESTS_PER_READ, Table};
use crate::Result;
use crate::files::manifest_list::ListedManifest;
use crate::files::manifest_parts::{
    CachedColumnBounds, CachedPartitionFiles, GroupedManifest, ManifestPartitions, PartitionEntries,
};
use crate::files::partition::Partition;
use crate::maintenance::compaction_input::{CompactionInput, SortColumns};
use parquet_engine::BoundSide;

/// The files of one partition in one manifest, with what a plan reads of
/// their columns.
pub(crate) struct PartitionSlice {
    pub(crate) partition: Arc<Partition>,
    /// The path the manifest list gives the manifest.
    pub(crate) manifest_path: Arc<str>,
    pub(crate) files: Arc<CachedPartitionFiles>,
    /// The ids of the fields whose bounds the plan reads, in order.
    field_ids: Arc<[i32]>,
    /// What the manifest records of each of those fields, in their order;
    /// `None` for one the manifest records nothing of in the files.
    bounds: Vec<Option<Arc<CachedColumnBounds>>>,
}

impl PartitionSlice {
    /// The files `files` of `partition` in the manifest at `manifest_path`,
    /// with `bounds`, what the manifest records of each of the fields
    /// `field_ids`, which are in order.
    pub(crate) fn new(
        partition: Arc<Partition>,
        manifest_path: Arc<str>,
        files: Arc<CachedPartitionFiles>,
        field_ids: Arc<[i32]>,
        bounds: Vec<Option<Arc<CachedColumnBounds>>>,
    ) -> Self {
        Self {
            partition,
            manifest_path,
            files,
            field_ids,
            bounds,
        }
    }

    /// What the manifest records of the field `field_id` in the files;
    /// `None` for a field the plan does not read the bounds of, and for one
    /// the manifest records nothing of.
    pub(crate) fn find_bounds(&self, field_id: i32) -> Option<&CachedColumnBounds> {
        let position = self.field_ids.binary_search(&field_id).ok()?;
        self.bounds[position].as_deref()
    }

    /// The files, in the manifest's order, each as the record maintenance
    /// takes of it: with its bounds on the columns of `sort_columns`,
    /// whose bounds the slice was read with. A column the manifest records
    /// nothing of, or that the files' schema does not have as a primitive,
    /// bounds no file.
    pub(crate) fn build_compaction_inputs(
        &self,
        sort_columns: &Arc<SortColumns>,
    ) -> Vec<CompactionInput> {
        let files = &self.files;
        let sort_bounds: Vec<Option<&CachedColumnBounds>> = sort_columns
            .field_ids()
            .enumerate()
            .map(|(position, field_id)| {
                let bounds = self.find_bounds(field_id);
                bounds.filter(|_| sort_columns.is_bounded_under(position, files.schema_id()))
            })
            .collect();
        let positions = 0..files.count_files();
        positions
            .map(|position| {
                let file = FileRef {
                    path: ObjectPath::new(files.path(position)),
                    size: files.size(position),
                };
                let bounds = sort_bounds.iter().flat_map(|bounds| {
                    [BoundSide::Lower, BoundSide::Upper]
                        .map(|side| bounds.and_then(|bounds| bounds.read_literal(side, position)))
                });
                CompactionInput {
                    file,
                    row_count: files.record_count(position),
                    commit_sequence: files.commit_sequence(position),
                    bounds: bounds.collect(),
                    sort_columns: Arc::clone(sort_columns),
                }
            })
            .collect()
    }
}

/// One manifest asked for the files of one partition.
type AskedSlice<'a> = (&'a ListedManifest, &'a Arc<Partition>);

/// Which partitions a plan wants the files of.
pub(crate) enum WantedPartitions<'w> {
    /// These few, in their order and each once: the one a merge or a commit
    /// works in. No manifest is parsed to learn whether it holds one of
    /// them.
    Listed(&'w [Arc<Partition>]),
    /// Those the check keeps, which is asked once for each partition.
    Matching(&'w mut PartitionCheck<'w>),
}

/// A manifest with a partition it holds a live file of.
type MatchedPair<'m> = (&'m ListedManifest, Arc<Partition>);

/// A partition a plan wants, with the manifests that hold a live file of
/// it.
pub(crate) struct PartitionManifests<'m> {
    pub(crate) partition: Arc<Partition>,
    manifests: Vec<&'m ListedManifest>,
}

/// Whether a plan wants the files of a partition a manifest it parses
/// holds.
pub(super) type PartitionCheck<'w> = dyn FnMut(&Arc<Partition>) -> Result<bool> + 'w;

/// What one manifest answered of each partition it was asked for: `None`
/// for a partition the manifest holds no file of.
type ManifestSlices = HashMap<Arc<Partition>, Option<Arc<PartitionSlice>>>;

/// What one plan reads of a table version's data manifests, through the
/// cache: it holds every part it found or built until the plan is done,
/// whatever the cache keeps of them.
pub(crate) struct ManifestReader<'t> {
    table: &'t Table,
    cache: Arc<TableMetadataCache>,
    /// The ids of the fields whose bounds the plan reads, in order.
    field_ids: Arc<[i32]>,
    /// What each manifest answered of each partition it was asked for, by
    /// manifest path.
    slices: HashMap<Arc<str>, ManifestSlices>,
    /// The partitions each manifest the plan asked holds, by manifest
    /// path.
    partitions: HashMap<Arc<str>, Arc<ManifestPartitions>>,
    /// Where each part the plan read is.
    pub(super) read_parts: Vec<CachedPartKey>,
}

impl<'t> ManifestReader<'t> {
    /// A reader for a plan over `table` that reads the bounds of the
    /// fields `field_ids`.
    pub(crate) fn new(table: &'t Table, mut field_ids: Vec<i32>) -> Self {
        field_ids.sort_unstable();
        field_ids.dedup();
        Self {
            table,
            cache: Arc::clone(&table.cache_slot.cache),
            field_ids: Arc::from(field_ids),
            slices: HashMap::new(),
            partitions: HashMap::new(),
            read_parts: Vec::new(),
        }
    }

    /// Whether the plan already knows what `manifest` holds of the
    /// `partition`.
    fn knows_slice(&self, manifest: &ListedManifest, partition: &Partition) -> bool {
        self.slices
            .get(manifest.path())
            .is_some_and(|slices| slices.contains_key(partition))
    }

    fn keep_slice(
        &mut self,
        manifest: &ListedManifest,
        partition: &Arc<Partition>,
        slice: Option<Arc<PartitionSlice>>,
    ) {
        self.slices
            .entry(Arc::clone(manifest.path()))
            .or_default()
            .insert(Arc::clone(partition), slice);
    }

    /// Learn the partitions each of `manifests` holds as far as that takes
    /// no manifest read, and hand back the manifests left unknown, each
    /// once. A manifest whose manifest-list entry tells its partitions
    /// ([`ListedManifest::find_listed_partitions`]) has them from the
    /// entry, kept for the plan and not stored in the cache; any other is
    /// looked up in the cache.
    fn find_partitions<'a>(&mut self, manifests: &[&'a ListedManifest]) -> Vec<&'a ListedManifest> {
        let table = self.table;
        let mut asked_paths: HashSet<&str> = HashSet::new();
        let mut unlisted: Vec<&ListedManifest> = Vec::new();
        for manifest in manifests.iter().copied() {
            let path = manifest.path();
            if self.partitions.contains_key(path) || !asked_paths.insert(path) {
                continue;
            }
            match manifest.find_listed_partitions(table.metadata()) {
                Some(partitions) => {
                    self.partitions
                        .insert(Arc::clone(path), Arc::new(partitions));
                }
                None => unlisted.push(manifest),
            }
        }
        let found = self.cache.find_manifest_partitions(
            table.cache_slot.table_id,
            table.format_version(),
            unlisted.iter().map(|manifest| manifest.path()),
        );
        let mut missing = Vec::new();
        for (manifest, found) in unlisted.into_iter().zip(found) {
            match found {
                Some(partitions) => {
                    self.partitions
                        .insert(Arc::clone(manifest.path()), partitions);
                    let key = CachedPartKey::of_partitions(manifest.path());
                    self.read_parts.push(key);
                }
                None => missing.push(manifest),
            }
        }
        missing
    }

    /// Look in the cache for what each of `asked` holds, the files of a
    /// partition in a manifest with the bounds the plan reads of them, and
    /// hand back those it cannot tell without parsing the manifest. A
    /// manifest whose partitions are found
    /// ([`find_partitions`](Self::find_partitions)) answers for a
    /// partition it holds no file of.
    fn find_slices<'a>(&mut self, asked: &[AskedSlice<'a>]) -> Vec<AskedSlice<'a>> {
        let table = self.table;
        let (table_id, format_version) = (table.cache_slot.table_id, table.format_version());
        let unknown: Vec<AskedSlice<'a>> = asked
            .iter()
            .copied()
            .filter(|(manifest, partition)| !self.knows_slice(manifest, partition))
            .collect();
        let asked_files = unknown.iter();
        let asked_files = asked_files.map(|(manifest, partition)| (manifest.path(), *partition));
        let found_files = self
            .cache
            .find_manifest_files(table_id, format_version, asked_files);

        // The bounds of the columns the plan reads, for the files found.
        let mut asked_bounds: Vec<(&Arc<str>, &Arc<Partition>, i32)> = Vec::new();
        for ((manifest, partition), files) in unknown.iter().zip(&found_files) {
            let Some(files) = files else {
                continue;
            };
            for field_id in self.field_ids.iter() {
                if files.find_stats_field(*field_id).is_some() {
                    asked_bounds.push((manifest.path(), partition, *field_id));
                }
            }
        }
        let mut found_bounds = self
            .cache
            .find_column_bounds(table_id, format_version, asked_bounds.into_iter())
            .into_iter();

        let without_files: Vec<&ListedManifest> = unknown
            .iter()
            .zip(&found_files)
            .filter(|(_, files)| files.is_none())
            .map(|((manifest, _), _)| *manifest)
            .collect();
        self.find_partitions(&without_files);

        let mut unresolved = Vec::new();
        for ((manifest, partition), files) in unknown.into_iter().zip(found_files) {
            let Some(files) = files else {
                let holds_none = self
                    .partitions
                    .get(manifest.path())
                    .is_some_and(|partitions| !partitions.holds(partition));
                if holds_none {
                    self.keep_slice(manifest, partition, None);
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
                let found = found_bounds
                    .next()
                    .expect("the cache answers for every column asked of it");
                is_every_column_found &= found.is_some();
                if found.is_some() {
                    self.read_parts.push(CachedPartKey::of_bounds(
                        manifest.path(),
                        partition,
                        *field_id,
                    ));
                }
                bounds.push(found);
            }
            if is_every_column_found {
                let files_key = CachedPartKey::of_files(manifest.path(), partition);
                self.read_parts.push(files_key);
                let field_ids = Arc::clone(&self.field_ids);
                let manifest_path = Arc::clone(manifest.path());
                let slice = PartitionSlice::new(
                    Arc::clone(partition),
                    manifest_path,
                    files,
                    field_ids,
                    bounds,
                );
                self.keep_slice(manifest, partition, Some(Arc::new(slice)));
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
    /// first the cache lets go of, while the cache has room for it.
    fn parse_manifests(
        &mut self,
        manifests: &[&ListedManifest],
        wants: &mut PartitionCheck<'_>,
    ) -> Result<()> {
        let table = self.table;
        for manifests in manifests.chunks(MANIFESTS_PER_READ) {
            let objects = manifests
                .iter()
                .map(|manifest| {
                    let file = manifest.file();
                    let length = file.manifest_length.max(0) as u64;
                    table
                        .location
                        .locate_file_at_uri(&file.manifest_path, length)
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
        wants: &mut PartitionCheck<'_>,
    ) -> Result<()> {
        let table = self.table;
        let (table_id, format_version) = (table.cache_slot.table_id, table.format_version());
        let manifest_path = listed.path();
        let partitions = Arc::new(grouped.list_partitions());
        self.cache.store_manifest_partitions(
            table_id,
            format_version,
            manifest_path,
            Arc::clone(&partitions),
        );
        self.read_parts
            .push(CachedPartKey::of_partitions(manifest_path));
        self.partitions
            .insert(Arc::clone(manifest_path), partitions);

        let build_files = |partition: &PartitionEntries| {
            partition.build_files(listed.file(), table.metadata(), &table.location)
        };
        let mut unwanted = Vec::new();
        for entries in grouped.partitions() {
            let partition = entries.partition();
            if !wants(partition)? {
                unwanted.push(entries);
                continue;
            }
            let files = Arc::new(build_files(entries)?);
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
            let columns = entries.build_columns(&built_field_ids);
            for (field_id, column) in built_field_ids.into_iter().zip(columns) {
                let column = Arc::new(column);
                match self.field_ids.binary_search(&field_id) {
                    Ok(position) => {
                        self.read_parts.push(CachedPartKey::of_bounds(
                            manifest_path,
                            partition,
                            field_id,
                        ));
                        read_bounds.push((field_id, Arc::clone(&column)));
                        bounds[position] = Some(column);
                    }
                    Err(_) => unread_bounds.push((field_id, column)),
                }
            }
            self.read_parts
                .push(CachedPartKey::of_files(manifest_path, partition));
            let cache = &self.cache;
            cache.store_manifest_files(
                table_id,
                format_version,
                manifest_path,
                partition,
                Arc::clone(&files),
            );
            cache.store_column_bounds(
                table_id,
                format_version,
                manifest_path,
                partition,
                read_bounds,
            );
            cache.store_unasked_column_bounds(
                table_id,
                format_version,
                manifest_path,
                partition,
                unread_bounds,
            );
            let field_ids = Arc::clone(&self.field_ids);
            let manifest_path = Arc::clone(manifest_path);
            let slice = PartitionSlice::new(
                Arc::clone(partition),
                manifest_path,
                files,
                field_ids,
                bounds,
            );
            self.keep_slice(listed, partition, Some(Arc::new(slice)));
        }

        // The partitions nobody asked for, until the cache has no room for
        // one. What is wrong with an entry of one is reported to whoever
        // asks for the partition.
        for entries in unwanted {
            let partition = entries.partition();
            if self.cache.free_bytes() == 0 {
                break;
            }
            let Ok(files) = build_files(entries) else {
                continue;
            };
            let columns = entries.build_columns(files.stats_field_ids());
            let bounds = files
                .stats_field_ids()
                .iter()
                .zip(columns)
                .map(|(field_id, column)| (*field_id, Arc::new(column)))
                .collect();
            let cache = &self.cache;
            let files = Arc::new(files);
            let is_held = cache.store_unasked_manifest_files(
                table_id,
                format_version,
                manifest_path,
                partition,
                files,
            ) && cache.store_unasked_column_bounds(
                table_id,
                format_version,
                manifest_path,
                partition,
                bounds,
            );
            if !is_held {
                break;
            }
        }
        Ok(())
    }

    /// The partitions each of `manifests` holds a live file of, in the
    /// order given. A manifest whose partitions neither its manifest-list
    /// entry tells nor the cache holds
    /// ([`find_partitions`](Self::find_partitions)) is read and parsed,
    /// which keeps the files of the partitions `wants` accepts for the
    /// plan. Two plans that miss the same manifest at the same time each
    /// parse it and store its parts.
    fn read_partitions(
        &mut self,
        manifests: &[&ListedManifest],
        wants: &mut PartitionCheck<'_>,
    ) -> Result<Vec<Arc<ManifestPartitions>>> {
        let missing = self.find_partitions(manifests);
        self.parse_manifests(&missing, wants)?;
        Ok(manifests
            .iter()
            .map(|manifest| Arc::clone(&self.partitions[manifest.path()]))
            .collect())
    }

    /// The partitions among `wanted` that `manifests` hold a live file of,
    /// in their order, each with the manifests that hold one, in the order
    /// given: one walk over the manifests, each matched by the partitions
    /// it holds, which its manifest-list entry tells of some with nothing
    /// read ([`find_partitions`](Self::find_partitions)). A manifest
    /// parsed to learn its partitions keeps the files of those a check
    /// keeps for the plan.
    pub(crate) fn match_partitions<'m>(
        &mut self,
        manifests: &[&'m ListedManifest],
        wanted: WantedPartitions<'_>,
    ) -> Result<Vec<PartitionManifests<'m>>> {
        let pairs: Vec<MatchedPair<'m>> = match wanted {
            WantedPartitions::Listed(partitions) => self.match_listed(manifests, partitions),
            WantedPartitions::Matching(keeps) => {
                let mut kept: HashMap<Arc<Partition>, bool> = HashMap::new();
                let mut keeps_once = |partition: &Arc<Partition>| -> Result<bool> {
                    if let Some(is_kept) = kept.get(partition) {
                        return Ok(*is_kept);
                    }
                    let is_kept = keeps(partition)?;
                    kept.insert(Arc::clone(partition), is_kept);
                    Ok(is_kept)
                };
                let held = self.read_partitions(manifests, &mut keeps_once)?;
                let mut kept_pairs = Vec::new();
                for (manifest, partitions) in manifests.iter().zip(&held) {
                    for partition in partitions.iter() {
                        if keeps_once(partition)? {
                            kept_pairs.push((*manifest, Arc::clone(partition)));
                        }
                    }
                }
                kept_pairs
            }
        };

        let mut matched: Vec<PartitionManifests<'m>> = Vec::new();
        // Where each partition stands among `matched`.
        let mut positions: HashMap<Arc<Partition>, usize> = HashMap::new();
        for (manifest, partition) in pairs {
            let position = positions.entry(Arc::clone(&partition));
            let position = *position.or_insert_with(|| {
                let manifests = Vec::new();
                matched.push(PartitionManifests {
                    partition,
                    manifests,
                });
                matched.len() - 1
            });
            matched[position].manifests.push(manifest);
        }
        matched.sort_unstable_by(|left, right| left.partition.cmp(&right.partition));
        Ok(matched)
    }

    /// Each of `manifests` with each of `wanted`, a few partitions in
    /// their order, it holds a live file of, as far as that is told with no
    /// manifest read. A manifest is paired with the wanted partitions the
    /// summaries of its manifest-list entry admit, and one that admits any
    /// has its partitions looked for
    /// ([`find_partitions`](Self::find_partitions)), which leave the pairs
    /// it holds a file of. A manifest whose partitions are not found
    /// keeps every pair, of which it may turn out to hold no file.
    fn match_listed<'m>(
        &mut self,
        manifests: &[&'m ListedManifest],
        wanted: &[Arc<Partition>],
    ) -> Vec<MatchedPair<'m>> {
        let mut pairs: Vec<MatchedPair<'m>> = Vec::new();
        let mut admitting: Vec<&ListedManifest> = Vec::new();
        for manifest in manifests {
            let pairs_before = pairs.len();
            let admitted = wanted.iter();
            let admitted = admitted.filter(|wanted| manifest.may_hold_partition(wanted));
            pairs.extend(admitted.map(|wanted| (*manifest, Arc::clone(wanted))));
            if pairs.len() > pairs_before {
                admitting.push(manifest);
            }
        }
        self.find_partitions(&admitting);
        pairs.retain(|(manifest, partition)| {
            let held = self.partitions.get(manifest.path());
            held.is_none_or(|held| held.holds(partition))
        });
        pairs
    }

    /// Find out what each of `asked` holds, the files of a partition in a
    /// manifest with the bounds the plan reads of them. A manifest is read
    /// and parsed when the cache does not hold what the plan reads of it,
    /// once for all the partitions asked of it, and keeps for the plan the
    /// files of the partitions `wants` accepts.
    fn resolve_slices(
        &mut self,
        asked: &[AskedSlice<'_>],
        wants: &mut PartitionCheck<'_>,
    ) -> Result<()> {
        let unresolved = self.find_slices(asked);
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
            if !self.knows_slice(manifest, partition) {
                self.keep_slice(manifest, partition, None);
            }
        }
        Ok(())
    }

    /// The files of each of `matched` with the bounds the plan reads of
    /// them, in the order given: a slice for each manifest that holds a
    /// file of the partition, in the order the match lists them. The cache
    /// is asked for the files and the bounds of those pairs and of no
    /// other. A manifest is read and parsed when the cache does not hold
    /// what the plan reads of it, once for all the partitions.
    pub(crate) fn load_slices(
        &mut self,
        matched: &[PartitionManifests<'_>],
    ) -> Result<Vec<Vec<Arc<PartitionSlice>>>> {
        let asked: Vec<AskedSlice<'_>> = matched
            .iter()
            .flat_map(|held| {
                let manifests = held.manifests.iter();
                manifests.map(move |manifest| (*manifest, &held.partition))
            })
            .collect();
        let wanted: HashSet<&Partition> = matched.iter().map(|held| &*held.partition).collect();
        self.resolve_slices(&asked, &mut |partition| Ok(wanted.contains(&**partition)))?;
        Ok(matched
            .iter()
            .map(|held| {
                let manifests = held.manifests.iter();
                manifests
                    .filter_map(|manifest| {
                        let slices = self.slices.get(manifest.path())?;
                        slices.get(&held.partition)?.clone()
                    })
                    .collect()
            })
            .collect())
    }

    /// What each of `manifests` lists, in the order given: the files of
    /// every partition the manifest holds a live file of, in the order of
    /// the partitions, with the bounds the plan reads of them. A
    /// manifest whose manifest-list entry counts no live file is not read;
    /// any other is read and parsed when the cache does not hold what the
    /// plan reads of it.
    pub(crate) fn load_manifests(
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
                        slices.get(partition)?.clone()
                    })
                    .collect()
            })
            .collect())
    }
}
