use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;

use super::CompactionPass;
use super::sort_key::{BoundSide, compare_sort_bounds, has_sort_range};
use crate::Result;
use crate::table_format::CompactionFile;
use object_storage::{FileRef, ObjectPath};
use parquet_engine::RowGroupMetadata;
use parquet_engine::file_entry::FileEntry;

/// A file's row groups' min/max on every sort column, in the table's sort
/// order, each column's in row-group order. `None` marks a row group whose
/// footer recorded no bounds for a column; its rows can lie anywhere in the
/// file's range, so the contested-group count includes it.
pub(super) type RowGroupRanges = Vec<Vec<Option<(ArrayRef, ArrayRef)>>>;

/// Extract each of a file's row groups' min/max on each of `sort_by` from
/// `row_groups`, its footer metadata.
pub(super) fn read_row_group_ranges(
    row_groups: &[Arc<RowGroupMetadata>],
    sort_by: &[String],
) -> RowGroupRanges {
    sort_by
        .iter()
        .map(|sort_column| {
            row_groups
                .iter()
                .map(|row_group| {
                    let column = row_group.schema.index_of(sort_column).ok()?;
                    let statistics = row_group.column_statistics(column)?;
                    let min = statistics.min()?.into_inner();
                    let max = statistics.max()?.into_inner();
                    Some((min, max))
                })
                .collect()
        })
        .collect()
}

/// What reads the bounds of the row groups of each of the files an
/// observation measures, in the order given, from their footers.
pub(super) type ReadRowGroupRanges<'r> = dyn Fn(&[FileRef]) -> Result<Vec<RowGroupRanges>> + 'r;

/// Where the record `file` is held: what tells it from another record of
/// the same file, for as long as a list holds it.
pub(super) fn locate_record(file: &Arc<dyn CompactionFile>) -> usize {
    Arc::as_ptr(file).cast::<()>() as usize
}

/// A layout-optimization candidate: a file's entry beside its row groups'
/// min/max on every sort column. A pair is measured on the column
/// that tells the two files apart, which is not always the first: files
/// holding one identical value on the leading keys are ordered by the next
/// one, and that is where their row groups have to be compared.
struct LayoutCandidate<'a> {
    entry: &'a FileEntry,
    row_group_ranges: RowGroupRanges,
}

impl LayoutCandidate<'_> {
    /// The file's row group count, which every sort column's bounds share.
    fn row_group_count(&self) -> usize {
        self.row_group_ranges
            .first()
            .map_or(0, |per_column| per_column.len())
    }
}

/// The file's min/max on `column` from its entry's statistics.
fn column_bounds<'a>(entry: &'a FileEntry, column: &str) -> Option<(&'a ArrayRef, &'a ArrayRef)> {
    let stats = entry.stats.as_ref()?;
    Some((stats.min_values.get(column)?, stats.max_values.get(column)?))
}

fn compare(left: &ArrayRef, right: &ArrayRef) -> Option<Ordering> {
    if left.len() != 1 || right.len() != 1 || left.is_null(0) || right.is_null(0) {
        return None;
    }
    // One column's bounds reach selection from two places that need not decode
    // them alike: a file's own statistics come from its table's metadata, its
    // row groups' from its footer, and the same logical type can arrive as `Utf8`
    // from one and `Utf8View` from the other. Cast the pair together rather
    // than calling it incomparable, which would leave every comparison
    // undecided.
    if left.data_type() != right.data_type() {
        let right = cast(right.as_ref(), left.data_type()).ok()?;
        return compare_same_type(left, &right);
    }
    compare_same_type(left, right)
}

/// Order two single-value arrays of one type through one typed comparator,
/// built once per call: selection compares bounds by the million, and a
/// comparator costs a fraction of the comparison kernels' result arrays.
fn compare_same_type(left: &ArrayRef, right: &ArrayRef) -> Option<Ordering> {
    let comparator = make_comparator(left.as_ref(), right.as_ref(), SortOptions::default()).ok()?;
    Some(comparator(0, 0))
}

/// Whether `file` can be one of a layout pair: it holds a row, and the
/// column its table's sort key leads with is bounded on both sides, no
/// higher below than above.
fn is_measurable(file: &dyn CompactionFile) -> bool {
    let read_leading_bound = |side| file.read_sort_bound(0, side);
    let has_rows = file.row_count().is_some_and(|rows| rows > 0);
    has_rows
        && has_sort_range(file)
        && read_leading_bound(BoundSide::Lower) <= read_leading_bound(BoundSide::Upper)
}

/// How much two files' rows interleave, from their row-group bounds.
#[derive(Clone, Copy)]
struct PairOverlap {
    /// The smaller of the two files' contested row-group shares. Zero when
    /// the ranges overlap but no row group of either file nests in the
    /// other's range.
    contested_share: f64,
    /// Whether the pair is worth a routine rewrite: both files have more
    /// than [`CONTESTED_ROW_GROUP_PERCENT_BAR`] percent of their row groups
    /// contested. A final sweep ignores this and takes any measured pair.
    clears_bar: bool,
}

/// The sort column a pair of measurable files is measured on, by position,
/// or `None` for a pair no pass may ever merge.
///
/// The pair is decided on the first sort column where the two files are not
/// the same single value. Ranges that are disjoint or touch at one value
/// share no stretch of the key and never merge, and neither do files holding
/// one identical key value throughout, which have nothing to sort. Both
/// rules hold in every pass; they are what makes compaction terminate. A
/// bound that cannot be compared keeps the pair rather than dropping it.
fn find_deciding_column(left: &FileEntry, right: &FileEntry, sort_by: &[String]) -> Option<usize> {
    let left_stats = left.stats.as_ref()?;
    let right_stats = right.stats.as_ref()?;
    for (column, name) in sort_by.iter().enumerate() {
        let (Some(left_min), Some(left_max), Some(right_min), Some(right_max)) = (
            left_stats.min_values.get(name),
            left_stats.max_values.get(name),
            right_stats.min_values.get(name),
            right_stats.max_values.get(name),
        ) else {
            return None;
        };
        let same_single_value = compare(left_min, left_max) == Some(Ordering::Equal)
            && compare(right_min, right_max) == Some(Ordering::Equal)
            && compare(left_min, right_min) == Some(Ordering::Equal);
        if same_single_value {
            continue;
        }
        // Bounds are inclusive: a maximum equal to the other minimum touches
        // it at exactly one value.
        if matches!(
            compare(left_max, right_min),
            Some(Ordering::Less | Ordering::Equal)
        ) || matches!(
            compare(right_max, left_min),
            Some(Ordering::Less | Ordering::Equal)
        ) {
            return None;
        }
        return Some(column);
    }
    // One identical key value throughout both files: nothing to sort, and a
    // rewrite could only cut the value's run and hand the pair back.
    None
}

/// Measure a pair on the sort column at `column`, named `name`, from the two
/// files' row groups. The smaller contested share is kept: a narrow file
/// nested in a much wider one must not outrank a pair that restructures
/// both sides.
fn overlap_from_row_groups(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    column: usize,
    name: &str,
) -> Option<PairOverlap> {
    let left_range = column_bounds(left.entry, name)?;
    let right_range = column_bounds(right.entry, name)?;
    let left_contested = count_contested_row_groups(left, column, right_range);
    let right_contested = count_contested_row_groups(right, column, left_range);
    let clears_bar = left_contested >= required_contested_row_groups(left.row_group_count())
        && right_contested >= required_contested_row_groups(right.row_group_count());
    let left_share = left_contested as f64 / left.row_group_count().max(1) as f64;
    let right_share = right_contested as f64 / right.row_group_count().max(1) as f64;
    Some(PairOverlap {
        contested_share: left_share.min(right_share),
        clears_bar,
    })
}

/// What each pass may merge. A routine round merges only full cliques,
/// exactly `clique_size` files that pairwise clear the contested bar, and
/// smaller cliques stand. A final sweep merges two to `clique_size` files
/// connected by measured overlap, bar or no bar and not necessarily a
/// clique, so a wide file goes together with the narrow files nested in
/// its range.
impl CompactionPass {
    /// Whether the pass may use a remembered pair.
    fn admits(self, overlap: PairOverlap) -> bool {
        match self {
            Self::Routine => overlap.clears_bar,
            Self::Final => true,
        }
    }
}

/// Two files an observation has to measure against each other.
type CandidatePair<'f> = (&'f Arc<dyn CompactionFile>, &'f Arc<dyn CompactionFile>);

/// Every pair of the measurable files of `large`, one partition's large
/// files in the order they stand in, that holds a file `is_new` accepts and
/// may share a stretch of the sort key.
///
/// The files come in order of their lowest key, and the sweep keeps in
/// reach those whose highest key has not yet been passed; only those can
/// share a stretch of the key with the current file. A new file is paired
/// with every file in reach, a file measured before only with the new
/// ones, so the first observation of a partition yields its overlapping
/// pairs rather than all pairs.
fn collect_pairs<'f>(
    large: impl Iterator<Item = &'f Arc<dyn CompactionFile>>,
    is_new: &dyn Fn(&Arc<dyn CompactionFile>) -> bool,
) -> Vec<CandidatePair<'f>> {
    let mut pairs = Vec::new();
    // The files whose range the sweep has not passed: every one, and the
    // new ones among them.
    let mut in_reach: Vec<&'f Arc<dyn CompactionFile>> = Vec::new();
    let mut new_in_reach: Vec<&'f Arc<dyn CompactionFile>> = Vec::new();
    for candidate in large.filter(|file| is_measurable(file.as_ref())) {
        let reaches_candidate = |partner: &&'f Arc<dyn CompactionFile>| {
            let (partner, candidate) = (partner.as_ref(), candidate.as_ref());
            compare_sort_bounds(partner, BoundSide::Upper, candidate, BoundSide::Lower).is_ge()
        };
        new_in_reach.retain(reaches_candidate);
        let is_new_candidate = is_new(candidate);
        let partners = if is_new_candidate {
            in_reach.retain(reaches_candidate);
            &in_reach
        } else {
            &new_in_reach
        };
        pairs.extend(partners.iter().map(|partner| (*partner, candidate)));
        in_reach.push(candidate);
        if is_new_candidate {
            new_in_reach.push(candidate);
        }
    }
    pairs
}

/// The large files of a partition no observation measured yet.
#[derive(Default)]
enum UnmeasuredFiles {
    /// Every one: no observation of the partition succeeded yet.
    #[default]
    Every,
    /// Those that joined since the last observation that succeeded, each
    /// by where its record is held.
    Joined(HashSet<usize>),
}

/// A file that has a partner.
struct PairedFile {
    /// The file's record, as its partition's list holds it.
    record: Arc<dyn CompactionFile>,
    /// The file's partners, and how far each overlaps it.
    partners: HashMap<Arc<ObjectPath>, PairOverlap>,
}

/// What the rounds know of the layout of one partition of a table.
///
/// An overlap depends only on the two files' footers and the bounds their
/// table's metadata records, so it is measured once: an observation
/// measures the files that joined the partition since the one before
/// against the files of the partition they can interleave with. The
/// partition's lists tell the cache of every large file that joins
/// ([`OverlapCache::add_unmeasured_file`]) and of every file that leaves,
/// which is forgotten there and then ([`OverlapCache::forget_file`]).
///
/// Between rounds it holds the pairs that clear the contested bar, each
/// with its share, and nothing else: a pair below the bar is one no routine
/// round merges, and nothing is held of a file of no pair. A final sweep, which
/// merges those too, measures every pair again when it reaches the
/// partition, holds them all while it works on it and lets go of the ones
/// below the bar when it is done.
///
/// A pair is kept under both of its files, so a file's partners are one
/// lookup away. The pairs name a file by one shared copy of its path, so
/// each is stored once however many pairs contain it.
#[derive(Default)]
pub(super) struct OverlapCache {
    /// The files that have a partner, each under the one copy of its path
    /// the pairs name it by.
    paired: HashMap<Arc<ObjectPath>, PairedFile>,
    /// The large files no observation measured yet.
    unmeasured: UnmeasuredFiles,
    /// Whether the pairs below the contested bar are held too: while a
    /// final sweep works on the partition.
    holds_every_pair: bool,
}

impl OverlapCache {
    /// Bring the cache up to date with `large`, the partition's large
    /// files in the order they stand in, for a round running `pass`: the
    /// files that joined since the last observation that succeeded are
    /// measured against the files they can interleave with, and count as
    /// measured once that has succeeded. A routine round lets go of the
    /// pairs only a final sweep holds. The first observation of a final
    /// sweep measures every file again, and what it measured takes the
    /// place of what was held once every pair is measured.
    ///
    /// No footer is read and no entry built to tell which pairs may share
    /// a stretch of the key: that is decided by comparing the bounds on the
    /// sort key where the records hold them ([`collect_pairs`]). Only the
    /// files of those pairs have their entries built, off their records,
    /// and only those of the pairs the entries' bounds do not rule out have
    /// their footers read, by `read_row_group_ranges`.
    pub(super) fn observe<'f>(
        &mut self,
        large: impl Iterator<Item = &'f Arc<dyn CompactionFile>>,
        read_row_group_ranges: &ReadRowGroupRanges<'_>,
        sort_by: &[String],
        pass: CompactionPass,
    ) -> Result<()> {
        match pass {
            CompactionPass::Final if !self.holds_every_pair => {
                let mut every_pair = Self {
                    holds_every_pair: true,
                    ..Self::default()
                };
                let pairs = collect_pairs(large, &|_| true);
                every_pair.measure_pairs(pairs, read_row_group_ranges, sort_by)?;
                *self = every_pair;
                return Ok(());
            }
            CompactionPass::Final => {}
            CompactionPass::Routine => self.forget_pairs_below_bar(),
        }
        let pairs = match &self.unmeasured {
            UnmeasuredFiles::Joined(joined) if joined.is_empty() => return Ok(()),
            UnmeasuredFiles::Joined(joined) => {
                collect_pairs(large, &|file| joined.contains(&locate_record(file)))
            }
            UnmeasuredFiles::Every => collect_pairs(large, &|_| true),
        };
        self.measure_pairs(pairs, read_row_group_ranges, sort_by)
    }

    /// Count `file`, a large file that joined the partition, as one no
    /// observation measured.
    pub(super) fn add_unmeasured_file(&mut self, file: &Arc<dyn CompactionFile>) {
        if let UnmeasuredFiles::Joined(joined) = &mut self.unmeasured {
            joined.insert(locate_record(file));
        }
    }

    /// The record of the file at `path`, one that has a partner, as its
    /// partition's list holds it.
    pub(super) fn find_paired_file(&self, path: &ObjectPath) -> &Arc<dyn CompactionFile> {
        &self.paired[path].record
    }

    /// Forget `file`, which left the partition: every pair that holds it,
    /// and that it was yet to be measured.
    pub(super) fn forget_file(&mut self, file: &Arc<dyn CompactionFile>) {
        if let UnmeasuredFiles::Joined(joined) = &mut self.unmeasured {
            joined.remove(&locate_record(file));
        }
        if let Some((name, _)) = self.paired.get_key_value(&file.file().path) {
            let name = Arc::clone(name);
            self.forget_pairs_of(&name);
        }
    }

    /// Know the file of `listed` by `kept`, another record of it, which
    /// its partition's list holds from here on.
    pub(super) fn replace_record(
        &mut self,
        listed: &Arc<dyn CompactionFile>,
        kept: &Arc<dyn CompactionFile>,
    ) {
        if let UnmeasuredFiles::Joined(joined) = &mut self.unmeasured
            && joined.remove(&locate_record(listed))
        {
            joined.insert(locate_record(kept));
        }
        if let Some(paired) = self.paired.get_mut(&kept.file().path) {
            paired.record = Arc::clone(kept);
        }
    }

    /// Let go of the pairs below the contested bar, which only the final
    /// sweep that measured them merges: what stays is what the rounds
    /// remember between them.
    pub(super) fn forget_pairs_below_bar(&mut self) {
        if !self.holds_every_pair {
            return;
        }
        for file in self.paired.values_mut() {
            file.partners.retain(|_, overlap| overlap.clears_bar);
        }
        self.paired.retain(|_, file| !file.partners.is_empty());
        self.holds_every_pair = false;
    }

    /// Measure each of `pairs` and remember, under both of their files,
    /// those that share a stretch of the sort key and that the cache holds:
    /// every one during a final sweep, else those that clear the contested
    /// bar. No file is left to measure once that has succeeded.
    fn measure_pairs(
        &mut self,
        pairs: Vec<CandidatePair<'_>>,
        read_row_group_ranges: &ReadRowGroupRanges<'_>,
        sort_by: &[String],
    ) -> Result<()> {
        /// How many files' footers are held at a time.
        const FILES_PER_READ: usize = 256;

        // The entry of each file of a pair, by where its record is held.
        let mut entries: HashMap<usize, FileEntry> = HashMap::new();
        for file in pairs.iter().flat_map(|(left, right)| [*left, *right]) {
            let entry = entries.entry(locate_record(file));
            entry.or_insert_with(|| file.build_entry());
        }
        let decided: Vec<(CandidatePair<'_>, usize)> = pairs
            .into_iter()
            .filter_map(|(left, right)| {
                let left_entry = &entries[&locate_record(left)];
                let right_entry = &entries[&locate_record(right)];
                let column = find_deciding_column(left_entry, right_entry, sort_by)?;
                Some(((left, right), column))
            })
            .collect();

        // The files of the pairs left, each once, in the order each first
        // appears.
        let mut listed = HashSet::new();
        let measured: Vec<&Arc<dyn CompactionFile>> = decided
            .iter()
            .flat_map(|((left, right), _)| [*left, *right])
            .filter(|file| listed.insert(locate_record(file)))
            .collect();
        let mut candidates: HashMap<usize, LayoutCandidate<'_>> = HashMap::new();
        for files in measured.chunks(FILES_PER_READ) {
            let named: Vec<FileRef> = files.iter().map(|file| file.file().clone()).collect();
            for (file, row_group_ranges) in files.iter().zip(read_row_group_ranges(&named)?) {
                let held_at = locate_record(file);
                let candidate = LayoutCandidate {
                    entry: &entries[&held_at],
                    row_group_ranges,
                };
                candidates.insert(held_at, candidate);
            }
        }

        for ((left, right), column) in decided {
            let Some(overlap) = overlap_from_row_groups(
                &candidates[&locate_record(left)],
                &candidates[&locate_record(right)],
                column,
                &sort_by[column],
            ) else {
                continue;
            };
            if !self.holds_every_pair && !overlap.clears_bar {
                continue;
            }
            let (left_name, right_name) = (self.name_file(left), self.name_file(right));
            for (name, partner) in [(&left_name, &right_name), (&right_name, &left_name)] {
                let file = self.paired.get_mut(name).expect("a named file is held");
                file.partners.insert(Arc::clone(partner), overlap);
            }
        }
        self.unmeasured = UnmeasuredFiles::Joined(HashSet::new());
        Ok(())
    }

    /// The one copy of the path of `file` the pairs name it by: the one
    /// held, or a new one, held with the file's record from here on.
    fn name_file(&mut self, file: &Arc<dyn CompactionFile>) -> Arc<ObjectPath> {
        let path = &file.file().path;
        if let Some((name, _)) = self.paired.get_key_value(path) {
            return Arc::clone(name);
        }
        let name = Arc::new(path.clone());
        let paired = PairedFile {
            record: Arc::clone(file),
            partners: HashMap::new(),
        };
        self.paired.insert(Arc::clone(&name), paired);
        name
    }

    /// Choose the files the next merge rewrites together, as paths, or
    /// `None` when no merge should start.
    ///
    /// Files are tried as seeds from the one with the most partners the
    /// pass admits down, ties broken by the best of those shares and then
    /// by path so the choice is stable. In a routine round only a complete
    /// clique of `clique_size` mutually overlapping files is merged, so a
    /// seed whose neighborhood cannot fill one is passed over for one that
    /// can. In a final sweep the densest seed's connected group is always
    /// taken, up to `clique_size` files.
    pub(super) fn select_group(
        &self,
        reserved: &HashSet<ObjectPath>,
        pass: CompactionPass,
        clique_size: usize,
    ) -> Option<Vec<Arc<ObjectPath>>> {
        // Each file a group may be grown from, with how many admitted
        // partners it has and the best of their shares.
        let mut seeds: Vec<(&Arc<ObjectPath>, usize, f64)> = self
            .paired
            .iter()
            .filter(|(path, _)| !reserved.contains(path.as_ref()))
            .filter_map(|(path, file)| {
                let (count, best) = file
                    .partners
                    .iter()
                    .filter(|(partner, overlap)| {
                        pass.admits(**overlap) && !reserved.contains(partner.as_ref())
                    })
                    .fold((0, f64::NEG_INFINITY), |(count, best), (_, overlap)| {
                        (count + 1, best.max(overlap.contested_share))
                    });
                (count > 0).then_some((path, count, best))
            })
            .collect();
        seeds.sort_unstable_by(
            |(left, left_count, left_best), (right, right_count, right_best)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| right_best.total_cmp(left_best))
                    .then_with(|| left.as_str().cmp(right.as_str()))
            },
        );
        for (seed, _, _) in seeds {
            match pass {
                CompactionPass::Routine => {
                    if let Some(clique) = self.grow_clique(seed, reserved, clique_size) {
                        return Some(clique);
                    }
                }
                CompactionPass::Final => {
                    return Some(self.grow_connected_group(seed, reserved, clique_size));
                }
            }
        }
        None
    }

    /// Grow a clique from `seed`: up to `clique_size` files that pairwise
    /// clear the contested bar. Candidates are tried highest lowest-share
    /// first, ties to the lexicographically first path, and a choice that
    /// cannot be completed is undone for the next one, so a full clique
    /// around `seed` is found whenever one exists.
    fn grow_clique(
        &self,
        seed: &Arc<ObjectPath>,
        reserved: &HashSet<ObjectPath>,
        clique_size: usize,
    ) -> Option<Vec<Arc<ObjectPath>>> {
        let candidates: Vec<(&Arc<ObjectPath>, f64)> = self.paired[seed]
            .partners
            .iter()
            .filter(|(partner, overlap)| overlap.clears_bar && !reserved.contains(partner.as_ref()))
            .map(|(partner, overlap)| (partner, overlap.contested_share))
            .collect();
        let mut group = vec![seed.clone()];
        self.extend_clique(&mut group, candidates, clique_size)
            .then_some(group)
    }

    /// One level of the clique search: try each of `candidates` as the next
    /// member of `group` and recurse on the candidates that pair with it.
    /// Returns true once `group` holds `clique_size` files.
    fn extend_clique(
        &self,
        group: &mut Vec<Arc<ObjectPath>>,
        mut candidates: Vec<(&Arc<ObjectPath>, f64)>,
        clique_size: usize,
    ) -> bool {
        if group.len() == clique_size {
            return true;
        }
        // Stop once even all remaining candidates could not fill the clique.
        while group.len() + candidates.len() >= clique_size {
            let Some(newest) = take_best_candidate(&mut candidates) else {
                break;
            };
            // Candidates that pair with the newest member, each keeping the
            // lowest share it has against the group so far.
            let newest_partners = &self.paired[newest].partners;
            let narrowed = candidates
                .iter()
                .filter_map(|(path, lowest)| {
                    let overlap = newest_partners.get(*path)?;
                    overlap
                        .clears_bar
                        .then_some((*path, lowest.min(overlap.contested_share)))
                })
                .collect();
            group.push(newest.clone());
            if self.extend_clique(group, narrowed, clique_size) {
                return true;
            }
            // No full clique contains the newest member: undo the choice so
            // the next candidate is tried from the same group.
            group.pop();
        }
        false
    }

    /// The final sweep's group around `seed`: the seed and up to
    /// `max_group_size - 1` of its partners, highest contested share first,
    /// ties to the lexicographically first path. Every member overlaps the
    /// seed, which suffices because a merge sorts and re-cuts the whole
    /// group, so its outputs never overlap each other whatever the inputs
    /// shared.
    fn grow_connected_group(
        &self,
        seed: &Arc<ObjectPath>,
        reserved: &HashSet<ObjectPath>,
        max_group_size: usize,
    ) -> Vec<Arc<ObjectPath>> {
        let mut partners: Vec<(&Arc<ObjectPath>, f64)> = self.paired[seed]
            .partners
            .iter()
            .filter(|(partner, _)| !reserved.contains(partner.as_ref()))
            .map(|(partner, overlap)| (partner, overlap.contested_share))
            .collect();
        partners.sort_unstable_by(|(left_path, left_share), (right_path, right_share)| {
            right_share
                .total_cmp(left_share)
                .then_with(|| left_path.as_str().cmp(right_path.as_str()))
        });
        std::iter::once(seed.clone())
            .chain(partners.into_iter().map(|(partner, _)| partner.clone()))
            .take(max_group_size)
            .collect()
    }

    /// Forget every pair that holds the file at `path`, and with it each
    /// file the pair was the last of.
    fn forget_pairs_of(&mut self, path: &Arc<ObjectPath>) {
        let Some(file) = self.paired.remove(path) else {
            return;
        };
        for partner in file.partners.keys() {
            let is_left_alone = self.paired.get_mut(partner).is_some_and(|partner| {
                partner.partners.remove(path);
                partner.partners.is_empty()
            });
            if is_left_alone {
                self.paired.remove(partner);
            }
        }
    }
}

/// Remove and return the candidate with the highest share, ties to the
/// lexicographically first path, or `None` when there is no candidate.
fn take_best_candidate<'s>(
    candidates: &mut Vec<(&'s Arc<ObjectPath>, f64)>,
) -> Option<&'s Arc<ObjectPath>> {
    let best_index = candidates
        .iter()
        .enumerate()
        .max_by(
            |(_, (left_path, left_share)), (_, (right_path, right_share))| {
                left_share
                    .total_cmp(right_share)
                    .then_with(|| right_path.as_str().cmp(left_path.as_str()))
            },
        )
        .map(|(index, _)| index)?;
    Some(candidates.swap_remove(best_index).0)
}

/// A file clears the bar when more than this percent of its row groups are
/// contested, so it is also how much narrower a partner may be and still
/// count as the same scale. [`super::DEFAULT_LAYOUT_CLIQUE_SIZE`] is tied
/// to it.
pub(super) const CONTESTED_ROW_GROUP_PERCENT_BAR: usize = 20;

/// Contested row groups a file of `total` row groups needs to clear the bar.
/// Five groups need two, fifty need eleven. A single outlier row can
/// fabricate one contested group, so the requirement climbs past one at
/// five groups.
fn required_contested_row_groups(total: usize) -> usize {
    total * CONTESTED_ROW_GROUP_PERCENT_BAR / 100 + 1
}

/// How many of `file`'s row groups lie within `partner`'s range on the sort
/// column at `column`, bounds inclusive. Those groups sit where the partner
/// also holds rows, so a rewrite can restructure them against it; a group
/// reaching past the partner's range is anchored to rows only its own file
/// holds. A group without recorded bounds counts as contested.
fn count_contested_row_groups(
    file: &LayoutCandidate,
    column: usize,
    partner: (&ArrayRef, &ArrayRef),
) -> usize {
    let (partner_min, partner_max) = partner;
    file.row_group_ranges[column]
        .iter()
        .filter(|bounds| {
            let Some((min, max)) = bounds else {
                return true;
            };
            matches!(
                compare(min, partner_min),
                Some(Ordering::Greater | Ordering::Equal)
            ) && matches!(
                compare(max, partner_max),
                Some(Ordering::Less | Ordering::Equal)
            )
        })
        .count()
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Int64Array, StringArray, StringViewArray};

    use super::*;
    use crate::compact::sort_key::{SortValue, compare_file_order};
    use arrow_schema::DataType;
    use object_storage::{FileRef, ObjectPath};
    use parquet_engine::FileStats;

    /// A candidate as a partition's list holds it: its bounds on the sort
    /// columns are read off its entry's statistics, integers and strings.
    struct CandidateRecord {
        rows: Option<u64>,
        entry: FileEntry,
        sort_by: Vec<String>,
    }

    impl CompactionFile for CandidateRecord {
        fn file(&self) -> &FileRef {
            &self.entry.file
        }

        fn row_count(&self) -> Option<u64> {
            self.rows
        }

        fn commit_sequence(&self) -> u64 {
            0
        }

        fn count_sort_columns(&self) -> usize {
            self.sort_by.len()
        }

        fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_> {
            let Some(stats) = self.entry.stats.as_deref() else {
                return SortValue::Missing;
            };
            let bounds = match side {
                BoundSide::Lower => &stats.min_values,
                BoundSide::Upper => &stats.max_values,
            };
            let Some(bound) = bounds.get(&self.sort_by[position]) else {
                return SortValue::Missing;
            };
            match bound.data_type() {
                DataType::Int64 => {
                    SortValue::Integer(bound.as_primitive::<Int64Type>().value(0).into())
                }
                DataType::Utf8 => SortValue::String(bound.as_string::<i32>().value(0)),
                DataType::Utf8View => SortValue::String(bound.as_string_view().value(0)),
                other => panic!("no candidate bounds a sort column by a {other}"),
            }
        }

        fn build_entry(&self) -> FileEntry {
            self.entry.clone()
        }
    }

    /// The record of the file of `entry`, a file of a table sorted by
    /// `sort_by`.
    fn build_record(entry: &FileEntry, sort_by: &[String]) -> Arc<dyn CompactionFile> {
        let stats = entry.stats.as_ref();
        let rows = stats
            .and_then(|stats| stats.num_records)
            .and_then(|rows| u64::try_from(rows).ok());
        Arc::new(CandidateRecord {
            rows,
            entry: entry.clone(),
            sort_by: sort_by.to_vec(),
        })
    }

    /// One partition's large files as its lists hold them from one
    /// observation to the next, with what was measured of them: the lists
    /// are what tells the cache which files joined and which left.
    #[derive(Default)]
    struct ListedPartition {
        cache: OverlapCache,
        /// The record of each file listed, by the file's path.
        listed: HashMap<ObjectPath, Arc<dyn CompactionFile>>,
    }

    impl std::ops::Deref for ListedPartition {
        type Target = OverlapCache;

        fn deref(&self) -> &OverlapCache {
            &self.cache
        }
    }

    impl std::ops::DerefMut for ListedPartition {
        fn deref_mut(&mut self) -> &mut OverlapCache {
            &mut self.cache
        }
    }

    impl ListedPartition {
        /// List `files` and no other: a file listed that is not among them
        /// leaves, and one not listed before joins. Returns the records
        /// listed, in the order a partition's files stand in.
        fn list_files(
            &mut self,
            files: &[LayoutCandidate<'_>],
            sort_by: &[String],
        ) -> Vec<Arc<dyn CompactionFile>> {
            let is_among_files =
                |path: &ObjectPath| files.iter().any(|file| &file.entry.file.path == path);
            let gone: Vec<_> = self
                .listed
                .extract_if(|path, _| !is_among_files(path))
                .collect();
            for (_, record) in gone {
                self.cache.forget_file(&record);
            }
            for file in files {
                if !self.listed.contains_key(&file.entry.file.path) {
                    let record = build_record(file.entry, sort_by);
                    self.cache.add_unmeasured_file(&record);
                    self.listed.insert(file.entry.file.path.clone(), record);
                }
            }
            let mut large: Vec<Arc<dyn CompactionFile>> = self.listed.values().cloned().collect();
            large.sort_by(|left, right| compare_file_order(left.as_ref(), right.as_ref()));
            large
        }
    }

    /// Bring `partition` up to date with `files`, the large files it holds
    /// now, for a round running `pass`. Returns the files whose footers the
    /// observation read.
    fn observe_candidates(
        partition: &mut ListedPartition,
        files: &[LayoutCandidate<'_>],
        sort_by: &[String],
        pass: CompactionPass,
    ) -> Vec<String> {
        let large = partition.list_files(files, sort_by);
        let footers_read = std::cell::RefCell::new(Vec::new());
        let find_candidate = |file: &FileRef| {
            let mut candidates = files.iter();
            candidates
                .find(|candidate| candidate.entry.file.path == file.path)
                .unwrap()
        };
        let read_row_group_ranges = |read: &[FileRef]| -> Result<Vec<RowGroupRanges>> {
            let mut footers_read = footers_read.borrow_mut();
            footers_read.extend(read.iter().map(|file| file.path.to_string()));
            Ok(read
                .iter()
                .map(|file| find_candidate(file).row_group_ranges.clone())
                .collect())
        };

        let cache = &mut partition.cache;
        cache
            .observe(large.iter(), &read_row_group_ranges, sort_by, pass)
            .unwrap();

        footers_read.into_inner()
    }

    /// The best remembered bar-clearing pair in `cache` among `files`,
    /// measuring the files no earlier routine round measured: the pair a
    /// routine round seeds a clique from, so tests can probe the scoring
    /// directly.
    /// Reserved files' pairs are passed over; ties go to the
    /// lexicographically first pair.
    fn select_best_pair(
        cache: &mut ListedPartition,
        files: &[LayoutCandidate<'_>],
        sort_by: &[String],
        reserved: &HashSet<ObjectPath>,
    ) -> Option<(f64, Arc<ObjectPath>, Arc<ObjectPath>)> {
        let _ = observe_candidates(cache, files, sort_by, CompactionPass::Routine);
        let (left, right, overlap) = cache
            .paired
            .iter()
            .flat_map(|(left, file)| {
                let partners = file.partners.iter();
                partners.map(move |(right, overlap)| (left, right, overlap))
            })
            .filter(|(left, right, overlap)| {
                left.as_str() < right.as_str()
                    && overlap.clears_bar
                    && !reserved.contains(left.as_ref())
                    && !reserved.contains(right.as_ref())
            })
            .max_by(
                |(left_a, right_a, overlap_a), (left_b, right_b, overlap_b)| {
                    overlap_a
                        .contested_share
                        .total_cmp(&overlap_b.contested_share)
                        .then_with(|| {
                            (left_b.as_str(), right_b.as_str())
                                .cmp(&(left_a.as_str(), right_a.as_str()))
                        })
                },
            )?;
        Some((overlap.contested_share, left.clone(), right.clone()))
    }

    /// Every pair the cache holds, bar or no bar.
    fn remembered_pairs(cache: &OverlapCache) -> usize {
        let partners = cache.paired.values().map(|file| file.partners.len());
        partners.sum::<usize>() / 2
    }

    /// Remembered pairs that clear the contested bar, reserved or not: the
    /// pairs a routine round may merge.
    fn eligible_pairs(cache: &OverlapCache) -> usize {
        cache
            .paired
            .values()
            .flat_map(|file| file.partners.values())
            .filter(|overlap| overlap.clears_bar)
            .count()
            / 2
    }

    /// Select over one partition with nothing remembered.
    fn select_pair(
        files: &[LayoutCandidate<'_>],
        sort_by: &[String],
    ) -> Option<(f64, Arc<ObjectPath>, Arc<ObjectPath>)> {
        select_best_pair(
            &mut ListedPartition::default(),
            files,
            sort_by,
            &HashSet::new(),
        )
    }

    /// Select a group over one partition with nothing remembered and nothing
    /// reserved, as a round running `pass` does.
    fn select_group(
        files: &[LayoutCandidate<'_>],
        sort_by: &[String],
        pass: CompactionPass,
        clique_size: usize,
    ) -> Option<Vec<Arc<ObjectPath>>> {
        let mut cache = ListedPartition::default();
        observe_candidates(&mut cache, files, sort_by, pass);
        cache.select_group(&HashSet::new(), pass, clique_size)
    }

    fn group_paths(group: &[Arc<ObjectPath>]) -> Vec<&str> {
        group.iter().map(|path| path.as_str()).collect()
    }

    fn int_stat(value: i64) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }

    fn string_stat(value: &str) -> ArrayRef {
        Arc::new(StringViewArray::from(vec![value]))
    }

    fn stats_entry(path: &str, columns: &[(&str, ArrayRef, ArrayRef)]) -> FileEntry {
        let mut entry = FileEntry::new(FileRef {
            path: ObjectPath::new(path),
            size: 1,
        });
        entry.stats = Some(Arc::new(FileStats {
            num_records: Some(1),
            min_values: columns
                .iter()
                .map(|(name, min, _)| ((*name).to_string(), min.clone()))
                .collect(),
            max_values: columns
                .iter()
                .map(|(name, _, max)| ((*name).to_string(), max.clone()))
                .collect(),
            null_counts: std::collections::HashMap::new(),
        }));
        entry
    }

    /// An entry sorted by one integer `id` column, `size` bytes large.
    fn sized_entry(path: &str, min: i64, max: i64, size: u64) -> FileEntry {
        let mut entry = stats_entry(path, &[("id", int_stat(min), int_stat(max))]);
        entry.file.size = size;
        entry
    }

    fn utf8_stat(value: &str) -> ArrayRef {
        Arc::new(StringArray::from(vec![value]))
    }

    /// A candidate whose file has one row group spanning its whole range on
    /// each of `sort_by`, the layout a small single-group parquet has.
    fn candidate<'a>(entry: &'a FileEntry, sort_by: &[&str]) -> LayoutCandidate<'a> {
        let stats = entry.stats.as_ref().unwrap();
        LayoutCandidate {
            entry,
            row_group_ranges: sort_by
                .iter()
                .map(|column| {
                    vec![Some((
                        stats.min_values[*column].clone(),
                        stats.max_values[*column].clone(),
                    ))]
                })
                .collect(),
        }
    }

    /// A candidate of a singly sorted table, whose row groups span `groups` on
    /// its one sort column.
    fn candidate_with_row_groups<'a>(
        entry: &'a FileEntry,
        groups: &[(i64, i64)],
    ) -> LayoutCandidate<'a> {
        LayoutCandidate {
            entry,
            row_group_ranges: vec![
                groups
                    .iter()
                    .map(|(min, max)| Some((int_stat(*min), int_stat(*max))))
                    .collect(),
            ],
        }
    }

    /// A candidate of a table sorted by `leading_keys` string columns and one
    /// integer column last, `leading` giving each leading key's row-group
    /// bounds and `trailing` the last key's.
    fn keyed_candidate<'a>(
        entry: &'a FileEntry,
        leading_keys: usize,
        leading: &[(&str, &str)],
        trailing: &[(i64, i64)],
    ) -> LayoutCandidate<'a> {
        let leading: Vec<Option<(ArrayRef, ArrayRef)>> = leading
            .iter()
            .map(|(min, max)| Some((string_stat(min), string_stat(max))))
            .collect();
        let mut row_group_ranges = vec![leading; leading_keys];
        row_group_ranges.push(
            trailing
                .iter()
                .map(|(min, max)| Some((int_stat(*min), int_stat(*max))))
                .collect(),
        );
        LayoutCandidate {
            entry,
            row_group_ranges,
        }
    }

    fn candidate_with_string_row_groups<'a>(
        entry: &'a FileEntry,
        groups: &[(&str, &str)],
    ) -> LayoutCandidate<'a> {
        LayoutCandidate {
            entry,
            row_group_ranges: vec![
                groups
                    .iter()
                    .map(|(min, max)| Some((string_stat(min), string_stat(max))))
                    .collect(),
            ],
        }
    }

    /// Files holding one identical value on the leading keys are told apart
    /// by the next one, and it is that key's row groups that are counted.
    #[test]
    fn equal_singleton_prefix_measures_the_next_sort_column() {
        let sort_by = ["region".into(), "id".into()];
        let left = stats_entry(
            "left",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(0), int_stat(100)),
            ],
        );
        let right = stats_entry(
            "right",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(50), int_stat(150)),
            ],
        );
        let files = [
            keyed_candidate(
                &left,
                1,
                &[("us", "us"); 5],
                &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)],
            ),
            keyed_candidate(
                &right,
                1,
                &[("us", "us"); 5],
                &[(50, 70), (70, 90), (90, 110), (110, 130), (130, 150)],
            ),
        ];

        let (overlap, _, _) = select_pair(&files, &sort_by).unwrap();

        // Two of each file's five groups lie within the other's id range,
        // which the shared region says nothing about.
        assert!((overlap - 0.4).abs() < 1e-12);
        assert!(select_pair(&files, &["region".into(), "missing".into()]).is_none());
    }

    /// Two files of one deployment and one service, holding different stretches
    /// of time, are ordered apart by that time: nothing to merge.
    #[test]
    fn identical_leading_keys_with_disjoint_next_key_is_no_pair() {
        let sort_by = ["deployment".into(), "service".into(), "time".into()];
        let early = stats_entry(
            "early",
            &[
                ("deployment", string_stat("a"), string_stat("a")),
                ("service", string_stat("a"), string_stat("a")),
                ("time", int_stat(1), int_stat(10)),
            ],
        );
        let late = stats_entry(
            "late",
            &[
                ("deployment", string_stat("a"), string_stat("a")),
                ("service", string_stat("a"), string_stat("a")),
                ("time", int_stat(20), int_stat(30)),
            ],
        );
        let files = [
            keyed_candidate(
                &early,
                2,
                &[("a", "a"); 5],
                &[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
            ),
            keyed_candidate(
                &late,
                2,
                &[("a", "a"); 5],
                &[(20, 22), (23, 24), (25, 26), (27, 28), (29, 30)],
            ),
        ];

        let best = select_pair(&files, &sort_by);

        assert!(best.is_none());
    }

    /// A pair holding one identical key value throughout has nothing to
    /// sort and is never selected.
    #[test]
    fn identical_value_pairs_never_merge() {
        let sort_by = ["region".into(), "id".into()];
        let left = stats_entry(
            "left",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(7), int_stat(7)),
            ],
        );
        let right = stats_entry(
            "right",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(7), int_stat(7)),
            ],
        );
        let files = [
            candidate(&left, &["region", "id"]),
            candidate(&right, &["region", "id"]),
        ];

        assert!(select_pair(&files, &sort_by).is_none());
        assert!(select_pair(&files, &["region".into(), "missing".into()]).is_none());
    }

    /// Two files spanning the same range, each a run of the low value, one
    /// transition group, and a run of the high value: every group lies within
    /// the shared range, so the pair is contested in full and the merge can
    /// pull the two values into files of their own.
    #[test]
    fn same_range_files_with_boundary_runs_pair_in_full() {
        let shape = [
            (10, 10),
            (10, 10),
            (10, 10),
            (10, 10),
            (10, 10),
            (10, 10),
            (10, 20),
            (20, 20),
            (20, 20),
        ];
        let left = stats_entry("left", &[("id", int_stat(10), int_stat(20))]);
        let right = stats_entry("right", &[("id", int_stat(10), int_stat(20))]);
        let files = [
            candidate_with_row_groups(&left, &shape),
            candidate_with_row_groups(&right, &shape),
        ];

        let (share, _, _) = select_pair(&files, &["id".into()]).unwrap();

        assert!((share - 1.0).abs() < 1e-12);
    }

    /// Two files spanning the same wide deployment range, their num key one
    /// and the same value throughout: the deployment column decides the pair,
    /// every row group lies within the shared range, and the pair is
    /// contested in full.
    #[test]
    fn same_range_files_with_a_single_next_key_value_pair_in_full() {
        let sort_by = ["deployment".into(), "num".into()];
        let left = stats_entry(
            "left",
            &[
                ("deployment", string_stat("a"), string_stat("z")),
                ("num", int_stat(1), int_stat(1)),
            ],
        );
        let right = stats_entry(
            "right",
            &[
                ("deployment", string_stat("a"), string_stat("z")),
                ("num", int_stat(1), int_stat(1)),
            ],
        );
        let deployments = [("a", "f"), ("f", "m"), ("m", "z")];
        let files = [
            keyed_candidate(&left, 1, &deployments, &[(1, 1); 3]),
            keyed_candidate(&right, 1, &deployments, &[(1, 1); 3]),
        ];

        let (share, _, _) = select_pair(&files, &sort_by).unwrap();

        assert!((share - 1.0).abs() < 1e-12);
    }

    /// Two files spanning the same deployment range, their num ranges shifted
    /// against each other: the deployment column decides the pair before num
    /// is ever consulted, and every row group lies within the partner's equal
    /// range.
    #[test]
    fn same_range_files_with_shifted_next_key_pair_in_full() {
        let sort_by = ["deployment".into(), "num".into()];
        let left = stats_entry(
            "left",
            &[
                ("deployment", string_stat("a"), string_stat("b")),
                ("num", int_stat(1), int_stat(200)),
            ],
        );
        let right = stats_entry(
            "right",
            &[
                ("deployment", string_stat("a"), string_stat("b")),
                ("num", int_stat(3), int_stat(200)),
            ],
        );
        let deployments = [("a", "a"), ("a", "b"), ("b", "b")];
        let files = [
            keyed_candidate(&left, 1, &deployments, &[(1, 100), (100, 200), (1, 200)]),
            keyed_candidate(&right, 1, &deployments, &[(3, 90), (90, 200), (3, 200)]),
        ];

        let (share, _, _) = select_pair(&files, &sort_by).unwrap();

        assert!((share - 1.0).abs() < 1e-12);
    }

    /// Two ranges meeting at exactly one value never merge, however heavy the
    /// runs of that value at the shared edge: a rewrite could only split the
    /// merged rows at that same value and hand back the two edges it started
    /// from.
    #[test]
    fn touching_ranges_with_heavy_boundary_runs_never_merge() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(10), int_stat(20))]);
        let files = [
            candidate_with_row_groups(&low, &[(0, 5), (5, 10), (10, 10), (10, 10)]),
            candidate_with_row_groups(&high, &[(10, 10), (10, 10), (10, 15), (15, 20)]),
        ];

        assert!(select_pair(&files, &["id".into()]).is_none());
    }

    /// Against a narrower partner, only the groups lying within the partner's
    /// range count: a boundary singleton is in, a group past the range and a
    /// group spanning beyond it are out.
    #[test]
    fn groups_past_the_partner_range_do_not_count() {
        let narrow = stats_entry("narrow", &[("id", int_stat(0), int_stat(10))]);
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(30))]);
        let files = [
            candidate_with_row_groups(&narrow, &[(0, 5), (5, 10)]),
            candidate_with_row_groups(&wide, &[(0, 0), (20, 20), (0, 30)]),
        ];

        let (share, _, _) = select_pair(&files, &["id".into()]).unwrap();

        assert!((share - 1.0 / 3.0).abs() < 1e-12);
    }

    /// A pair that only touches at a boundary value is passed over for one
    /// that genuinely interleaves.
    #[test]
    fn touching_pair_defers_to_an_interleaving_pair() {
        let range = sized_entry("range", 0, 100, 60);
        let pinned = sized_entry("pinned", 100, 100, 60);
        let nested = sized_entry("nested", 0, 50, 10);
        let files = [
            candidate_with_row_groups(&range, &[(0, 30), (30, 60), (60, 100)]),
            candidate(&pinned, &["id"]),
            candidate(&nested, &["id"]),
        ];

        let (share, left, right) = select_pair(&files, &["id".into()]).unwrap();

        let mut pair = [left.as_str(), right.as_str()];
        pair.sort();
        assert_eq!(pair, ["nested", "range"]);
        assert!((share - 1.0 / 3.0).abs() < 1e-12);
    }

    /// Four files of five row groups each: a wide file, a file shifted half a
    /// range past it, a narrow file inside it, and one nowhere near the rest.
    fn layered_candidates() -> Vec<LayoutCandidate<'static>> {
        static WIDE: std::sync::OnceLock<FileEntry> = std::sync::OnceLock::new();
        static SHIFTED: std::sync::OnceLock<FileEntry> = std::sync::OnceLock::new();
        static CONTAINED: std::sync::OnceLock<FileEntry> = std::sync::OnceLock::new();
        static DISJOINT: std::sync::OnceLock<FileEntry> = std::sync::OnceLock::new();
        let wide = WIDE.get_or_init(|| stats_entry("wide", &[("id", int_stat(0), int_stat(100))]));
        let shifted =
            SHIFTED.get_or_init(|| stats_entry("shifted", &[("id", int_stat(50), int_stat(150))]));
        let contained = CONTAINED
            .get_or_init(|| stats_entry("contained", &[("id", int_stat(40), int_stat(45))]));
        let disjoint = DISJOINT
            .get_or_init(|| stats_entry("disjoint", &[("id", int_stat(500), int_stat(600))]));
        vec![
            candidate_with_row_groups(
                shifted,
                &[(50, 70), (70, 90), (90, 110), (110, 130), (130, 150)],
            ),
            candidate_with_row_groups(
                disjoint,
                &[(500, 520), (520, 540), (540, 560), (560, 580), (580, 600)],
            ),
            candidate_with_row_groups(wide, &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(
                contained,
                &[(40, 41), (41, 42), (42, 43), (43, 44), (44, 45)],
            ),
        ]
    }

    /// After a merge, the next round measures only the merge's output against
    /// the survivors and no longer knows the pairs of the files it replaced.
    #[test]
    fn a_merge_output_is_measured_against_the_survivors() {
        let shape = [(0, 50), (50, 100)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let survivor = stats_entry("survivor", &[("id", int_stat(500), int_stat(600))]);
        let survivor_shape = [(500, 550), (550, 600)];
        let mut cache = ListedPartition::default();
        let before = vec![
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&survivor, &survivor_shape),
        ];
        select_best_pair(&mut cache, &before, &["id".into()], &HashSet::new());
        let merged = stats_entry("merged", &[("id", int_stat(500), int_stat(600))]);
        let after = vec![
            candidate_with_row_groups(&survivor, &survivor_shape),
            candidate_with_row_groups(&merged, &survivor_shape),
        ];

        let (share, left, right) =
            select_best_pair(&mut cache, &after, &["id".into()], &HashSet::new()).unwrap();

        assert!((share - 1.0).abs() < 1e-12);
        let mut pair = [left.as_str(), right.as_str()];
        pair.sort();
        assert_eq!(pair, ["merged", "survivor"]);
        assert_eq!(
            remembered_pairs(&cache),
            1,
            "the merged-away files' pairs are forgotten"
        );
    }

    /// A file some merge is already rewriting keeps its remembered overlaps but
    /// is passed over while it is reserved, so the round picks another pair.
    #[test]
    fn a_reserved_file_is_passed_over() {
        let shape = [(0, 50), (50, 100)];
        let far_shape = [(500, 550), (550, 600)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let far_first = stats_entry("far_first", &[("id", int_stat(500), int_stat(600))]);
        let far_second = stats_entry("far_second", &[("id", int_stat(500), int_stat(600))]);
        let files = vec![
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&far_first, &far_shape),
            candidate_with_row_groups(&far_second, &far_shape),
        ];
        let mut cache = ListedPartition::default();
        let (_, left, right) =
            select_best_pair(&mut cache, &files, &["id".into()], &HashSet::new()).unwrap();
        let reserved = HashSet::from([(*left).clone(), (*right).clone()]);

        let (_, next_left, next_right) =
            select_best_pair(&mut cache, &files, &["id".into()], &reserved).unwrap();

        assert!(!reserved.contains(next_left.as_ref()));
        assert!(!reserved.contains(next_right.as_ref()));
    }

    /// Three mutually overlapping files form the group; a fourth that pairs
    /// with one member but not the rest stays out however much room is left.
    #[test]
    fn grows_the_best_pair_into_a_clique_and_refuses_a_star_satellite() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(100))]);
        let mid = stats_entry("mid", &[("id", int_stat(50), int_stat(150))]);
        let wide = stats_entry("wide", &[("id", int_stat(40), int_stat(140))]);
        let satellite = stats_entry("satellite", &[("id", int_stat(105), int_stat(150))]);
        let files = [
            candidate_with_row_groups(&low, &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(
                &mid,
                &[(50, 70), (70, 90), (90, 110), (110, 130), (130, 150)],
            ),
            candidate_with_row_groups(
                &wide,
                &[(40, 60), (60, 80), (80, 100), (100, 120), (120, 140)],
            ),
            candidate_with_row_groups(
                &satellite,
                &[(105, 114), (114, 123), (123, 132), (132, 141), (141, 150)],
            ),
        ];

        let group = select_group(&files, &["id".into()], CompactionPass::Routine, 3).unwrap();

        // The satellite pairs with mid alone, so it cannot complete the clique.
        assert_eq!(group_paths(&group), ["mid", "wide", "low"]);
    }

    /// Four files that all overlap each other form a clique of exactly the
    /// configured width, equal shares resolved to the lexicographically
    /// first path.
    #[test]
    fn takes_exactly_the_clique_size() {
        let shape = [(0, 50), (50, 100)];
        let range = [("id", int_stat(0), int_stat(100))];
        let first = stats_entry("a", &range);
        let second = stats_entry("b", &range);
        let third = stats_entry("c", &range);
        let fourth = stats_entry("d", &range);
        let files = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&third, &shape),
            candidate_with_row_groups(&fourth, &shape),
        ];

        let group = select_group(&files, &["id".into()], CompactionPass::Routine, 3).unwrap();

        assert_eq!(group_paths(&group), ["a", "b", "c"]);
    }

    /// A width of two returns exactly the pair that pair selection picks.
    #[test]
    fn returns_the_selected_pair_unchanged_at_clique_size_two() {
        let files = layered_candidates();

        let group = select_group(&files, &["id".into()], CompactionPass::Routine, 2).unwrap();
        let (_, left, right) = select_pair(&files, &["id".into()]).unwrap();

        assert_eq!(group.len(), 2);
        assert_eq!(group[0], left);
        assert_eq!(group[1], right);
    }

    /// A file some merge is rewriting keeps its remembered overlaps but
    /// never joins a group while it is reserved: it cannot complete a
    /// clique, and a final sweep leaves it out.
    #[test]
    fn keeps_a_reserved_file_out_of_a_group() {
        let shape = [(0, 50), (50, 100)];
        let range = [("id", int_stat(0), int_stat(100))];
        let first = stats_entry("a", &range);
        let second = stats_entry("b", &range);
        let third = stats_entry("c", &range);
        let files = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&third, &shape),
        ];
        let reserved = HashSet::from([ObjectPath::new("c")]);
        let mut cache = ListedPartition::default();
        observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        let clique = cache.select_group(&reserved, CompactionPass::Routine, 3);
        let settled = cache
            .select_group(&reserved, CompactionPass::Final, 3)
            .unwrap();

        assert!(clique.is_none());
        assert_eq!(group_paths(&settled), ["a", "b"]);
    }

    /// Among two candidates that pair with every member, the one whose worst
    /// pairwise share is higher joins first.
    #[test]
    fn prefers_the_candidate_with_the_higher_lowest_share() {
        let tight_shape = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)];
        let left = stats_entry("left", &[("id", int_stat(0), int_stat(100))]);
        let right = stats_entry("right", &[("id", int_stat(0), int_stat(100))]);
        let close = stats_entry("close", &[("id", int_stat(0), int_stat(165))]);
        let far = stats_entry("far", &[("id", int_stat(0), int_stat(225))]);
        let files = [
            candidate_with_row_groups(&left, &tight_shape),
            candidate_with_row_groups(&right, &tight_shape),
            candidate_with_row_groups(
                &close,
                &[(0, 33), (33, 66), (66, 99), (99, 132), (132, 165)],
            ),
            candidate_with_row_groups(
                &far,
                &[(0, 45), (45, 90), (90, 135), (135, 180), (180, 225)],
            ),
        ];

        let group = select_group(&files, &["id".into()], CompactionPass::Routine, 3).unwrap();

        // Three of close's five groups lie inside the pair's range against
        // two of far's, so close carries the higher lowest share.
        assert_eq!(group_paths(&group), ["left", "right", "close"]);
    }

    /// Five mutually overlapping files are one short of the clique, so a
    /// routine selection waits; the sixth arrival completes it.
    #[test]
    fn waits_for_a_full_clique_before_merging() {
        let shape = [(0, 50), (50, 100)];
        let range = [("id", int_stat(0), int_stat(100))];
        let entries: Vec<FileEntry> = (0..6)
            .map(|index| stats_entry(&format!("file-{index}"), &range))
            .collect();
        let five: Vec<LayoutCandidate> = entries[..5]
            .iter()
            .map(|entry| candidate_with_row_groups(entry, &shape))
            .collect();
        let six: Vec<LayoutCandidate> = entries
            .iter()
            .map(|entry| candidate_with_row_groups(entry, &shape))
            .collect();

        let pass = CompactionPass::Routine;

        let waiting = select_group(&five, &["id".into()], pass, 6);
        let clique = select_group(&six, &["id".into()], pass, 6).unwrap();

        assert!(waiting.is_none());
        assert_eq!(clique.len(), 6);
    }

    /// A top-scoring pair whose clique tops out below the clique size is passed
    /// over for a lower-ranked seed that completes one.
    #[test]
    fn passes_a_capped_pair_over_for_a_full_clique() {
        let range = [("id", int_stat(500), int_stat(600))];
        let hot_first = stats_entry("hot-a", &range);
        let hot_second = stats_entry("hot-b", &range);
        let steps: Vec<FileEntry> = (0..6i64)
            .map(|step| {
                stats_entry(
                    &format!("step-{step}"),
                    &[("id", int_stat(step * 10), int_stat(step * 10 + 140))],
                )
            })
            .collect();
        let mut files = vec![
            candidate_with_row_groups(&hot_first, &[(500, 550), (550, 600)]),
            candidate_with_row_groups(&hot_second, &[(500, 550), (550, 600)]),
        ];
        let step_groups: Vec<Vec<(i64, i64)>> = (0..6i64)
            .map(|step| {
                (0..5)
                    .map(|group| (step * 10 + group * 28, step * 10 + (group + 1) * 28))
                    .collect()
            })
            .collect();
        for (entry, groups) in steps.iter().zip(&step_groups) {
            files.push(candidate_with_row_groups(entry, groups));
        }

        let group = select_group(&files, &["id".into()], CompactionPass::Routine, 6).unwrap();

        let mut paths = group_paths(&group);
        paths.sort_unstable();
        assert_eq!(
            paths,
            ["step-0", "step-1", "step-2", "step-3", "step-4", "step-5"]
        );
    }

    /// A final sweep takes the best pair's group whatever its size.
    #[test]
    fn final_sweep_takes_a_group_short_of_the_clique_size() {
        let shape = [(0, 50), (50, 100)];
        let range = [("id", int_stat(0), int_stat(100))];
        let entries: Vec<FileEntry> = (0..5)
            .map(|index| stats_entry(&format!("file-{index}"), &range))
            .collect();
        let files: Vec<LayoutCandidate> = entries
            .iter()
            .map(|entry| candidate_with_row_groups(entry, &shape))
            .collect();

        let group = select_group(&files, &["id".into()], CompactionPass::Final, 6).unwrap();

        assert_eq!(group.len(), 5);
    }

    /// A wide file and a narrow one inside its range: a pair below the
    /// contested bar.
    fn below_bar_entries() -> (FileEntry, FileEntry) {
        (
            stats_entry("wide", &[("id", int_stat(1), int_stat(100))]),
            stats_entry("narrow", &[("id", int_stat(3), int_stat(4))]),
        )
    }

    /// The candidates of [`below_bar_entries`], five row groups each.
    fn below_bar_candidates<'a>(
        wide: &'a FileEntry,
        narrow: &'a FileEntry,
    ) -> [LayoutCandidate<'a>; 2] {
        [
            candidate_with_row_groups(wide, &[(1, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(narrow, &[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4)]),
        ]
    }

    /// A routine round measures a pair below the contested bar and remembers
    /// nothing of it.
    #[test]
    fn a_routine_round_does_not_remember_a_pair_below_the_bar() {
        let (wide, narrow) = below_bar_entries();
        let files = below_bar_candidates(&wide, &narrow);
        let mut cache = ListedPartition::default();

        let read = observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        assert_eq!(read, ["wide", "narrow"]);
        assert_eq!(remembered_pairs(&cache), 0);
        assert!(
            cache
                .select_group(&HashSet::new(), CompactionPass::Routine, 6)
                .is_none()
        );
    }

    /// A final sweep measures again the pair a routine round did not
    /// remember, and merges it.
    #[test]
    fn a_final_sweep_measures_again_a_pair_below_the_bar_and_takes_it() {
        let (wide, narrow) = below_bar_entries();
        let files = below_bar_candidates(&wide, &narrow);
        let mut cache = ListedPartition::default();
        observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        let read = observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Final);
        let settled = cache.select_group(&HashSet::new(), CompactionPass::Final, 6);

        assert_eq!(read, ["wide", "narrow"]);
        assert_eq!(group_paths(&settled.unwrap()), ["narrow", "wide"]);
    }

    /// Once a final sweep is done with a partition, the pairs below the bar
    /// it held are gone, and no file is measured again.
    #[test]
    fn a_final_sweep_leaves_no_pair_below_the_bar_behind() {
        let (wide, narrow) = below_bar_entries();
        let files = below_bar_candidates(&wide, &narrow);
        let mut cache = ListedPartition::default();
        observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Final);

        cache.forget_pairs_below_bar();
        let read_by_the_next_round =
            observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        assert_eq!(remembered_pairs(&cache), 0);
        assert!(read_by_the_next_round.is_empty());
    }

    /// A second routine round over the files the first one measured reads
    /// no footer.
    #[test]
    fn a_second_routine_round_over_unchanged_files_measures_nothing() {
        let files = layered_candidates();
        let mut cache = ListedPartition::default();
        let read_first =
            observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        let read_second =
            observe_candidates(&mut cache, &files, &["id".into()], CompactionPass::Routine);

        assert!(!read_first.is_empty());
        assert!(read_second.is_empty());
    }

    /// A round after one file was added measures the pairs that file is in
    /// and no other: the pair measured before is not read again.
    #[test]
    fn a_round_after_one_new_file_measures_only_its_pairs() {
        let shape = [(0, 50), (50, 100)];
        let far_shape = [(500, 550), (550, 600)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let far = stats_entry("far", &[("id", int_stat(500), int_stat(600))]);
        let added = stats_entry("added", &[("id", int_stat(500), int_stat(600))]);
        let mut cache = ListedPartition::default();
        let before = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&far, &far_shape),
        ];
        observe_candidates(&mut cache, &before, &["id".into()], CompactionPass::Routine);
        let after = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&far, &far_shape),
            candidate_with_row_groups(&added, &far_shape),
        ];

        let mut read =
            observe_candidates(&mut cache, &after, &["id".into()], CompactionPass::Routine);

        read.sort();
        assert_eq!(read, ["added", "far"]);
        assert_eq!(remembered_pairs(&cache), 2);
    }

    /// A file that left the partition is forgotten, with its pairs.
    #[test]
    fn a_file_that_is_gone_is_forgotten() {
        let shape = [(0, 50), (50, 100)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let mut cache = ListedPartition::default();
        let both = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
        ];
        observe_candidates(&mut cache, &both, &["id".into()], CompactionPass::Routine);
        let survivor = [candidate_with_row_groups(&first, &shape)];

        observe_candidates(
            &mut cache,
            &survivor,
            &["id".into()],
            CompactionPass::Routine,
        );

        assert_eq!(remembered_pairs(&cache), 0);
    }

    /// A file in a pair is named, wherever the pair names it, by the one
    /// copy of its path held for it as a paired file.
    #[test]
    fn a_paired_file_is_named_by_the_one_copy_of_its_path() {
        let shape = [(0, 50), (50, 100)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let mut cache = ListedPartition::default();
        let both = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
        ];

        observe_candidates(&mut cache, &both, &["id".into()], CompactionPass::Routine);

        let is_the_one_copy = |path: &Arc<ObjectPath>| {
            cache
                .paired
                .get_key_value(path)
                .is_some_and(|(name, _)| Arc::ptr_eq(name, path))
        };
        assert_eq!(remembered_pairs(&cache), 1);
        assert!(
            cache
                .paired
                .values()
                .all(|file| file.partners.keys().all(is_the_one_copy))
        );
    }

    /// A partition left without a file a layout merge may rewrite leaves
    /// nothing remembered.
    #[test]
    fn nothing_is_remembered_of_a_partition_left_without_a_candidate() {
        let shape = [(0, 50), (50, 100)];
        let first = stats_entry("first", &[("id", int_stat(0), int_stat(100))]);
        let second = stats_entry("second", &[("id", int_stat(0), int_stat(100))]);
        let mut cache = ListedPartition::default();
        let both = [
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
        ];
        observe_candidates(&mut cache, &both, &["id".into()], CompactionPass::Routine);
        let none: [LayoutCandidate<'_>; 0] = [];

        observe_candidates(&mut cache, &none, &["id".into()], CompactionPass::Routine);

        assert!(cache.paired.is_empty());
    }

    /// A hub's second satellite is below the bar, so no clique of three
    /// exists; the final sweep's connected growth takes the hub with both
    /// satellites in one group.
    #[test]
    fn final_sweep_takes_a_hub_with_its_satellites() {
        let hub = stats_entry("hub", &[("id", int_stat(0), int_stat(100))]);
        let near = stats_entry("near", &[("id", int_stat(0), int_stat(40))]);
        let far = stats_entry("far", &[("id", int_stat(41), int_stat(80))]);
        let files = [
            candidate_with_row_groups(&hub, &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(&near, &[(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]),
            candidate_with_row_groups(&far, &[(41, 48), (48, 56), (56, 64), (64, 72), (72, 80)]),
        ];

        let clique = select_group(&files, &["id".into()], CompactionPass::Routine, 3);
        let settled = select_group(&files, &["id".into()], CompactionPass::Final, 6).unwrap();

        assert!(clique.is_none());
        assert_eq!(group_paths(&settled), ["hub", "near", "far"]);
    }

    /// Pairs that merely touch at a boundary value and pairs holding one
    /// identical key value are never selected, in either mode.
    #[test]
    fn keeps_touching_and_identical_value_pairs_apart_in_every_mode() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(10), int_stat(20))]);
        let full = sized_entry("full", 7, 7, 70);
        let heavy = sized_entry("heavy", 7, 7, 70);
        let touching = [
            candidate_with_row_groups(&low, &[(0, 5), (5, 10)]),
            candidate_with_row_groups(&high, &[(10, 15), (15, 20)]),
        ];
        let stacked = [candidate(&full, &["id"]), candidate(&heavy, &["id"])];

        let routine = CompactionPass::Routine;
        let final_sweep = CompactionPass::Final;

        let routine_touch = select_group(&touching, &["id".into()], routine, 6);
        let settled_touch = select_group(&touching, &["id".into()], final_sweep, 6);
        let routine_stack = select_group(&stacked, &["id".into()], routine, 6);
        let settled_stack = select_group(&stacked, &["id".into()], final_sweep, 6);

        assert!(routine_touch.is_none());
        assert!(settled_touch.is_none());
        assert!(routine_stack.is_none());
        assert!(settled_stack.is_none());
    }

    /// Every pair that clears the gates is counted, not only the best one,
    /// and files sharing no range add none.
    #[test]
    fn count_eligible_pairs_counts_every_qualifying_pair() {
        let range = [("id", int_stat(0), int_stat(100))];
        let first = stats_entry("first", &range);
        let second = stats_entry("second", &range);
        let third = stats_entry("third", &range);
        let apart = stats_entry("apart", &[("id", int_stat(200), int_stat(300))]);
        let files = [
            candidate_with_row_groups(&first, &[(0, 100)]),
            candidate_with_row_groups(&second, &[(0, 100)]),
            candidate_with_row_groups(&third, &[(0, 100)]),
            candidate_with_row_groups(&apart, &[(200, 300)]),
        ];

        let mut cache = ListedPartition::default();
        select_best_pair(&mut cache, &files, &["id".into()], &HashSet::new());

        assert_eq!(eligible_pairs(&cache), 3);
    }

    #[test]
    fn selection_picks_the_highest_overlap_pair_among_many_files() {
        let files = layered_candidates();

        let (share, left, right) = select_pair(&files, &["id".into()]).unwrap();

        // Two of wide's five groups lie within shifted's range and two of
        // shifted's within wide's; no other pair qualifies at all.
        assert!((share - 0.4).abs() < 1e-12);
        let mut pair = [left.as_str(), right.as_str()];
        pair.sort();
        assert_eq!(pair, ["shifted", "wide"]);
    }

    #[test]
    fn disjoint_files_produce_no_pair() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(20), int_stat(30))]);

        let best = select_pair(
            &[candidate(&low, &["id"]), candidate(&high, &["id"])],
            &["id".into()],
        );

        assert!(best.is_none());
    }

    #[test]
    fn files_without_rows_are_not_selected() {
        let mut empty = stats_entry("empty", &[("id", int_stat(0), int_stat(100))]);
        Arc::get_mut(empty.stats.as_mut().unwrap())
            .unwrap()
            .num_records = Some(0);
        let full = stats_entry("full", &[("id", int_stat(0), int_stat(100))]);
        let shape = [(0, 50), (50, 100)];
        let files = [
            candidate_with_row_groups(&empty, &shape),
            candidate_with_row_groups(&full, &shape),
        ];

        assert!(select_pair(&files, &["id".into()]).is_none());
    }

    #[test]
    fn files_with_inverted_leading_bounds_are_not_selected() {
        let inverted = stats_entry("inverted", &[("id", int_stat(100), int_stat(0))]);
        let covering = stats_entry("covering", &[("id", int_stat(-10), int_stat(110))]);
        let files = [
            LayoutCandidate {
                entry: &inverted,
                row_group_ranges: vec![vec![None]],
            },
            LayoutCandidate {
                entry: &covering,
                row_group_ranges: vec![vec![None]],
            },
        ];

        assert!(select_pair(&files, &["id".into()]).is_none());
    }

    #[test]
    fn contained_singleton_never_attracts_a_merge() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let singleton = stats_entry("singleton", &[("id", int_stat(50), int_stat(50))]);
        let partner = stats_entry("partner", &[("id", int_stat(55), int_stat(170))]);
        let wide_groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)];
        let singleton_groups = [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50)];
        let partner_groups = [(55, 65), (65, 85), (90, 110), (110, 140), (140, 170)];

        let alone = select_pair(
            &[
                candidate_with_row_groups(&wide, &wide_groups),
                candidate_with_row_groups(&singleton, &singleton_groups),
            ],
            &["id".into()],
        );
        let (share, left, right) = select_pair(
            &[
                candidate_with_row_groups(&wide, &wide_groups),
                candidate_with_row_groups(&singleton, &singleton_groups),
                candidate_with_row_groups(&partner, &partner_groups),
            ],
            &["id".into()],
        )
        .unwrap();

        // The singleton sits in one of wide's five groups, too few to rewrite.
        assert!(alone.is_none());
        assert!((share - 0.4).abs() < 1e-12);
        let mut pair = [left.as_str(), right.as_str()];
        pair.sort();
        assert_eq!(pair, ["partner", "wide"]);
    }

    #[test]
    fn row_group_requirement_is_over_a_fifth_of_the_groups() {
        assert_eq!(required_contested_row_groups(1), 1);
        assert_eq!(required_contested_row_groups(4), 1);
        assert_eq!(required_contested_row_groups(5), 2);
        assert_eq!(required_contested_row_groups(9), 2);
        assert_eq!(required_contested_row_groups(10), 3);
        assert_eq!(required_contested_row_groups(50), 11);
    }

    #[test]
    fn narrow_file_touching_one_row_group_is_not_selected() {
        let wide = stats_entry("wide", &[("id", int_stat(1), int_stat(100))]);
        let narrow = stats_entry("narrow", &[("id", int_stat(3), int_stat(4))]);
        let files = [
            candidate_with_row_groups(&wide, &[(1, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(&narrow, &[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4)]),
        ];

        let best = select_pair(&files, &["id".into()]);

        assert!(best.is_none());
    }

    #[test]
    fn interleaved_row_groups_meet_the_requirement_on_both_sides() {
        let left = stats_entry("left", &[("id", int_stat(1), int_stat(50))]);
        let right = stats_entry("right", &[("id", int_stat(25), int_stat(75))]);
        let files = [
            candidate_with_row_groups(&left, &[(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]),
            candidate_with_row_groups(&right, &[(25, 35), (35, 45), (45, 55), (55, 65), (65, 75)]),
        ];

        let (share, _, _) = select_pair(&files, &["id".into()]).unwrap();

        // Two of each file's five groups lie within the other's range.
        assert!((share - 0.4).abs() < 1e-12);
    }

    #[test]
    fn row_groups_without_bounds_count_as_contested() {
        let wide = stats_entry("wide", &[("id", int_stat(1), int_stat(100))]);
        let narrow = stats_entry("narrow", &[("id", int_stat(3), int_stat(4))]);
        let files = [
            LayoutCandidate {
                entry: &wide,
                row_group_ranges: vec![vec![None; 5]],
            },
            candidate_with_row_groups(&narrow, &[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4)]),
        ];

        let (share, _, _) = select_pair(&files, &["id".into()]).unwrap();

        assert!((share - 1.0).abs() < 1e-12);
    }

    /// A file's statistics come from its table's metadata and its row groups'
    /// from its footer, which decode a string column to `Utf8` and `Utf8View`. The
    /// two still have to compare.
    #[test]
    fn row_groups_are_counted_against_differently_encoded_file_stats() {
        let mut left = stats_entry("left", &[("id", utf8_stat("a"), utf8_stat("m"))]);
        left.file.size = 1;
        let right = stats_entry("right", &[("id", utf8_stat("g"), utf8_stat("z"))]);
        let files = [
            candidate_with_string_row_groups(
                &left,
                &[("a", "c"), ("d", "f"), ("g", "i"), ("j", "l"), ("l", "m")],
            ),
            candidate_with_string_row_groups(
                &right,
                &[("g", "k"), ("k", "m"), ("o", "s"), ("s", "w"), ("w", "z")],
            ),
        ];

        let (share, _, _) = select_pair(&files, &["id".into()]).unwrap();

        // Three of left's groups lie within right's range and two of right's
        // within left's, none of which is visible while the two encodings are
        // held apart.
        assert!((share - 0.4).abs() < 1e-12);
    }

    /// A string sort column is ranked by the same contested row groups as
    /// any other type: the pair sharing more of its groups wins.
    #[test]
    fn string_ranges_rank_by_contested_row_groups() {
        let left = stats_entry("left", &[("customer", string_stat("a"), string_stat("m"))]);
        let heavy = stats_entry("heavy", &[("customer", string_stat("c"), string_stat("z"))]);
        let light = stats_entry("light", &[("customer", string_stat("k"), string_stat("z"))]);
        let files = [
            candidate_with_string_row_groups(
                &left,
                &[("a", "c"), ("d", "f"), ("g", "i"), ("j", "l"), ("l", "m")],
            ),
            candidate_with_string_row_groups(
                &heavy,
                &[("c", "e"), ("f", "j"), ("k", "o"), ("p", "t"), ("u", "z")],
            ),
            candidate_with_string_row_groups(
                &light,
                &[("k", "m"), ("m", "p"), ("q", "s"), ("t", "v"), ("w", "z")],
            ),
        ];

        let (share, first, second) = select_pair(&files, &["customer".into()]).unwrap();

        // Three of heavy's five groups lie within light's range and all of
        // light's within heavy's, against the two of heavy's that lie within
        // left's.
        assert!((share - 0.6).abs() < 1e-12);
        let mut pair = [first.as_str(), second.as_str()];
        pair.sort();
        assert_eq!(pair, ["heavy", "light"]);
    }

    /// String bounds of unequal length sweep in key order: a prefix orders
    /// before its extensions, so the file ending at "b" still reaches the
    /// file starting at "ab" and the pair is found.
    #[test]
    fn sweeps_string_bounds_of_unequal_length_in_key_order() {
        let short = stats_entry("short", &[("customer", string_stat("a"), string_stat("b"))]);
        let long = stats_entry(
            "long",
            &[("customer", string_stat("ab"), string_stat("bz"))],
        );
        let far = stats_entry("far", &[("customer", string_stat("c"), string_stat("cz"))]);
        let files = [
            candidate_with_string_row_groups(&far, &[("c", "cc"), ("cc", "cz")]),
            candidate_with_string_row_groups(&long, &[("ab", "b"), ("b", "bb"), ("bb", "bz")]),
            candidate_with_string_row_groups(&short, &[("a", "aa"), ("aa", "ab"), ("ab", "b")]),
        ];

        let (share, first, second) = select_pair(&files, &["customer".into()]).unwrap();

        let mut pair = [first.as_str(), second.as_str()];
        pair.sort();
        assert_eq!(pair, ["long", "short"]);
        assert!((share - 1.0 / 3.0).abs() < 1e-12);
    }

    /// A cache that remembers exactly the given bar-clearing pairs.
    fn cache_from_pairs(pairs: &[(&str, &str, f64)]) -> OverlapCache {
        let mut cache = OverlapCache::default();
        for (left, right, contested_share) in pairs {
            for (path, partner) in [(left, right), (right, left)] {
                let name = Arc::new(ObjectPath::new(*path));
                let file = cache.paired.entry(name).or_insert_with(|| PairedFile {
                    record: build_record(&stats_entry(path, &[]), &[]),
                    partners: HashMap::new(),
                });
                let overlap = PairOverlap {
                    contested_share: *contested_share,
                    clears_bar: true,
                };
                file.partners
                    .insert(Arc::new(ObjectPath::new(*partner)), overlap);
            }
        }
        cache
    }

    /// Every member of the clique has a stronger partner outside it, so
    /// taking the best partner first dead-ends from every seed and the
    /// search has to back up to find the clique.
    #[test]
    fn backs_up_when_the_best_partner_is_outside_the_clique() {
        let cache = cache_from_pairs(&[
            ("hub", "b", 0.5),
            ("hub", "c", 0.5),
            ("b", "c", 0.5),
            ("hub", "distractor-a", 0.9),
            ("b", "distractor-b", 0.9),
            ("c", "distractor-c", 0.9),
        ]);

        let group = cache
            .select_group(&HashSet::new(), CompactionPass::Routine, 3)
            .unwrap();

        let mut paths = group_paths(&group);
        paths.sort_unstable();
        assert_eq!(paths, ["b", "c", "hub"]);
    }
}
