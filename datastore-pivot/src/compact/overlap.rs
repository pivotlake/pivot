use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_cast::cast;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;

use super::CompactionPass;
use crate::TableFile;
use crate::manifest::DeltaFileEntry;
use object_storage::ObjectPath;

/// A layout-optimization candidate: a file's manifest entry beside its row
/// groups' min/max on every sort column, in the table's sort order. A pair is
/// measured on the column that tells the two files apart, which is not always
/// the first: files holding one identical value on the leading keys are
/// ordered by the next one, and that is where their row groups have to be
/// compared. `None` marks a row group whose footer recorded no bounds for a
/// column; its rows can lie anywhere in the file's range, so the
/// contested-group count includes it.
pub(super) struct LayoutCandidate<'a> {
    pub(super) entry: &'a DeltaFileEntry,
    pub(super) row_group_ranges: Vec<Vec<Option<(ArrayRef, ArrayRef)>>>,
}

impl<'a> LayoutCandidate<'a> {
    /// Extract each of the file's row groups' min/max on each of `sort_by`
    /// from its footer metadata.
    pub(super) fn from_table_file(file: &'a TableFile, sort_by: &[String]) -> Self {
        let row_group_ranges = sort_by
            .iter()
            .map(|sort_column| {
                file.row_groups
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
            .collect();
        Self {
            entry: &file.entry,
            row_group_ranges,
        }
    }

    /// The file's row group count, which every sort column's bounds share.
    fn row_group_count(&self) -> usize {
        self.row_group_ranges
            .first()
            .map_or(0, |per_column| per_column.len())
    }

    /// The file's min/max on `column` from its manifest statistics.
    fn column_bounds(&self, column: &str) -> Option<(&ArrayRef, &ArrayRef)> {
        let stats = self.entry.stats.as_ref()?;
        Some((stats.min_values.get(column)?, stats.max_values.get(column)?))
    }

    /// Whether this file's maximum lies strictly before `other`'s minimum
    /// on the sort key, so the two share no stretch of it: the first column
    /// that separates the bounds decides, a tie defers to the next column,
    /// and a missing, null, or incomparable bound keeps the pair together.
    fn ends_before(&self, other: &LayoutCandidate, sort_by: &[String]) -> bool {
        for column in sort_by {
            let (Some((_, self_max)), Some((other_min, _))) =
                (self.column_bounds(column), other.column_bounds(column))
            else {
                return false;
            };
            match compare(self_max, other_min) {
                Some(Ordering::Less) => return true,
                Some(Ordering::Equal) => continue,
                _ => return false,
            }
        }
        false
    }
}

/// Order two files by their minimums on the sort key, column after column
/// the way rows order; a missing or null bound orders first. Ties break by
/// path only to keep the sweep deterministic.
fn compare_minimums(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    sort_by: &[String],
) -> Ordering {
    for column in sort_by {
        let order = match (known_minimum(left, column), known_minimum(right, column)) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(left_min), Some(right_min)) => {
                compare(left_min, right_min).unwrap_or(Ordering::Equal)
            }
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    left.entry
        .file
        .path
        .as_str()
        .cmp(right.entry.file.path.as_str())
}

/// The file's recorded, non-null minimum on `column`.
fn known_minimum<'a>(candidate: &'a LayoutCandidate, column: &str) -> Option<&'a ArrayRef> {
    let (min, _) = candidate.column_bounds(column)?;
    (!min.is_null(0)).then_some(min)
}

fn compare(left: &ArrayRef, right: &ArrayRef) -> Option<Ordering> {
    if left.len() != 1 || right.len() != 1 || left.is_null(0) || right.is_null(0) {
        return None;
    }
    // One column's bounds reach selection from two places that need not decode
    // them alike: a file's own statistics come from the table log, its row
    // groups' from its footer, and the same logical type can arrive as `Utf8`
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

/// A file without statistics, rows, or ordered leading bounds cannot
/// participate in a layout pair.
fn is_measurable(candidate: &LayoutCandidate, leading_column: &str) -> bool {
    let Some(stats) = candidate.entry.stats.as_ref() else {
        return false;
    };
    if !stats.num_records.is_some_and(|records| records > 0) {
        return false;
    }
    let Some((min, max)) = candidate.column_bounds(leading_column) else {
        return false;
    };
    compare(min, max).is_some_and(|order| order != Ordering::Greater)
}

/// How much two files' rows interleave, from their row-group bounds.
#[derive(Clone, Copy)]
pub(super) struct PairOverlap {
    /// The smaller of the two files' contested row-group shares. Zero when
    /// the ranges overlap but no row group of either file nests in the
    /// other's range.
    pub(super) contested_share: f64,
    /// Whether the pair is worth a routine rewrite: both files have more
    /// than [`CONTESTED_ROW_GROUP_PERCENT_BAR`] percent of their row groups
    /// contested. A final sweep ignores this and takes any measured pair.
    pub(super) clears_bar: bool,
}

/// Measure how `left` and `right` interleave, or `None` for a pair no pass
/// may ever merge.
///
/// The pair is decided on the first sort column where the two files are not
/// the same single value. Ranges that are disjoint or touch at one value
/// share no stretch of the key and never merge, and neither do files holding
/// one identical key value throughout, which have nothing to sort. Both
/// rules hold in every pass; they are what makes compaction terminate. A
/// bound that cannot be compared keeps the pair rather than dropping it.
pub(super) fn measure_pair(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    sort_by: &[String],
) -> Option<PairOverlap> {
    let leading_column = sort_by.first()?;
    if !is_measurable(left, leading_column) || !is_measurable(right, leading_column) {
        return None;
    }
    let left_stats = left.entry.stats.as_ref()?;
    let right_stats = right.entry.stats.as_ref()?;
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
        return overlap_from_row_groups(left, right, column, name);
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
    let left_range = left.column_bounds(name)?;
    let right_range = right.column_bounds(name)?;
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

/// One table's pair overlaps, remembered between rounds.
///
/// An overlap depends only on the two files' footers and manifest stats, so
/// it is measured once: a round measures the files it has not seen before
/// against the files of their partition they can interleave with, and
/// forgets the files that are gone. Every pair sharing a stretch of the key
/// is kept under both of its files, so a file's partners are one lookup
/// away. Paths are shared, so each is stored once however many pairs
/// contain it.
#[derive(Default)]
pub(super) struct OverlapCache {
    /// One shared copy of each live file's path, the handle partner lists
    /// are keyed by.
    known_paths: HashSet<Arc<ObjectPath>>,
    /// Each live file's remembered partners and how far they overlap it.
    partners: HashMap<Arc<ObjectPath>, HashMap<Arc<ObjectPath>, PairOverlap>>,
}

impl OverlapCache {
    /// Bring the cache up to date with `partitions`, each one partition's
    /// candidates: measure the files no earlier round has seen against the
    /// files of their partition they can interleave with, and forget the
    /// files that are gone.
    ///
    /// Each partition is swept in order of minimum on the whole sort key,
    /// keeping the files whose maximum has not yet been passed; only those
    /// can share a stretch of the key with the current file. A new file is
    /// measured against every file in reach, a file seen before only
    /// against the new ones, so a pass measures only pairs with a new file,
    /// and the first observation of a table measures its overlapping pairs
    /// rather than all pairs. Reach is trimmed only when a new file comes
    /// up: a file dropped there ends before every later file too, and a
    /// known file measures so few pairs that a stale one costs nothing.
    pub(super) fn observe(&mut self, partitions: &[&[LayoutCandidate<'_>]], sort_by: &[String]) {
        let Some(leading_column) = sort_by.first() else {
            return;
        };
        let mut live = HashSet::new();
        let mut new_paths = HashSet::new();
        for candidate in partitions.iter().flat_map(|partition| partition.iter()) {
            let path = &candidate.entry.file.path;
            live.insert(path.as_str());
            if !self.known_paths.contains(path) {
                let shared = Arc::new(path.clone());
                self.partners.entry(shared.clone()).or_default();
                self.known_paths.insert(shared);
                new_paths.insert(path.as_str());
            }
        }
        self.retain_live_files(&live);

        for partition in partitions {
            let mut ordered: Vec<&LayoutCandidate<'_>> = partition
                .iter()
                .filter(|candidate| is_measurable(candidate, leading_column))
                .collect();
            ordered.sort_unstable_by(|left, right| compare_minimums(left, right, sort_by));
            // Files whose range the sweep has not passed, each flagged new.
            let mut in_reach: Vec<(&LayoutCandidate<'_>, bool)> = Vec::new();
            for candidate in ordered {
                let candidate_is_new = new_paths.contains(candidate.entry.file.path.as_str());
                let candidate_path = self.interned(&candidate.entry.file.path);
                if candidate_is_new {
                    in_reach.retain(|(partner, _)| !partner.ends_before(candidate, sort_by));
                }
                for (partner, partner_is_new) in &in_reach {
                    if !candidate_is_new && !partner_is_new {
                        continue;
                    }
                    let Some(overlap) = measure_pair(partner, candidate, sort_by) else {
                        continue;
                    };
                    let partner_path = self.interned(&partner.entry.file.path);
                    self.partners
                        .entry(partner_path.clone())
                        .or_default()
                        .insert(candidate_path.clone(), overlap);
                    self.partners
                        .entry(candidate_path.clone())
                        .or_default()
                        .insert(partner_path, overlap);
                }
                in_reach.push((candidate, candidate_is_new));
            }
        }
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
            .partners
            .iter()
            .filter(|(path, _)| !reserved.contains(path.as_ref()))
            .filter_map(|(path, partners)| {
                let (count, best) = partners
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
        let candidates: Vec<(&Arc<ObjectPath>, f64)> = self.partners[seed]
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
            let newest_partners = &self.partners[newest];
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
        let mut partners: Vec<(&Arc<ObjectPath>, f64)> = self.partners[seed]
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

    /// The shared copy of a live file's path, the handle partner lists are
    /// keyed by.
    fn interned(&self, path: &ObjectPath) -> Arc<ObjectPath> {
        self.known_paths
            .get(path)
            .expect("the path is a live file")
            .clone()
    }

    /// Keep only `live` files and the pairs between two of them.
    fn retain_live_files(&mut self, live: &HashSet<&str>) {
        self.known_paths.retain(|path| live.contains(path.as_str()));
        self.partners.retain(|path, _| live.contains(path.as_str()));
        for partners in self.partners.values_mut() {
            partners.retain(|path, _| live.contains(path.as_str()));
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

    use arrow_array::{ArrayRef, Int64Array, StringArray, StringViewArray};

    use super::*;
    use crate::manifest::FileStats;
    use object_storage::{FileRef, ObjectPath};

    /// The best remembered bar-clearing pair in `cache` among `partitions`,
    /// measuring the files no earlier round has seen: the pair a routine
    /// round seeds a clique from, so tests can probe the scoring directly.
    /// Reserved files' pairs are passed over; ties go to the
    /// lexicographically first pair.
    fn select_best_pair(
        cache: &mut OverlapCache,
        partitions: &[&[LayoutCandidate<'_>]],
        sort_by: &[String],
        reserved: &HashSet<ObjectPath>,
    ) -> Option<(f64, Arc<ObjectPath>, Arc<ObjectPath>)> {
        cache.observe(partitions, sort_by);
        let (left, right, overlap) = cache
            .partners
            .iter()
            .flat_map(|(left, partners)| {
                partners
                    .iter()
                    .map(move |(right, overlap)| (left, right, overlap))
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

    /// Every pair the cache remembers, bar or no bar.
    fn remembered_pairs(cache: &OverlapCache) -> usize {
        cache.partners.values().map(HashMap::len).sum::<usize>() / 2
    }

    /// Remembered pairs that clear the contested bar, reserved or not: the
    /// pairs a routine round may merge.
    fn eligible_pairs(cache: &OverlapCache) -> usize {
        cache
            .partners
            .values()
            .flat_map(|partners| partners.values())
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
            &mut OverlapCache::default(),
            &[files],
            sort_by,
            &HashSet::new(),
        )
    }

    /// Select a group over one partition with nothing remembered and nothing
    /// reserved.
    fn select_group(
        files: &[LayoutCandidate<'_>],
        sort_by: &[String],
        pass: CompactionPass,
        clique_size: usize,
    ) -> Option<Vec<Arc<ObjectPath>>> {
        let mut cache = OverlapCache::default();
        cache.observe(&[files], sort_by);
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

    fn stats_entry(path: &str, columns: &[(&str, ArrayRef, ArrayRef)]) -> DeltaFileEntry {
        let mut entry = DeltaFileEntry::new(FileRef {
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
    fn sized_entry(path: &str, min: i64, max: i64, size: u64) -> DeltaFileEntry {
        let mut entry = stats_entry(path, &[("id", int_stat(min), int_stat(max))]);
        entry.file.size = size;
        entry
    }

    fn utf8_stat(value: &str) -> ArrayRef {
        Arc::new(StringArray::from(vec![value]))
    }

    /// A candidate whose file has one row group spanning its whole range on
    /// each of `sort_by`, the layout a small single-group parquet has.
    fn candidate<'a>(entry: &'a DeltaFileEntry, sort_by: &[&str]) -> LayoutCandidate<'a> {
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
        entry: &'a DeltaFileEntry,
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
        entry: &'a DeltaFileEntry,
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
        entry: &'a DeltaFileEntry,
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
        static WIDE: std::sync::OnceLock<DeltaFileEntry> = std::sync::OnceLock::new();
        static SHIFTED: std::sync::OnceLock<DeltaFileEntry> = std::sync::OnceLock::new();
        static CONTAINED: std::sync::OnceLock<DeltaFileEntry> = std::sync::OnceLock::new();
        static DISJOINT: std::sync::OnceLock<DeltaFileEntry> = std::sync::OnceLock::new();
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
        let mut cache = OverlapCache::default();
        let before = vec![
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&survivor, &survivor_shape),
        ];
        select_best_pair(&mut cache, &[&before], &["id".into()], &HashSet::new());
        let merged = stats_entry("merged", &[("id", int_stat(500), int_stat(600))]);
        let after = vec![
            candidate_with_row_groups(&survivor, &survivor_shape),
            candidate_with_row_groups(&merged, &survivor_shape),
        ];

        let (share, left, right) =
            select_best_pair(&mut cache, &[&after], &["id".into()], &HashSet::new()).unwrap();

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
        let mut cache = OverlapCache::default();
        let (_, left, right) =
            select_best_pair(&mut cache, &[&files], &["id".into()], &HashSet::new()).unwrap();
        let reserved = HashSet::from([(*left).clone(), (*right).clone()]);

        let (_, next_left, next_right) =
            select_best_pair(&mut cache, &[&files], &["id".into()], &reserved).unwrap();

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
        let mut cache = OverlapCache::default();
        cache.observe(&[&files], &["id".into()]);

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
        let entries: Vec<DeltaFileEntry> = (0..6)
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
        let steps: Vec<DeltaFileEntry> = (0..6i64)
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
        let entries: Vec<DeltaFileEntry> = (0..5)
            .map(|index| stats_entry(&format!("file-{index}"), &range))
            .collect();
        let files: Vec<LayoutCandidate> = entries
            .iter()
            .map(|entry| candidate_with_row_groups(entry, &shape))
            .collect();

        let group = select_group(&files, &["id".into()], CompactionPass::Final, 6).unwrap();

        assert_eq!(group.len(), 5);
    }

    /// A pair below the contested bar is remembered for the final sweep but
    /// never selected by a routine round, which counts no eligible pair.
    #[test]
    fn remembers_a_below_bar_pair_for_the_final_sweep_only() {
        let wide = stats_entry("wide", &[("id", int_stat(1), int_stat(100))]);
        let narrow = stats_entry("narrow", &[("id", int_stat(3), int_stat(4))]);
        let files = [
            candidate_with_row_groups(&wide, &[(1, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(&narrow, &[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4)]),
        ];
        let mut cache = OverlapCache::default();
        cache.observe(&[&files], &["id".into()]);

        let routine = cache.select_group(&HashSet::new(), CompactionPass::Routine, 6);
        let settled = cache.select_group(&HashSet::new(), CompactionPass::Final, 6);

        assert!(routine.is_none());
        assert_eq!(eligible_pairs(&cache), 0);
        assert_eq!(group_paths(&settled.unwrap()), ["narrow", "wide"]);
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

        let mut cache = OverlapCache::default();
        select_best_pair(&mut cache, &[&files], &["id".into()], &HashSet::new());

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

    /// A file's statistics come from the table log and its row groups' from
    /// its footer, which decode a string column to `Utf8` and `Utf8View`. The
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
                let path = Arc::new(ObjectPath::new(*path));
                let partner = Arc::new(ObjectPath::new(*partner));
                cache.known_paths.insert(path.clone());
                cache.known_paths.insert(partner.clone());
                cache.partners.entry(path).or_default().insert(
                    partner,
                    PairOverlap {
                        contested_share: *contested_share,
                        clears_bar: true,
                    },
                );
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
