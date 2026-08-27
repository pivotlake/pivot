use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use arrow_array::{Array, ArrayRef, Datum};
use arrow_cast::cast;
use arrow_ord::cmp;

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
                        let min = statistics.min.as_ref()?.get().0.slice(0, 1);
                        let max = statistics.max.as_ref()?.get().0.slice(0, 1);
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

fn compare_same_type(left: &ArrayRef, right: &ArrayRef) -> Option<Ordering> {
    if cmp::eq(left, right).ok()?.value(0) {
        return Some(Ordering::Equal);
    }
    Some(if cmp::lt(left, right).ok()?.value(0) {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

/// Find the most contested pair among `candidates`. The caller has already
/// grouped these large-file candidates by partition. Only pairs with a
/// positive score are candidates, so `None` means no rewrite is worthwhile. A
/// sweep over the files sorted by their minimum on the first sort column
/// scores every pair whose ranges intersect, and only those: the cost is
/// proportional to the number of actually overlapping pairs, linearithmic
/// for a well-layered table and quadratic only while heavy overlap persists,
/// which is exactly when merges keep firing and shrinking it. A pair whose
/// rewrite cannot separate its overlap and cannot fit under `target_bytes` in
/// one file is skipped here, so it neither loops through futile rewrites nor
/// shadows a mergeable pair behind it, and so is a pair without enough
/// contested row groups on each side to make the rewrite restructure real
/// data.
///
/// The reference the remembered scores of [`LayoutScores`] are held to; a
/// round itself no longer sweeps.
#[cfg(test)]
pub(super) fn highest_scoring_pair<'a>(
    candidates: &[LayoutCandidate<'a>],
    sort_by: &[String],
    target_bytes: u64,
) -> Option<(f64, &'a DeltaFileEntry, &'a DeltaFileEntry)> {
    let first_column = sort_by.first()?;
    let mut ranges: Vec<SortColumnRange<'a, '_>> = candidates
        .iter()
        .filter_map(|candidate| extract_column_range(candidate, first_column))
        .collect();
    // Validated same-schema scalars always compare; an incomparable pair only
    // weakens the sweep order, never the scores.
    ranges.sort_by(|left, right| compare(left.min, right.min).unwrap_or(Ordering::Equal));

    let mut best: Option<(f64, &DeltaFileEntry, &DeltaFileEntry)> = None;
    let mut active: Vec<&SortColumnRange<'a, '_>> = Vec::new();
    for range in &ranges {
        // Stats are inclusive, so a partner whose maximum equals this minimum
        // still shares rows with it: the row groups meeting at that value are
        // contested, and equal singletons defer to the next sort column.
        active.retain(|partner| compare(partner.max, range.min) != Some(Ordering::Less));
        for partner in &active {
            let left = partner.candidate.entry;
            let right = range.candidate.entry;
            let Some(score) = gated_pair_score(partner, range, sort_by, target_bytes) else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|(best_score, _, _)| score > *best_score)
            {
                best = Some((score, left, right));
                // The score ceiling; no later pair can displace this one.
                if score >= 1.0 {
                    return best;
                }
            }
        }
        active.push(range);
    }
    best
}

/// The score a pair earns once it clears every gate: its combined size must
/// fit the output ceiling unless a rewrite can separate it, its rows must
/// interleave in sort order, and both files must have enough contested row
/// groups. `None` is a pair no round should merge.
///
/// The score itself is the pair's contested row-group fraction, which is
/// counted from the footers rather than estimated: a file's own fraction is
/// the share of its row groups whose bounds meet the other file's range, and
/// the pair takes the smaller of the two. That is the share of each file a
/// rewrite can restructure, and taking the smaller keeps a narrow file nested
/// inside a much wider one from outranking a pair that restructures both
/// sides.
fn gated_pair_score(
    left: &SortColumnRange,
    right: &SortColumnRange,
    sort_by: &[String],
    target_bytes: u64,
) -> Option<f64> {
    let left_entry = left.candidate.entry;
    let right_entry = right.candidate.entry;
    let combined_size = left_entry.file.size.saturating_add(right_entry.file.size);
    if combined_size > target_bytes && is_irreducible_pair(left_entry, right_entry, sort_by) {
        return None;
    }
    let decision = decide_pair(left_entry, right_entry, sort_by)?;
    let split = split_row_groups(left.candidate, right.candidate, &decision);
    if !split.meets_row_group_requirement() {
        return None;
    }
    Some(split.contested_fraction())
}

/// Which sort column tells a pair's two files apart, and their bounds on it.
enum PairDecision<'a> {
    /// Both files hold the same single value on every sort column: their
    /// layouts are the same one throughout.
    Identical,
    /// The first sort column on which the files are not that same single
    /// value, by its position in the sort order, with each file's range on it.
    Column {
        index: usize,
        left: (&'a ArrayRef, &'a ArrayRef),
        right: (&'a ArrayRef, &'a ArrayRef),
    },
}

/// Where a pair is decided, or `None` for a pair no round should merge: one
/// whose rows do not interleave, or one whose statistics do not carry a sort
/// column. Files are sorted by the whole key, so a column decides the pair
/// only once the two files are not the same single value on it: while both
/// hold one identical value there, their rows are ordered by the next key.
/// Statistics are inclusive, so ranges touching at a boundary value still
/// interleave.
fn decide_pair<'a>(
    left: &'a DeltaFileEntry,
    right: &'a DeltaFileEntry,
    sort_by: &[String],
) -> Option<PairDecision<'a>> {
    let (Some(left_stats), Some(right_stats)) = (&left.stats, &right.stats) else {
        return None;
    };
    for (index, column) in sort_by.iter().enumerate() {
        let (Some(left_min), Some(left_max), Some(right_min), Some(right_max)) = (
            left_stats.min_values.get(column),
            left_stats.max_values.get(column),
            right_stats.min_values.get(column),
            right_stats.max_values.get(column),
        ) else {
            return None;
        };
        let same_value = compare(left_min, left_max) == Some(Ordering::Equal)
            && compare(right_min, right_max) == Some(Ordering::Equal)
            && compare(left_min, right_min) == Some(Ordering::Equal);
        if same_value {
            continue;
        }
        if compare(left_max, right_min) == Some(Ordering::Less)
            || compare(right_max, left_min) == Some(Ordering::Less)
        {
            return None;
        }
        return Some(PairDecision::Column {
            index,
            left: (left_min, left_max),
            right: (right_min, right_max),
        });
    }
    Some(PairDecision::Identical)
}

/// Whether two files' ranges on the sweep column share any value. Stats are
/// inclusive, and an incomparable pair is kept rather than dropped, which is
/// what the sweep in [`highest_scoring_pair`] does too.
fn ranges_intersect(left: &SortColumnRange, right: &SortColumnRange) -> bool {
    compare(left.max, right.min) != Some(Ordering::Less)
        && compare(right.max, left.min) != Some(Ordering::Less)
}

/// The pair scores of one table, remembered between compaction rounds.
///
/// A pair's score depends only on the two files' footers and manifest stats,
/// which never change while the files exist, so a round need not score every
/// intersecting pair again: [`highest_scoring_pair`] does, and on a table of
/// two thousand heavily overlapping files that is over a million pairs of
/// Arrow comparisons every few seconds. Here a round scores only the files it
/// has not seen before (normally the two outputs of the last merge) against
/// the rest of their partition, forgets the pairs of files that are gone, and
/// takes the best of what it remembers. Only pairs that cleared the gates
/// with a positive score are kept: files are numbered as they appear, so a
/// remembered pair is two small ids and its score.
pub(super) struct LayoutScores {
    ids: HashMap<ObjectPath, u32>,
    next_id: u32,
    /// Files scored against every other file of their partition.
    scored: HashSet<u32>,
    /// Positive gated scores, keyed by the pair's ids as `low << 32 | high`.
    /// Single precision is plenty for ranking pairs, and it is what keeps a
    /// million remembered pairs small.
    pairs: HashMap<u64, f32>,
}

/// The pair a round picked, beside what it was picked out of.
pub(super) struct SelectedPair<'a> {
    pub(super) score: f64,
    pub(super) left: &'a DeltaFileEntry,
    pub(super) right: &'a DeltaFileEntry,
    pub(super) row_groups: SplitRowGroups,
    /// Overlapping pairs the round could have picked from, this one included.
    /// The pairs holding a file that a merge in flight is rewriting are not
    /// among them: this round cannot merge those.
    pub(super) selectable_pairs: usize,
}

/// A pair's row groups split by whether they overlap the other file's range on
/// the sweep column. Overlapping groups are the ones a rewrite can
/// restructure; the rest already sit outside the contested range, as do the
/// groups that only reach the other range's boundary value. A row group whose
/// footer recorded no bounds counts as overlapping, the same way the selection
/// gate counts it.
pub(super) struct SplitRowGroups {
    pub(super) left_overlapping: usize,
    pub(super) left_disjoint: usize,
    pub(super) right_overlapping: usize,
    pub(super) right_disjoint: usize,
}

impl SplitRowGroups {
    /// The share of the less contested file's row groups that a rewrite can
    /// restructure. A pair only reaches this once the gate has found contested
    /// groups on both sides, so neither file is without row groups here.
    fn contested_fraction(&self) -> f64 {
        let left = self.left_overlapping as f64
            / (self.left_overlapping + self.left_disjoint).max(1) as f64;
        let right = self.right_overlapping as f64
            / (self.right_overlapping + self.right_disjoint).max(1) as f64;
        left.min(right)
    }

    /// Whether both files have more than a fifth of their row groups
    /// overlapping the other, which is what makes a rewrite worthwhile.
    fn meets_row_group_requirement(&self) -> bool {
        let left_total = self.left_overlapping + self.left_disjoint;
        let right_total = self.right_overlapping + self.right_disjoint;
        self.left_overlapping >= required_contested_row_groups(left_total)
            && self.right_overlapping >= required_contested_row_groups(right_total)
    }
}

/// Count each file's row groups against the other file's range on the column
/// that decides the pair. Two files that are the same single value on every
/// sort column hold one another's rows throughout, so all of their groups are
/// contested.
fn split_row_groups(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    decision: &PairDecision,
) -> SplitRowGroups {
    let (left_overlapping, right_overlapping) = match decision {
        PairDecision::Identical => (left.row_group_count(), right.row_group_count()),
        PairDecision::Column {
            index,
            left: left_range,
            right: right_range,
        } => (
            count_contested_row_groups(left, *index, *right_range),
            count_contested_row_groups(right, *index, *left_range),
        ),
    };
    SplitRowGroups {
        left_overlapping,
        left_disjoint: left.row_group_count() - left_overlapping,
        right_overlapping,
        right_disjoint: right.row_group_count() - right_overlapping,
    }
}

impl LayoutScores {
    pub(super) fn new() -> Self {
        Self {
            ids: HashMap::new(),
            next_id: 0,
            scored: HashSet::new(),
            pairs: HashMap::new(),
        }
    }

    /// The highest-scoring pair among `partitions`, each one partition's
    /// large-file candidates, on the same terms as [`highest_scoring_pair`],
    /// skipping the pairs that hold a file some merge in flight is rewriting.
    pub(super) fn best_pair<'a>(
        &mut self,
        partitions: &[Vec<LayoutCandidate<'a>>],
        sort_by: &[String],
        target_bytes: u64,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<SelectedPair<'a>> {
        let first_column = sort_by.first()?;

        let mut entries: HashMap<u32, &'a DeltaFileEntry> = HashMap::new();
        for candidate in partitions.iter().flatten() {
            let id = self.id_of(&candidate.entry.file.path);
            entries.insert(id, candidate.entry);
        }
        self.forget_missing(&entries);

        // The ranges outlive the scoring so the pair that wins can report how
        // its row groups sit against each other. A file without a range on the
        // sweep column is in no pair, here or in what earlier rounds
        // remembered, so every remembered id has one.
        let mut ranges: HashMap<u32, SortColumnRange<'a, '_>> = HashMap::new();
        let mut partition_ids: Vec<Vec<u32>> = Vec::new();
        for partition in partitions {
            let mut ids: Vec<u32> = Vec::new();
            for candidate in partition {
                let Some(range) = extract_column_range(candidate, first_column) else {
                    continue;
                };
                let id = self.ids[&candidate.entry.file.path];
                ranges.insert(id, range);
                ids.push(id);
            }
            partition_ids.push(ids);
        }

        for ids in &partition_ids {
            // A pair of two new files is scored from the first one's side
            // only.
            let mut scored_this_round: HashSet<u32> = HashSet::new();
            for id in ids {
                if self.scored.contains(id) {
                    continue;
                }
                let range = &ranges[id];
                for partner_id in ids {
                    if partner_id == id || scored_this_round.contains(partner_id) {
                        continue;
                    }
                    let partner = &ranges[partner_id];
                    if !ranges_intersect(range, partner) {
                        continue;
                    }
                    if let Some(score) = gated_pair_score(range, partner, sort_by, target_bytes) {
                        self.pairs.insert(pair_key(*id, *partner_id), score as f32);
                    }
                }
                scored_this_round.insert(*id);
            }
            self.scored.extend(scored_this_round);
        }

        // A reserved file keeps its scores: the merge holding it can fail, and
        // the pairs it is in are as good then as they are now. It is the
        // selection that has to pass them over.
        let held: HashSet<u32> = reserved
            .iter()
            .filter_map(|path| self.ids.get(path).copied())
            .collect();
        let is_selectable = |key: u64| {
            let (low, high) = pair_ids(key);
            !held.contains(&low) && !held.contains(&high)
        };

        // Ties go to the pair numbered first, so the choice is stable across
        // rounds.
        let (&key, &score) = self
            .pairs
            .iter()
            .filter(|(key, _)| is_selectable(**key))
            .max_by(|(left_key, left), (right_key, right)| {
                left.total_cmp(right).then_with(|| right_key.cmp(left_key))
            })?;
        let (low, high) = pair_ids(key);
        // A remembered pair was decided once already, on statistics that do
        // not change while the two files exist, so this repeats that verdict
        // rather than reaching a new one.
        let decision = decide_pair(entries[&low], entries[&high], sort_by)?;
        Some(SelectedPair {
            score: f64::from(score),
            left: entries[&low],
            right: entries[&high],
            row_groups: split_row_groups(
                ranges[&low].candidate,
                ranges[&high].candidate,
                &decision,
            ),
            selectable_pairs: self.pairs.keys().filter(|key| is_selectable(**key)).count(),
        })
    }

    fn id_of(&mut self, path: &ObjectPath) -> u32 {
        *self.ids.entry(path.clone()).or_insert_with(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        })
    }

    /// Drop every file not among `live`, and every pair one of them was in.
    fn forget_missing(&mut self, live: &HashMap<u32, &DeltaFileEntry>) {
        self.ids.retain(|_, id| live.contains_key(id));
        self.scored.retain(|id| live.contains_key(id));
        self.pairs.retain(|&key, _| {
            let (low, high) = pair_ids(key);
            live.contains_key(&low) && live.contains_key(&high)
        });
    }
}

fn pair_key(left: u32, right: u32) -> u64 {
    let (low, high) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    (u64::from(low) << 32) | u64::from(high)
}

fn pair_ids(key: u64) -> (u32, u32) {
    ((key >> 32) as u32, key as u32)
}

/// A file's min/max on one sort column, kept beside its candidate so sweep
/// pairs can be gated on row groups and reported as entries.
struct SortColumnRange<'a, 'c> {
    candidate: &'c LayoutCandidate<'a>,
    min: &'c ArrayRef,
    max: &'c ArrayRef,
}

fn extract_column_range<'a, 'c>(
    candidate: &'c LayoutCandidate<'a>,
    column: &str,
) -> Option<SortColumnRange<'a, 'c>> {
    let stats = candidate.entry.stats.as_ref()?;
    if stats.num_records? <= 0 {
        return None;
    }
    let min = stats.min_values.get(column)?;
    let max = stats.max_values.get(column)?;
    if compare(min, max)? == Ordering::Greater {
        return None;
    }
    Some(SortColumnRange {
        candidate,
        min,
        max,
    })
}

/// Contested row groups a file with `total` row groups must have before a
/// merge is worthwhile: strictly more than a fifth of them. Five groups need
/// two, fifty need eleven. One contested group is all a single outlier row can
/// fabricate, so the requirement climbs past it once a file has five groups;
/// scaling with the count keeps a many-group file from qualifying on a sliver
/// of itself. This is the only bar a pair has to clear: a round merges every
/// pair that clears it, most contested first.
fn required_contested_row_groups(total: usize) -> usize {
    total / 5 + 1
}

/// How many of `file`'s row groups overlap `partner`'s range on the sort
/// column at `column`.
fn count_contested_row_groups(
    file: &LayoutCandidate,
    column: usize,
    partner: (&ArrayRef, &ArrayRef),
) -> usize {
    file.row_group_ranges[column]
        .iter()
        .filter(|bounds| {
            // A row group without recorded bounds can hold rows anywhere in
            // the file's range, so it counts as contested.
            let Some((min, max)) = bounds else {
                return true;
            };
            overlaps_range(min, max, partner)
        })
        .count()
}

/// Whether a row group's bounds overlap `partner`'s range rather than merely
/// reach it. Statistics are inclusive, so two ranges that meet at one value
/// touch without sharing a stretch of the key, and a rewrite that splits them
/// there has nothing to separate: only a shared stretch counts. Two files that
/// hold nothing but the same single value are the exception, having no width
/// to share and yet the same rows throughout.
fn overlaps_range(min: &ArrayRef, max: &ArrayRef, partner: (&ArrayRef, &ArrayRef)) -> bool {
    let (partner_min, partner_max) = partner;
    if compare(max, partner_min) == Some(Ordering::Greater)
        && compare(min, partner_max) == Some(Ordering::Less)
    {
        return true;
    }
    compare(min, max) == Some(Ordering::Equal)
        && compare(partner_min, partner_max) == Some(Ordering::Equal)
        && compare(min, partner_min) == Some(Ordering::Equal)
}

/// Whether no range rewrite can separate these files' overlapping rows. On
/// the first sort column where the two files are not the same singleton, the
/// pair is stuck when one file's entire range is a single value lying on the
/// other file's boundary: the merged rows then carry a run of that value at
/// the output's edge, and a split at row-group granularity reproduces a file
/// ending at that value beside the same singleton. A singleton strictly
/// inside the other range stays rewritable, because splitting around its run
/// yields ranges that touch at a point instead of overlapping. Files that are
/// the same singleton on every sort column are always stuck. Rewriting a
/// stuck pair whose combined bytes exceed the output ceiling loops forever;
/// under the ceiling one merged file removes the overlap outright.
pub(super) fn is_irreducible_pair(
    left: &DeltaFileEntry,
    right: &DeltaFileEntry,
    sort_by: &[String],
) -> bool {
    let (Some(left_stats), Some(right_stats)) = (&left.stats, &right.stats) else {
        return false;
    };
    if sort_by.is_empty() {
        return false;
    }
    for column in sort_by {
        let (Some(left_min), Some(left_max), Some(right_min), Some(right_max)) = (
            left_stats.min_values.get(column),
            left_stats.max_values.get(column),
            right_stats.min_values.get(column),
            right_stats.max_values.get(column),
        ) else {
            return false;
        };
        let left_singleton = compare(left_min, left_max) == Some(Ordering::Equal);
        let right_singleton = compare(right_min, right_max) == Some(Ordering::Equal);
        if left_singleton && right_singleton {
            if compare(left_min, right_min) == Some(Ordering::Equal) {
                continue;
            }
            return false;
        }
        let (singleton, other_min, other_max) = if left_singleton {
            (left_min, right_min, right_max)
        } else if right_singleton {
            (right_min, left_min, left_max)
        } else {
            return false;
        };
        return compare(singleton, other_min) == Some(Ordering::Equal)
            || compare(singleton, other_max) == Some(Ordering::Equal);
    }
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, StringArray, StringViewArray};

    use super::*;
    use crate::manifest::FileStats;
    use object_storage::{FileRef, ObjectPath};

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

        let (overlap, _, _) = highest_scoring_pair(&files, &sort_by, u64::MAX).unwrap();

        // Three of each file's five groups reach into the other's id range,
        // which the shared region says nothing about.
        assert!((overlap - 0.6).abs() < 1e-12);
        assert!(
            highest_scoring_pair(&files, &["region".into(), "missing".into()], u64::MAX).is_none()
        );
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

        let best = highest_scoring_pair(&files, &sort_by, u64::MAX);

        assert!(best.is_none());
    }

    #[test]
    fn irreducible_pairs_are_full_singletons_or_boundary_pinned_singletons() {
        let full_singleton = stats_entry(
            "full_singleton",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(7), int_stat(7)),
            ],
        );
        let same = stats_entry(
            "same",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(7), int_stat(7)),
            ],
        );
        let range_to_seven = stats_entry(
            "range_to_seven",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(0), int_stat(7)),
            ],
        );
        let range_around_seven = stats_entry(
            "range_around_seven",
            &[
                ("region", string_stat("us"), string_stat("us")),
                ("id", int_stat(0), int_stat(9)),
            ],
        );

        let sort_by = ["region".into(), "id".into()];
        assert!(is_irreducible_pair(&full_singleton, &same, &sort_by));
        assert!(is_irreducible_pair(
            &full_singleton,
            &range_to_seven,
            &sort_by
        ));
        assert!(!is_irreducible_pair(
            &full_singleton,
            &range_around_seven,
            &sort_by
        ));
        assert!(!is_irreducible_pair(
            &full_singleton,
            &same,
            &["region".into(), "missing".into()]
        ));
    }

    #[test]
    fn oversized_irreducible_pair_defers_to_a_mergeable_pair() {
        let sized_entry = |path: &str, min: i64, max: i64, size: u64| {
            let mut entry = stats_entry(path, &[("id", int_stat(min), int_stat(max))]);
            entry.file.size = size;
            entry
        };
        let range = sized_entry("range", 0, 100, 60);
        let pinned = sized_entry("pinned", 100, 100, 60);
        let contained = sized_entry("contained", 0, 50, 10);
        let files = [
            candidate(&range, &["id"]),
            candidate(&pinned, &["id"]),
            candidate(&contained, &["id"]),
        ];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        // The oversized pair is skipped, leaving one whose single row groups
        // are contested in whole.
        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "range");
        assert_eq!(right.file.path.as_str(), "contained");
    }

    /// Remembered scores pick what the sweep picks.
    #[test]
    fn remembered_scores_agree_with_the_sweep() {
        let files = layered_candidates();
        let mut scores = LayoutScores::new();

        let selected = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();

        let (score, left, right) = (selected.score, selected.left, selected.right);
        let (sweep_score, sweep_left, sweep_right) =
            highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();
        assert!((score - sweep_score).abs() < 1e-6);
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
        pair.sort();
        let mut sweep_pair = [
            sweep_left.file.path.as_str(),
            sweep_right.file.path.as_str(),
        ];
        sweep_pair.sort();
        assert_eq!(pair, sweep_pair);
    }

    /// A file some merge is already rewriting keeps its remembered scores but
    /// is passed over while it is reserved, so the round picks another pair.
    #[test]
    fn a_reserved_file_is_passed_over() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let overlapping = stats_entry("overlapping", &[("id", int_stat(10), int_stat(90))]);
        let far = stats_entry("far", &[("id", int_stat(500), int_stat(600))]);
        let far_overlapping =
            stats_entry("far-overlapping", &[("id", int_stat(510), int_stat(590))]);
        let files = vec![
            candidate(&wide, &["id"]),
            candidate(&overlapping, &["id"]),
            candidate(&far, &["id"]),
            candidate(&far_overlapping, &["id"]),
        ];
        let mut scores = LayoutScores::new();
        let selected = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();
        let reserved = HashSet::from([
            selected.left.file.path.clone(),
            selected.right.file.path.clone(),
        ]);

        let next = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &reserved,
            )
            .unwrap();

        assert!(!reserved.contains(&next.left.file.path));
        assert!(!reserved.contains(&next.right.file.path));
    }

    /// After a merge, the next round scores the merge's output against the
    /// survivors and no longer knows the pairs of the files it replaced.
    #[test]
    fn a_merge_output_is_scored_against_the_survivors() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let shifted = stats_entry("shifted", &[("id", int_stat(90), int_stat(190))]);
        let far = stats_entry("far", &[("id", int_stat(150), int_stat(250))]);
        let mut scores = LayoutScores::new();
        let before = vec![
            candidate(&wide, &["id"]),
            candidate(&shifted, &["id"]),
            candidate(&far, &["id"]),
        ];
        scores.best_pair(
            std::slice::from_ref(&before),
            &["id".into()],
            u64::MAX,
            &HashSet::new(),
        );
        let merged = stats_entry("merged", &[("id", int_stat(140), int_stat(200))]);
        let after = vec![candidate(&far, &["id"]), candidate(&merged, &["id"])];

        let selected = scores
            .best_pair(
                std::slice::from_ref(&after),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();

        // One row group each, both reaching into the other's range.
        assert!((selected.score - 1.0).abs() < 1e-6);
        let mut pair = [
            selected.left.file.path.as_str(),
            selected.right.file.path.as_str(),
        ];
        pair.sort();
        assert_eq!(pair, ["far", "merged"]);
        assert_eq!(
            scores.pairs.len(),
            1,
            "the merged-away files' pairs are forgotten"
        );
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

    #[test]
    fn sweep_selects_the_highest_overlap_pair_among_many_files() {
        let files = layered_candidates();

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Three of wide's five groups reach into shifted and three of
        // shifted's reach back, against the two of wide's that the contained
        // file touches.
        assert!((score - 0.6).abs() < 1e-12);
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["shifted", "wide"]);
    }

    #[test]
    fn disjoint_files_produce_no_pair() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(20), int_stat(30))]);

        let best = highest_scoring_pair(
            &[candidate(&low, &["id"]), candidate(&high, &["id"])],
            &["id".into()],
            u64::MAX,
        );

        assert!(best.is_none());
    }

    #[test]
    fn contained_singleton_never_attracts_a_merge() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let singleton = stats_entry("singleton", &[("id", int_stat(50), int_stat(50))]);
        let partner = stats_entry("partner", &[("id", int_stat(70), int_stat(170))]);
        let wide_groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)];
        let singleton_groups = [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50)];
        let partner_groups = [(70, 90), (90, 110), (110, 130), (130, 150), (150, 170)];

        let alone = highest_scoring_pair(
            &[
                candidate_with_row_groups(&wide, &wide_groups),
                candidate_with_row_groups(&singleton, &singleton_groups),
            ],
            &["id".into()],
            u64::MAX,
        );
        let (score, left, right) = highest_scoring_pair(
            &[
                candidate_with_row_groups(&wide, &wide_groups),
                candidate_with_row_groups(&singleton, &singleton_groups),
                candidate_with_row_groups(&partner, &partner_groups),
            ],
            &["id".into()],
            u64::MAX,
        )
        .unwrap();

        // The singleton sits in one of wide's five groups, too few to rewrite.
        assert!(alone.is_none());
        assert!((score - 0.4).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "wide");
        assert_eq!(right.file.path.as_str(), "partner");
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

        let best = highest_scoring_pair(&files, &["id".into()], u64::MAX);

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

        let (score, _, _) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Three of each file's five groups reach into the other's range.
        assert!((score - 0.6).abs() < 1e-12);
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

        let (score, _, _) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn full_singleton_group_merges_only_a_fitting_pair() {
        let sized_entry = |path: &str, size: u64| {
            let mut entry = stats_entry(path, &[("id", int_stat(7), int_stat(7))]);
            entry.file.size = size;
            entry
        };
        let first = sized_entry("first", 60);
        let second = sized_entry("second", 60);
        let small = sized_entry("small", 30);
        let files = [
            candidate(&first, &["id"]),
            candidate(&second, &["id"]),
            candidate(&small, &["id"]),
        ];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "first");
        assert_eq!(right.file.path.as_str(), "small");
        assert!(
            highest_scoring_pair(
                &[candidate(&first, &["id"]), candidate(&second, &["id"])],
                &["id".into()],
                100
            )
            .is_none()
        );
    }

    /// A file's statistics come from the table log and its row groups' from
    /// its footer, which decode a string column to `Utf8` and `Utf8View`. The
    /// two still have to compare.
    #[test]
    fn row_groups_are_counted_against_differently_encoded_file_stats() {
        let utf8_stat = |value: &str| Arc::new(StringArray::from(vec![value])) as ArrayRef;
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
                &[("g", "k"), ("k", "o"), ("o", "s"), ("s", "w"), ("w", "z")],
            ),
        ];

        let (score, _, _) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Three of left's groups reach past "g" and two of right's stay below
        // "m", none of which is visible while the two encodings are held
        // apart.
        assert!((score - 0.4).abs() < 1e-12);
    }

    /// A string sweep column is ranked by the same contested row groups as
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

        let (score, first, second) =
            highest_scoring_pair(&files, &["customer".into()], u64::MAX).unwrap();

        // Three of heavy's five groups reach into left, against two of
        // light's.
        assert!((score - 0.6).abs() < 1e-12);
        let mut pair = [first.file.path.as_str(), second.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["heavy", "left"]);
    }
}
