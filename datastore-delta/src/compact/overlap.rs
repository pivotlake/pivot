use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use arrow_arith::numeric::sub;
use arrow_array::{Array, ArrayRef, Datum, Float64Array};
use arrow_cast::cast;
use arrow_ord::cmp;
use arrow_schema::DataType;

use crate::TableFile;
use crate::manifest::DeltaFileEntry;
use object_storage::ObjectPath;

/// Overlap score for string ranges that genuinely intersect. String values
/// carry no meaningful width, so no uniform-distribution fraction can be
/// estimated for them; any real overlap counts the same, above the selection
/// threshold so overlapping string layouts still get rewritten.
const STRING_OVERLAP_SCORE: f64 = 0.5;

/// A file's min/max on one sort column, over which its rows are assumed to be
/// uniformly distributed.
#[derive(Clone, Copy)]
struct UniformRange<'a> {
    min: &'a ArrayRef,
    max: &'a ArrayRef,
}

/// A layout-optimization candidate: a file's manifest entry beside its row
/// groups' min/max on the sweep (first sort) column. `None` marks a row group
/// whose footer recorded no bounds for that column; its rows can lie anywhere
/// in the file's range, so the contested-group count includes it.
pub(super) struct LayoutCandidate<'a> {
    pub(super) entry: &'a DeltaFileEntry,
    pub(super) row_group_ranges: Vec<Option<(ArrayRef, ArrayRef)>>,
}

impl<'a> LayoutCandidate<'a> {
    /// Extract each of the file's row groups' min/max on `sweep_column` from
    /// its footer metadata.
    pub(super) fn from_table_file(file: &'a TableFile, sweep_column: &str) -> Self {
        let row_group_ranges = file
            .row_groups
            .iter()
            .map(|row_group| {
                let column = row_group.schema.index_of(sweep_column).ok()?;
                let statistics = row_group.column_statistics(column)?;
                let min = statistics.min.as_ref()?.get().0.slice(0, 1);
                let max = statistics.max.as_ref()?.get().0.slice(0, 1);
                Some((min, max))
            })
            .collect();
        Self {
            entry: &file.entry,
            row_group_ranges,
        }
    }
}

fn compare(left: &ArrayRef, right: &ArrayRef) -> Option<Ordering> {
    if left.len() != 1
        || right.len() != 1
        || left.is_null(0)
        || right.is_null(0)
        || left.data_type() != right.data_type()
    {
        return None;
    }
    if cmp::eq(left, right).ok()?.value(0) {
        return Some(Ordering::Equal);
    }
    Some(if cmp::lt(left, right).ok()?.value(0) {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

/// Width between two ordered numeric values, used to calculate what fraction
/// of one file's uniform distribution lies in an interval.
fn numeric_width(lower: &ArrayRef, upper: &ArrayRef) -> Option<f64> {
    let lower = cast(lower.as_ref(), &DataType::Float64).ok()?;
    let upper = cast(upper.as_ref(), &DataType::Float64).ok()?;
    let width = sub(&upper, &lower).ok()?;
    let width = width.as_any().downcast_ref::<Float64Array>()?.value(0);
    (width > 0.0 && width.is_finite()).then_some(width)
}

fn is_string_type(array: &ArrayRef) -> bool {
    matches!(
        array.data_type(),
        DataType::Utf8 | DataType::Utf8View | DataType::BinaryView
    )
}

/// Find the highest-overlap pair among `candidates`. The caller has already
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
        // still shares rows with it: wide ranges score zero there, but equal
        // singletons defer to the next sort column and must stay paired.
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
/// fit the output ceiling unless a rewrite can separate it, both files must
/// have enough contested row groups, and the overlap must be positive. `None`
/// is a pair no round should merge.
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
    if !enough_contested_row_groups(left, right) {
        return None;
    }
    let score = file_overlap(left_entry, right_entry, sort_by)?;
    (score > 0.0).then_some(score)
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
    ) -> Option<(f64, &'a DeltaFileEntry, &'a DeltaFileEntry)> {
        let first_column = sort_by.first()?;

        let mut entries: HashMap<u32, &'a DeltaFileEntry> = HashMap::new();
        for candidate in partitions.iter().flatten() {
            let id = self.id_of(&candidate.entry.file.path);
            entries.insert(id, candidate.entry);
        }
        self.forget_missing(&entries);

        for partition in partitions {
            let ranges: Vec<(u32, SortColumnRange<'a, '_>)> = partition
                .iter()
                .filter_map(|candidate| {
                    let range = extract_column_range(candidate, first_column)?;
                    Some((self.ids[&candidate.entry.file.path], range))
                })
                .collect();
            // A pair of two new files is scored from the first one's side
            // only.
            let mut scored_this_round: HashSet<u32> = HashSet::new();
            for (id, range) in &ranges {
                if self.scored.contains(id) {
                    continue;
                }
                for (partner_id, partner) in &ranges {
                    if partner_id == id || scored_this_round.contains(partner_id) {
                        continue;
                    }
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

        // Ties go to the pair numbered first, so the choice is stable across
        // rounds.
        let (&key, &score) = self
            .pairs
            .iter()
            .filter(|(key, _)| {
                let (low, high) = pair_ids(**key);
                !held.contains(&low) && !held.contains(&high)
            })
            .max_by(|(left_key, left), (right_key, right)| {
                left.total_cmp(right).then_with(|| right_key.cmp(left_key))
            })?;
        let (low, high) = pair_ids(key);
        Some((f64::from(score), entries[&low], entries[&high]))
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
/// two, fifty need eleven. One contested group is the irreducible overlap of
/// sorted files that touch at a boundary value, and is also all a single
/// outlier row can fabricate, so the requirement climbs past it once a file
/// has five groups; scaling with the count keeps a many-group file from
/// qualifying on a sliver of itself.
fn required_contested_row_groups(total: usize) -> usize {
    total / 5 + 1
}

/// Whether enough of each file's row groups intersect the other file's range
/// on the sweep column for a merge to restructure real data on both sides.
/// This is what stops one wide file from being merged, sliver by sliver,
/// against a run of narrow neighbors it barely shares rows with.
fn enough_contested_row_groups(left: &SortColumnRange, right: &SortColumnRange) -> bool {
    meets_row_group_requirement(left, right) && meets_row_group_requirement(right, left)
}

fn meets_row_group_requirement(file: &SortColumnRange, partner: &SortColumnRange) -> bool {
    let contested = file
        .candidate
        .row_group_ranges
        .iter()
        .filter(|bounds| {
            // A row group without recorded bounds can hold rows anywhere in
            // the file's range, so it counts as contested.
            let Some((min, max)) = bounds else {
                return true;
            };
            compare(max, partner.min) != Some(Ordering::Less)
                && compare(min, partner.max) != Some(Ordering::Greater)
        })
        .count();
    contested >= required_contested_row_groups(file.candidate.row_group_ranges.len())
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

fn uniform_range<'a>(entry: &'a DeltaFileEntry, column: &str) -> Option<UniformRange<'a>> {
    let stats = entry.stats.as_ref()?;
    if stats.num_records? <= 0 {
        return None;
    }
    let min = stats.min_values.get(column)?;
    let max = stats.max_values.get(column)?;
    if compare(min, max)? == Ordering::Greater {
        return None;
    }
    Some(UniformRange { min, max })
}

enum ColumnOverlap {
    /// Both files contain exactly the same value on this key, so the next key
    /// is the first one that can distinguish their layout.
    NextSortColumn,
    Score(f64),
}

/// Overlap of the pair on one sort column: the smaller of the two files' own
/// contested fractions, where a file's contested fraction is the share of its
/// rows, assumed uniform across its min/max range, that fall inside the
/// intersection. Requiring both files to be meaningfully contested keeps a
/// narrow file nested inside a much wider one from forcing a rewrite the wide
/// file barely benefits from; two files only merge when the rewrite
/// re-partitions a real share of each of them.
fn column_overlap(left: &UniformRange<'_>, right: &UniformRange<'_>) -> Option<ColumnOverlap> {
    let left_singleton = compare(left.min, left.max)? == Ordering::Equal;
    let right_singleton = compare(right.min, right.max)? == Ordering::Equal;
    if left_singleton && right_singleton {
        return Some(if compare(left.min, right.min)? == Ordering::Equal {
            ColumnOverlap::NextSortColumn
        } else {
            ColumnOverlap::Score(0.0)
        });
    }
    // A singleton spans no width, so its partner's contested fraction is zero
    // and so is the pair's minimum.
    if left_singleton || right_singleton {
        return Some(ColumnOverlap::Score(0.0));
    }

    let intersection_min = if compare(left.min, right.min)? == Ordering::Less {
        right.min
    } else {
        left.min
    };
    let intersection_max = if compare(left.max, right.max)? == Ordering::Greater {
        right.max
    } else {
        left.max
    };
    if compare(intersection_min, intersection_max)? != Ordering::Less {
        return Some(ColumnOverlap::Score(0.0));
    }

    if is_string_type(intersection_min) {
        return Some(ColumnOverlap::Score(STRING_OVERLAP_SCORE));
    }

    let covered_width = numeric_width(intersection_min, intersection_max)?;
    let left_fraction = covered_width / numeric_width(left.min, left.max)?;
    let right_fraction = covered_width / numeric_width(right.min, right.max)?;
    Some(ColumnOverlap::Score(
        left_fraction.min(right_fraction).min(1.0),
    ))
}

fn file_overlap(left: &DeltaFileEntry, right: &DeltaFileEntry, sort_by: &[String]) -> Option<f64> {
    for column in sort_by {
        match column_overlap(
            &uniform_range(left, column)?,
            &uniform_range(right, column)?,
        )? {
            ColumnOverlap::NextSortColumn => continue,
            ColumnOverlap::Score(score) => return Some(score),
        }
    }
    // Every sort key is the same singleton in both files: their layouts overlap
    // completely even though no key has a non-zero range width.
    Some(1.0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, StringViewArray};

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

    fn score(overlap: Option<ColumnOverlap>) -> Option<f64> {
        match overlap? {
            ColumnOverlap::NextSortColumn => None,
            ColumnOverlap::Score(score) => Some(score),
        }
    }

    /// A candidate whose file has one row group spanning its whole range on
    /// `sweep_column`, the layout a small single-group parquet has.
    fn candidate<'a>(entry: &'a DeltaFileEntry, sweep_column: &str) -> LayoutCandidate<'a> {
        let stats = entry.stats.as_ref().unwrap();
        LayoutCandidate {
            entry,
            row_group_ranges: vec![Some((
                stats.min_values[sweep_column].clone(),
                stats.max_values[sweep_column].clone(),
            ))],
        }
    }

    fn candidate_with_row_groups<'a>(
        entry: &'a DeltaFileEntry,
        groups: &[(i64, i64)],
    ) -> LayoutCandidate<'a> {
        LayoutCandidate {
            entry,
            row_group_ranges: groups
                .iter()
                .map(|(min, max)| Some((int_stat(*min), int_stat(*max))))
                .collect(),
        }
    }

    fn column_overlap_between(
        left_min: &ArrayRef,
        left_max: &ArrayRef,
        right_min: &ArrayRef,
        right_max: &ArrayRef,
    ) -> Option<ColumnOverlap> {
        column_overlap(
            &UniformRange {
                min: left_min,
                max: left_max,
            },
            &UniformRange {
                min: right_min,
                max: right_max,
            },
        )
    }

    #[test]
    fn numeric_overlap_is_the_smaller_contested_fraction() {
        let mutual = score(column_overlap_between(
            &int_stat(0),
            &int_stat(100),
            &int_stat(50),
            &int_stat(150),
        ))
        .unwrap();
        let contained = score(column_overlap_between(
            &int_stat(0),
            &int_stat(100),
            &int_stat(40),
            &int_stat(45),
        ))
        .unwrap();

        // [50, 100] is half of each range; [40, 45] is all of the narrow file
        // but only 5% of the wide one, and the wide file's side decides.
        assert!((mutual - 0.5).abs() < 1e-12);
        assert!((contained - 0.05).abs() < 1e-12);
    }

    #[test]
    fn singleton_overlap_follows_sort_key_rules() {
        assert!(matches!(
            column_overlap_between(&int_stat(7), &int_stat(7), &int_stat(7), &int_stat(7)),
            Some(ColumnOverlap::NextSortColumn)
        ));
        assert_eq!(
            score(column_overlap_between(
                &int_stat(7),
                &int_stat(7),
                &int_stat(8),
                &int_stat(8),
            )),
            Some(0.0)
        );
        assert_eq!(
            score(column_overlap_between(
                &int_stat(7),
                &int_stat(7),
                &int_stat(0),
                &int_stat(10),
            )),
            Some(0.0)
        );
        assert_eq!(
            score(column_overlap_between(
                &int_stat(11),
                &int_stat(11),
                &int_stat(0),
                &int_stat(10),
            )),
            Some(0.0)
        );
    }

    #[test]
    fn equal_singleton_prefix_advances_to_next_sort_column() {
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
        let files = [candidate(&left, "region"), candidate(&right, "region")];
        let (overlap, _, _) =
            highest_scoring_pair(&files, &["region".into(), "id".into()], u64::MAX).unwrap();
        assert!((overlap - 0.5).abs() < 1e-12);
        assert!(
            highest_scoring_pair(&files, &["region".into(), "missing".into()], u64::MAX).is_none()
        );
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
            candidate(&range, "id"),
            candidate(&pinned, "id"),
            candidate(&contained, "id"),
        ];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 0.5).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "range");
        assert_eq!(right.file.path.as_str(), "contained");
    }

    /// Remembered scores pick what the sweep picks.
    #[test]
    fn remembered_scores_agree_with_the_sweep() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let contained = stats_entry("contained", &[("id", int_stat(40), int_stat(45))]);
        let shifted = stats_entry("shifted", &[("id", int_stat(90), int_stat(190))]);
        let disjoint = stats_entry("disjoint", &[("id", int_stat(500), int_stat(600))]);
        let files = vec![
            candidate(&shifted, "id"),
            candidate(&disjoint, "id"),
            candidate(&wide, "id"),
            candidate(&contained, "id"),
        ];
        let mut scores = LayoutScores::new();

        let (score, left, right) = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();

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
            candidate(&wide, "id"),
            candidate(&overlapping, "id"),
            candidate(&far, "id"),
            candidate(&far_overlapping, "id"),
        ];
        let mut scores = LayoutScores::new();
        let (_, left, right) = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();
        let reserved = HashSet::from([left.file.path.clone(), right.file.path.clone()]);

        let (_, next_left, next_right) = scores
            .best_pair(
                std::slice::from_ref(&files),
                &["id".into()],
                u64::MAX,
                &reserved,
            )
            .unwrap();

        assert!(!reserved.contains(&next_left.file.path));
        assert!(!reserved.contains(&next_right.file.path));
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
            candidate(&wide, "id"),
            candidate(&shifted, "id"),
            candidate(&far, "id"),
        ];
        scores.best_pair(
            std::slice::from_ref(&before),
            &["id".into()],
            u64::MAX,
            &HashSet::new(),
        );
        let merged = stats_entry("merged", &[("id", int_stat(140), int_stat(200))]);
        let after = vec![candidate(&far, "id"), candidate(&merged, "id")];

        let (score, left, right) = scores
            .best_pair(
                std::slice::from_ref(&after),
                &["id".into()],
                u64::MAX,
                &HashSet::new(),
            )
            .unwrap();

        // [150, 200] contests half of merged's [140, 200] and half of far's
        // [150, 250].
        assert!((score - 0.5).abs() < 1e-6);
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["far", "merged"]);
        assert_eq!(
            scores.pairs.len(),
            1,
            "the merged-away files' pairs are forgotten"
        );
    }

    #[test]
    fn sweep_selects_the_highest_overlap_pair_among_many_files() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let contained = stats_entry("contained", &[("id", int_stat(40), int_stat(45))]);
        let shifted = stats_entry("shifted", &[("id", int_stat(90), int_stat(190))]);
        let disjoint = stats_entry("disjoint", &[("id", int_stat(500), int_stat(600))]);
        let files = [
            candidate(&shifted, "id"),
            candidate(&disjoint, "id"),
            candidate(&wide, "id"),
            candidate(&contained, "id"),
        ];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        // [90, 100] contests 10% of both wide and shifted, beating the
        // contained file, which contests only 5% of wide.
        assert!((score - 0.1).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "wide");
        assert_eq!(right.file.path.as_str(), "shifted");
    }

    #[test]
    fn disjoint_files_produce_no_pair() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(20), int_stat(30))]);

        let best = highest_scoring_pair(
            &[candidate(&low, "id"), candidate(&high, "id")],
            &["id".into()],
            u64::MAX,
        );

        assert!(best.is_none());
    }

    #[test]
    fn contained_singleton_never_attracts_a_merge() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let singleton = stats_entry("singleton", &[("id", int_stat(50), int_stat(50))]);
        let partner = stats_entry("partner", &[("id", int_stat(80), int_stat(180))]);

        let alone = highest_scoring_pair(
            &[candidate(&wide, "id"), candidate(&singleton, "id")],
            &["id".into()],
            u64::MAX,
        );
        let (score, left, right) = highest_scoring_pair(
            &[
                candidate(&wide, "id"),
                candidate(&singleton, "id"),
                candidate(&partner, "id"),
            ],
            &["id".into()],
            u64::MAX,
        )
        .unwrap();

        assert!(alone.is_none());
        assert!((score - 0.2).abs() < 1e-12);
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

        assert!((score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn row_groups_without_bounds_count_as_contested() {
        let wide = stats_entry("wide", &[("id", int_stat(1), int_stat(100))]);
        let narrow = stats_entry("narrow", &[("id", int_stat(3), int_stat(4))]);
        let files = [
            LayoutCandidate {
                entry: &wide,
                row_group_ranges: vec![None; 5],
            },
            candidate_with_row_groups(&narrow, &[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4)]),
        ];

        let (score, _, _) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0 / 99.0).abs() < 1e-12);
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
            candidate(&first, "id"),
            candidate(&second, "id"),
            candidate(&small, "id"),
        ];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "first");
        assert_eq!(right.file.path.as_str(), "small");
        assert!(
            highest_scoring_pair(
                &[candidate(&first, "id"), candidate(&second, "id")],
                &["id".into()],
                100
            )
            .is_none()
        );
    }

    #[test]
    fn intersecting_string_ranges_score_a_flat_half_overlap() {
        assert_eq!(
            score(column_overlap_between(
                &string_stat("customer-aaaaaaaa"),
                &string_stat("customer-zzzzzzzz"),
                &string_stat("customer-mmmmmmmm"),
                &string_stat("customer-tttttttt"),
            )),
            Some(STRING_OVERLAP_SCORE)
        );
        assert_eq!(
            score(column_overlap_between(
                &string_stat("a"),
                &string_stat("f"),
                &string_stat("m"),
                &string_stat("z"),
            )),
            Some(0.0)
        );
        assert_eq!(
            score(column_overlap_between(
                &string_stat("g"),
                &string_stat("g"),
                &string_stat("a"),
                &string_stat("z"),
            )),
            Some(0.0)
        );
    }
}
