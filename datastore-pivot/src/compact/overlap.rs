use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
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

/// Measure one pair against every gate, on the column that tells its two
/// files apart: the score a round ranks the pair by, or `None` for a pair no
/// round should merge.
///
/// Files are sorted by the whole key, so a column decides the pair only once
/// the two files are not the same single value on it: while both hold one
/// identical value there, their rows are ordered by the next key. On the
/// deciding column, ranges that meet at exactly one value share no stretch of
/// the key, and a rewrite could only split the merged rows at that same
/// value, handing back the two edges it started from, so a touching pair
/// never merges. An incomparable bound keeps the pair rather than dropping
/// it. A pair the gates pass is scored by its contested row-group fraction,
/// counted from the footers rather than estimated, and both files need more
/// than a fifth of their groups contested before the rewrite is worthwhile.
fn measure_pair(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    sort_by: &[String],
    target_bytes: u64,
) -> Option<f64> {
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
        // Disjoint, or touching at one value: the pair shares no stretch of
        // the key. Statistics are inclusive, so a maximum that only equals
        // the other minimum touches it at one value.
        if matches!(
            compare(left_max, right_min),
            Some(Ordering::Less | Ordering::Equal)
        ) || matches!(
            compare(right_max, left_min),
            Some(Ordering::Less | Ordering::Equal)
        ) {
            return None;
        }
        return score_contested_row_groups(left, right, column, name);
    }
    // One identical key value throughout both files: every row group holds
    // the partner's rows, so the pair is contested in full, and no rewrite
    // can pull it apart. Merging still helps when the pair fits one output,
    // which removes a file; past the ceiling the rewrite would split inside
    // the value's run and hand back the same pair forever.
    let combined_size = left.entry.file.size.saturating_add(right.entry.file.size);
    (combined_size <= target_bytes).then_some(1.0)
}

/// Score a pair on the column that decides it, from the two files' row
/// groups: count each side's contested groups against the other file's
/// range, and reject the pair unless both files have more than a fifth of
/// their groups contested, which is what makes a rewrite worthwhile.
/// `column` is the deciding column's position in the sort order and `name`
/// its name; the caller has already established both files carry bounds for
/// it.
fn score_contested_row_groups(
    left: &LayoutCandidate,
    right: &LayoutCandidate,
    column: usize,
    name: &str,
) -> Option<f64> {
    let left_range = left.column_bounds(name)?;
    let right_range = right.column_bounds(name)?;
    let left_contested = count_contested_row_groups(left, column, right_range);
    let right_contested = count_contested_row_groups(right, column, left_range);
    if left_contested < required_contested_row_groups(left.row_group_count())
        || right_contested < required_contested_row_groups(right.row_group_count())
    {
        return None;
    }
    // The gate has found contested groups on both sides, so neither file is
    // without row groups here. The smaller share is the score: it is what a
    // rewrite can restructure of the less contested file, and it keeps a
    // narrow file nested inside a much wider one from outranking a pair that
    // restructures both sides.
    let left_share = left_contested as f64 / left.row_group_count() as f64;
    let right_share = right_contested as f64 / right.row_group_count() as f64;
    Some(left_share.min(right_share))
}

/// One table's layout pair scores, remembered between compaction rounds.
///
/// A pair's score depends only on the two files' footers and manifest stats,
/// which never change while the files exist, so a round need not measure
/// every intersecting pair again: on a table of thousands of heavily
/// overlapping files that is millions of pairs every few seconds. A round
/// measures only the files it has not seen before (normally the outputs of
/// the last merges and freshly flushed files) against the rest of their
/// partition, forgets the pairs of files that are gone, and takes the best
/// of what it remembers. Only pairs that cleared every gate are kept. Paths
/// are shared, so each path string is stored once however many pairs contain
/// it.
#[derive(Default)]
pub(super) struct LayoutScoreCache {
    known_paths: HashSet<Arc<ObjectPath>>,
    pair_scores: HashMap<(Arc<ObjectPath>, Arc<ObjectPath>), f64>,
}

impl LayoutScoreCache {
    /// The best remembered merge group among `partitions`, each one
    /// partition's large-file candidates, measuring the files no earlier
    /// round has seen. At most `max_files` files, at least two; pairs only
    /// exist within a partition, so a group never crosses one. A file some
    /// merge is rewriting keeps its scores, since the merge can fail;
    /// `reserved` only makes selection pass its pairs over. Ties go to the
    /// lexicographically first path, so the choice is stable across rounds.
    pub(super) fn select_group<'a>(
        &mut self,
        partitions: &[&[LayoutCandidate<'a>]],
        sort_by: &[String],
        target_bytes: u64,
        reserved: &HashSet<ObjectPath>,
        max_files: usize,
    ) -> Option<Vec<&'a DeltaFileEntry>> {
        let mut entries_by_path = HashMap::new();
        let mut new_paths = HashSet::new();
        for candidate in partitions.iter().flat_map(|partition| partition.iter()) {
            let path = &candidate.entry.file.path;
            entries_by_path.insert(path.as_str(), candidate.entry);
            if !self.known_paths.contains(path) {
                self.known_paths.insert(Arc::new(path.clone()));
                new_paths.insert(path.as_str());
            }
        }
        self.retain_live_files(&entries_by_path);

        for partition in partitions {
            for candidate in *partition {
                let candidate_path = &candidate.entry.file.path;
                if !new_paths.contains(candidate_path.as_str()) {
                    continue;
                }
                let shared_path = Arc::clone(
                    self.known_paths
                        .get(candidate_path)
                        .expect("the path was just interned"),
                );
                for partner in *partition {
                    let partner_path = &partner.entry.file.path;
                    if candidate_path == partner_path
                        || (new_paths.contains(partner_path.as_str())
                            && partner_path.as_str() < candidate_path.as_str())
                    {
                        continue;
                    }
                    if let Some(score) = measure_pair(candidate, partner, sort_by, target_bytes) {
                        let shared_partner_path = self
                            .known_paths
                            .get(partner_path)
                            .expect("the path was just interned");
                        self.pair_scores
                            .insert(ordered_pair(&shared_path, shared_partner_path), score);
                    }
                }
            }
        }

        let group = self.grow_group(reserved, max_files.max(2))?;
        Some(
            group
                .iter()
                .map(|path| entries_by_path[path.as_str()])
                .collect(),
        )
    }

    /// Build one merge group from the remembered pair scores. The seed is the
    /// file whose eligible partners can fill the largest group, most heavily
    /// scored ones first: merging a file with everything it overlaps decodes
    /// and re-encodes each row once, where pairwise merges would rewrite the
    /// same rows again and again. The group then grows greedily by the file
    /// whose pairs into the current members sum highest, which prefers a set
    /// of mutually overlapping files (every member counts) over a chain of
    /// files that only touch one member each.
    fn grow_group(
        &self,
        reserved: &HashSet<ObjectPath>,
        max_files: usize,
    ) -> Option<Vec<Arc<ObjectPath>>> {
        let mut partners: HashMap<&Arc<ObjectPath>, Vec<(&Arc<ObjectPath>, f64)>> = HashMap::new();
        for ((left, right), &score) in &self.pair_scores {
            if reserved.contains(left.as_ref()) || reserved.contains(right.as_ref()) {
                continue;
            }
            partners.entry(left).or_default().push((right, score));
            partners.entry(right).or_default().push((left, score));
        }
        for scored in partners.values_mut() {
            scored.sort_by(|(left_path, left_score), (right_path, right_score)| {
                right_score
                    .total_cmp(left_score)
                    .then_with(|| left_path.as_str().cmp(right_path.as_str()))
            });
        }

        let partner_value = |scored: &[(&Arc<ObjectPath>, f64)]| {
            let filling = scored.len().min(max_files - 1);
            let score: f64 = scored[..filling].iter().map(|(_, score)| score).sum();
            (filling, score)
        };
        let (seed, _) = partners
            .iter()
            .max_by(|(left_path, left), (right_path, right)| {
                let (left_count, left_score) = partner_value(left);
                let (right_count, right_score) = partner_value(right);
                left_count
                    .cmp(&right_count)
                    .then_with(|| left_score.total_cmp(&right_score))
                    .then_with(|| right_path.as_str().cmp(left_path.as_str()))
            })?;

        let mut group = vec![Arc::clone(seed)];
        let mut member_paths = HashSet::from([seed.as_str()]);
        while group.len() < max_files {
            let next = partners
                .keys()
                .filter(|path| !member_paths.contains(path.as_str()))
                .filter_map(|path| {
                    let connection: f64 = partners[*path]
                        .iter()
                        .filter(|(partner, _)| member_paths.contains(partner.as_str()))
                        .map(|(_, score)| score)
                        .sum();
                    (connection > 0.0).then_some((connection, *path))
                })
                .max_by(|(left_score, left_path), (right_score, right_path)| {
                    left_score
                        .total_cmp(right_score)
                        .then_with(|| right_path.as_str().cmp(left_path.as_str()))
                });
            let Some((_, next)) = next else {
                break;
            };
            group.push(Arc::clone(next));
            member_paths.insert(next.as_str());
        }
        Some(group)
    }

    /// How many pairs cleared every merge gate at the last selection, whether
    /// or not a merge in flight holds one of their files.
    pub(super) fn eligible_pair_count(&self) -> usize {
        self.pair_scores.len()
    }

    /// Retain only `live` files and pairs containing two live files.
    fn retain_live_files(&mut self, live: &HashMap<&str, &DeltaFileEntry>) {
        self.known_paths
            .retain(|path| live.contains_key(path.as_str()));
        self.pair_scores.retain(|(left, right), _| {
            live.contains_key(left.as_str()) && live.contains_key(right.as_str())
        });
    }
}

fn ordered_pair(
    left: &Arc<ObjectPath>,
    right: &Arc<ObjectPath>,
) -> (Arc<ObjectPath>, Arc<ObjectPath>) {
    if left.as_str() < right.as_str() {
        (Arc::clone(left), Arc::clone(right))
    } else {
        (Arc::clone(right), Arc::clone(left))
    }
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

/// How many of `file`'s row groups are contested: lying within `partner`'s
/// range on the sort column at `column`, boundaries included. A contested
/// group's rows sit where the partner also holds rows, so a rewrite can
/// restructure them against the partner's; a boundary value is a place the
/// partner holds rows too, which is what lets two files spanning the same
/// range count their boundary-valued runs. A group reaching past the
/// partner's range is anchored to rows only its own file holds, and a rewrite
/// cannot tighten it against this partner.
fn count_contested_row_groups(
    file: &LayoutCandidate,
    column: usize,
    partner: (&ArrayRef, &ArrayRef),
) -> usize {
    let (partner_min, partner_max) = partner;
    file.row_group_ranges[column]
        .iter()
        .filter(|bounds| {
            // A row group without recorded bounds can hold rows anywhere in
            // the file's range, so it counts as contested.
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

    /// Select a two-file group over one partition through the production
    /// path, with nothing remembered, returning the pair beside its
    /// remembered score.
    fn select_pair<'a>(
        files: &[LayoutCandidate<'a>],
        sort_by: &[String],
        target_bytes: u64,
    ) -> Option<(f64, &'a DeltaFileEntry, &'a DeltaFileEntry)> {
        let mut cache = LayoutScoreCache::default();
        let group = cache.select_group(&[files], sort_by, target_bytes, &HashSet::new(), 2)?;
        let [left, right] = group[..] else {
            panic!("a two-file cap selects exactly a pair, got {}", group.len());
        };
        Some((pair_score(&cache, left, right), left, right))
    }

    /// The remembered score of the pair holding `left` and `right`.
    fn pair_score(cache: &LayoutScoreCache, left: &DeltaFileEntry, right: &DeltaFileEntry) -> f64 {
        let mut wanted = [left.file.path.as_str(), right.file.path.as_str()];
        wanted.sort_unstable();
        cache
            .pair_scores
            .iter()
            .find_map(|(pair, score)| {
                let mut paths = [pair.0.as_str(), pair.1.as_str()];
                paths.sort_unstable();
                (paths == wanted).then_some(*score)
            })
            .expect("the selected pair is remembered")
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

        let (overlap, _, _) = select_pair(&files, &sort_by, u64::MAX).unwrap();

        // Two of each file's five groups lie within the other's id range,
        // which the shared region says nothing about.
        assert!((overlap - 0.4).abs() < 1e-12);
        assert!(select_pair(&files, &["region".into(), "missing".into()], u64::MAX).is_none());
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

        let best = select_pair(&files, &sort_by, u64::MAX);

        assert!(best.is_none());
    }

    /// A pair holding one identical key value throughout cannot be pulled
    /// apart, so it merges only while it fits one output, which removes a
    /// file.
    #[test]
    fn identical_value_pair_merges_only_under_the_output_ceiling() {
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

        let (score, _, _) = select_pair(&files, &sort_by, u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert!(select_pair(&files, &sort_by, 1).is_none());
        assert!(select_pair(&files, &["region".into(), "missing".into()], u64::MAX).is_none());
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

        let (score, _, _) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
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

        let (score, _, _) = select_pair(&files, &sort_by, u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
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

        let (score, _, _) = select_pair(&files, &sort_by, u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
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

        assert!(select_pair(&files, &["id".into()], u64::MAX).is_none());
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

        let (score, _, _) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0 / 3.0).abs() < 1e-12);
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

        let (score, left, right) = select_pair(&files, &["id".into()], 100).unwrap();

        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["nested", "range"]);
        assert!((score - 1.0 / 3.0).abs() < 1e-12);
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
        let mut cache = LayoutScoreCache::default();
        let before = vec![
            candidate_with_row_groups(&first, &shape),
            candidate_with_row_groups(&second, &shape),
            candidate_with_row_groups(&survivor, &survivor_shape),
        ];
        cache.select_group(&[&before], &["id".into()], u64::MAX, &HashSet::new(), 2);
        let merged = stats_entry("merged", &[("id", int_stat(500), int_stat(600))]);
        let after = vec![
            candidate_with_row_groups(&survivor, &survivor_shape),
            candidate_with_row_groups(&merged, &survivor_shape),
        ];

        let group = cache
            .select_group(&[&after], &["id".into()], u64::MAX, &HashSet::new(), 2)
            .unwrap();

        let (left, right) = (group[0], group[1]);
        assert!((pair_score(&cache, left, right) - 1.0).abs() < 1e-12);
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["merged", "survivor"]);
        assert_eq!(
            cache.pair_scores.len(),
            1,
            "the merged-away files' pairs are forgotten"
        );
    }

    /// A file some merge is already rewriting keeps its remembered scores but
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
        let mut cache = LayoutScoreCache::default();
        let first_group = cache
            .select_group(&[&files], &["id".into()], u64::MAX, &HashSet::new(), 2)
            .unwrap();
        let reserved: HashSet<ObjectPath> = first_group
            .iter()
            .map(|entry| entry.file.path.clone())
            .collect();

        let next_group = cache
            .select_group(&[&files], &["id".into()], u64::MAX, &reserved, 2)
            .unwrap();

        assert_eq!(next_group.len(), 2);
        assert!(
            next_group
                .iter()
                .all(|entry| !reserved.contains(&entry.file.path))
        );
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

        let mut cache = LayoutScoreCache::default();
        cache.select_group(&[&files], &["id".into()], u64::MAX, &HashSet::new(), 2);

        assert_eq!(cache.eligible_pair_count(), 3);
    }

    #[test]
    fn selection_picks_the_highest_overlap_pair_among_many_files() {
        let files = layered_candidates();

        let (score, left, right) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Two of wide's five groups lie within shifted's range and two of
        // shifted's within wide's; no other pair qualifies at all.
        assert!((score - 0.4).abs() < 1e-12);
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
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
            u64::MAX,
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

        assert!(select_pair(&files, &["id".into()], u64::MAX).is_none());
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

        assert!(select_pair(&files, &["id".into()], u64::MAX).is_none());
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
            u64::MAX,
        );
        let (score, left, right) = select_pair(
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
        let mut pair = [left.file.path.as_str(), right.file.path.as_str()];
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

        let best = select_pair(&files, &["id".into()], u64::MAX);

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

        let (score, _, _) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Two of each file's five groups lie within the other's range.
        assert!((score - 0.4).abs() < 1e-12);
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

        let (score, _, _) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn full_singleton_group_merges_only_a_fitting_pair() {
        let first = sized_entry("first", 7, 7, 60);
        let second = sized_entry("second", 7, 7, 60);
        let small = sized_entry("small", 7, 7, 30);
        let files = [
            candidate(&first, &["id"]),
            candidate(&second, &["id"]),
            candidate(&small, &["id"]),
        ];

        let (score, left, right) = select_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "first");
        assert_eq!(right.file.path.as_str(), "small");
        assert!(
            select_pair(
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

        let (score, _, _) = select_pair(&files, &["id".into()], u64::MAX).unwrap();

        // Three of left's groups lie within right's range and two of right's
        // within left's, none of which is visible while the two encodings are
        // held apart.
        assert!((score - 0.4).abs() < 1e-12);
    }

    /// A wide file beside the three narrower files tiling its range, none of
    /// which overlap each other. One shape for the group-selection tests.
    fn star_candidates<'a>(
        hub: &'a DeltaFileEntry,
        low: &'a DeltaFileEntry,
        mid: &'a DeltaFileEntry,
        high: &'a DeltaFileEntry,
    ) -> Vec<LayoutCandidate<'a>> {
        vec![
            candidate_with_row_groups(
                hub,
                &[
                    (0, 50),
                    (50, 100),
                    (100, 150),
                    (150, 200),
                    (200, 250),
                    (250, 300),
                ],
            ),
            candidate_with_row_groups(low, &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(
                mid,
                &[(100, 120), (120, 140), (140, 160), (160, 180), (180, 200)],
            ),
            candidate_with_row_groups(
                high,
                &[(200, 220), (220, 240), (240, 260), (260, 280), (280, 300)],
            ),
        ]
    }

    /// A file overlapping several files that do not overlap one another pulls
    /// all of them into one group, with itself seeding it.
    #[test]
    fn a_file_overlapping_many_disjoint_files_groups_with_all_of_them() {
        let hub = stats_entry("hub", &[("id", int_stat(0), int_stat(300))]);
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(100))]);
        let mid = stats_entry("mid", &[("id", int_stat(100), int_stat(200))]);
        let high = stats_entry("high", &[("id", int_stat(200), int_stat(300))]);
        let files = star_candidates(&hub, &low, &mid, &high);

        let group = LayoutScoreCache::default()
            .select_group(&[&files], &["id".into()], u64::MAX, &HashSet::new(), 6)
            .unwrap();

        assert_eq!(group[0].file.path.as_str(), "hub");
        let mut paths: Vec<&str> = group.iter().map(|entry| entry.file.path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(paths, ["high", "hub", "low", "mid"]);
    }

    /// The file cap truncates a group rather than rejecting it.
    #[test]
    fn a_group_never_exceeds_the_file_cap() {
        let hub = stats_entry("hub", &[("id", int_stat(0), int_stat(300))]);
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(100))]);
        let mid = stats_entry("mid", &[("id", int_stat(100), int_stat(200))]);
        let high = stats_entry("high", &[("id", int_stat(200), int_stat(300))]);
        let files = star_candidates(&hub, &low, &mid, &high);

        let group = LayoutScoreCache::default()
            .select_group(&[&files], &["id".into()], u64::MAX, &HashSet::new(), 3)
            .unwrap();

        assert_eq!(group.len(), 3);
        assert_eq!(group[0].file.path.as_str(), "hub");
    }

    /// Files that all overlap one another outrank a file with as many
    /// partners that do not: every pair inside such a group is restructured
    /// by the one rewrite.
    #[test]
    fn mutually_overlapping_files_outrank_an_equally_wide_star() {
        let hub = stats_entry("hub", &[("id", int_stat(0), int_stat(200))]);
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(100))]);
        let high = stats_entry("high", &[("id", int_stat(100), int_stat(200))]);
        let tri_a = stats_entry("tri-a", &[("id", int_stat(1_000), int_stat(1_100))]);
        let tri_b = stats_entry("tri-b", &[("id", int_stat(1_000), int_stat(1_100))]);
        let tri_c = stats_entry("tri-c", &[("id", int_stat(1_000), int_stat(1_100))]);
        let triangle_shape = [(1_000, 1_050), (1_050, 1_100)];
        let files = vec![
            candidate_with_row_groups(&hub, &[(0, 50), (50, 100), (100, 150), (150, 200)]),
            candidate_with_row_groups(&low, &[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
            candidate_with_row_groups(
                &high,
                &[(100, 120), (120, 140), (140, 160), (160, 180), (180, 200)],
            ),
            candidate_with_row_groups(&tri_a, &triangle_shape),
            candidate_with_row_groups(&tri_b, &triangle_shape),
            candidate_with_row_groups(&tri_c, &triangle_shape),
        ];

        let group = LayoutScoreCache::default()
            .select_group(&[&files], &["id".into()], u64::MAX, &HashSet::new(), 3)
            .unwrap();

        let mut paths: Vec<&str> = group.iter().map(|entry| entry.file.path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(paths, ["tri-a", "tri-b", "tri-c"]);
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

        let (score, first, second) = select_pair(&files, &["customer".into()], u64::MAX).unwrap();

        // Three of heavy's five groups lie within light's range and all of
        // light's within heavy's, against the two of heavy's that lie within
        // left's.
        assert!((score - 0.6).abs() < 1e-12);
        let mut pair = [first.file.path.as_str(), second.file.path.as_str()];
        pair.sort();
        assert_eq!(pair, ["heavy", "light"]);
    }
}
