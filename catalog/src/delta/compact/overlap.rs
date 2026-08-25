use std::cmp::Ordering;

use arrow_arith::numeric::sub;
use arrow_array::{Array, ArrayRef, Float64Array};
use arrow_cast::cast;
use arrow_ord::cmp;
use arrow_schema::DataType;

use crate::delta::manifest::DeltaFileEntry;

/// Overlap score for string ranges that genuinely intersect. String values
/// carry no meaningful width, so no uniform-distribution fraction can be
/// estimated for them; any real overlap counts the same, above the selection
/// threshold so overlapping string layouts still get rewritten.
const STRING_OVERLAP_SCORE: f64 = 0.5;

#[derive(Clone, Copy)]
struct UniformRange<'a> {
    min: &'a ArrayRef,
    max: &'a ArrayRef,
    rows: f64,
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

/// Find the highest-overlap pair among `files`. The caller has already
/// grouped these large-file candidates by partition. A sweep over the files
/// sorted by their minimum on the first sort column scores every pair whose
/// ranges intersect, and only those: the cost is proportional to the number
/// of actually overlapping pairs, linearithmic for a well-layered table and
/// quadratic only while heavy overlap persists, which is exactly when merges
/// keep firing and shrinking it. A pair whose rewrite cannot separate its
/// overlap and cannot fit under `target_bytes` in one file is skipped here,
/// so it neither loops through futile rewrites nor shadows a mergeable pair
/// behind it.
pub(super) fn highest_scoring_pair<'a>(
    files: &[&'a DeltaFileEntry],
    sort_by: &[String],
    target_bytes: u64,
) -> Option<(f64, &'a DeltaFileEntry, &'a DeltaFileEntry)> {
    let first_column = sort_by.first()?;
    let mut ranges: Vec<SortColumnRange<'_>> = files
        .iter()
        .filter_map(|entry| extract_column_range(entry, first_column))
        .collect();
    // Validated same-schema scalars always compare; an incomparable pair only
    // weakens the sweep order, never the scores.
    ranges.sort_by(|left, right| compare(left.min, right.min).unwrap_or(Ordering::Equal));

    let mut best: Option<(f64, &DeltaFileEntry, &DeltaFileEntry)> = None;
    let mut active: Vec<&SortColumnRange<'_>> = Vec::new();
    for range in &ranges {
        // Stats are inclusive, so a partner whose maximum equals this minimum
        // still shares rows with it: wide ranges score zero there, but equal
        // singletons defer to the next sort column and must stay paired.
        active.retain(|partner| compare(partner.max, range.min) != Some(Ordering::Less));
        for partner in &active {
            let left = partner.entry;
            let right = range.entry;
            let combined_size = left.file.size.saturating_add(right.file.size);
            if combined_size > target_bytes && is_irreducible_pair(left, right, sort_by) {
                continue;
            }
            let Some(score) = file_overlap(left, right, sort_by) else {
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

/// A file's min/max on one sort column, kept beside its manifest entry so
/// sweep candidates can be reported as entries.
struct SortColumnRange<'a> {
    entry: &'a DeltaFileEntry,
    min: &'a ArrayRef,
    max: &'a ArrayRef,
}

fn extract_column_range<'a>(
    entry: &'a DeltaFileEntry,
    column: &str,
) -> Option<SortColumnRange<'a>> {
    let stats = entry.stats.as_ref()?;
    if stats.num_records? <= 0 {
        return None;
    }
    let min = stats.min_values.get(column)?;
    let max = stats.max_values.get(column)?;
    if compare(min, max)? == Ordering::Greater {
        return None;
    }
    Some(SortColumnRange { entry, min, max })
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
    let rows = stats.num_records?;
    if rows <= 0 {
        return None;
    }
    let min = stats.min_values.get(column)?;
    let max = stats.max_values.get(column)?;
    if compare(min, max)? == Ordering::Greater {
        return None;
    }
    Some(UniformRange {
        min,
        max,
        rows: rows as f64,
    })
}

enum ColumnOverlap {
    /// Both files contain exactly the same value on this key, so the next key
    /// is the first one that can distinguish their layout.
    NextSortColumn,
    Score(f64),
}

/// Estimated rows whose value lies in `interval_min..=interval_max`, summed
/// over the given files' distributions. Each file contributes its row count
/// uniformly across its own min/max range, so intervals are measured in rows
/// rather than raw value widths and a dense narrow file outweighs a sparse
/// wide one.
fn estimated_interval_rows(
    interval_min: &ArrayRef,
    interval_max: &ArrayRef,
    distributions: &[UniformRange<'_>],
) -> Option<f64> {
    let mut estimated_rows = 0.0;
    for range in distributions {
        if compare(range.min, range.max)? == Ordering::Equal {
            let contained = compare(range.min, interval_min)? != Ordering::Less
                && compare(range.min, interval_max)? != Ordering::Greater;
            if contained {
                estimated_rows += range.rows;
            }
            continue;
        }

        let intersection_min = if compare(range.min, interval_min)? == Ordering::Less {
            interval_min
        } else {
            range.min
        };
        let intersection_max = if compare(range.max, interval_max)? == Ordering::Greater {
            interval_max
        } else {
            range.max
        };
        if compare(intersection_min, intersection_max)? != Ordering::Less {
            continue;
        }

        let covered_width = numeric_width(intersection_min, intersection_max)?;
        let range_width = numeric_width(range.min, range.max)?;
        let covered_fraction = (covered_width / range_width).min(1.0);
        estimated_rows += range.rows * covered_fraction;
    }
    Some(estimated_rows)
}

/// Overlap of the pair on one sort column, judged on the two files' own row
/// distributions: the estimated rows inside their intersection, relative to
/// the rows spanned by the smaller of the two files.
fn column_overlap(left: &UniformRange<'_>, right: &UniformRange<'_>) -> Option<ColumnOverlap> {
    let left_min = left.min;
    let left_max = left.max;
    let right_min = right.min;
    let right_max = right.max;

    let left_singleton = compare(left_min, left_max)? == Ordering::Equal;
    let right_singleton = compare(right_min, right_max)? == Ordering::Equal;
    if left_singleton && right_singleton {
        return Some(if compare(left_min, right_min)? == Ordering::Equal {
            ColumnOverlap::NextSortColumn
        } else {
            ColumnOverlap::Score(0.0)
        });
    }
    if left_singleton {
        let contained = compare(left_min, right_min)? != Ordering::Less
            && compare(left_max, right_max)? != Ordering::Greater;
        return Some(ColumnOverlap::Score(if contained { 1.0 } else { 0.0 }));
    }
    if right_singleton {
        let contained = compare(right_min, left_min)? != Ordering::Less
            && compare(right_max, left_max)? != Ordering::Greater;
        return Some(ColumnOverlap::Score(if contained { 1.0 } else { 0.0 }));
    }

    let intersection_min = if compare(left_min, right_min)? == Ordering::Less {
        right_min
    } else {
        left_min
    };
    let intersection_max = if compare(left_max, right_max)? == Ordering::Greater {
        right_max
    } else {
        left_max
    };
    if compare(intersection_min, intersection_max)? != Ordering::Less {
        return Some(ColumnOverlap::Score(0.0));
    }

    if is_string_type(intersection_min) {
        return Some(ColumnOverlap::Score(STRING_OVERLAP_SCORE));
    }

    // The score is the fraction of the more-constrained file's rows that are
    // contested: estimated rows inside the intersection over the rows spanned
    // by the smaller of the two ranges. Normalizing by the smaller file makes
    // full containment score 1.0, so merges that completely absorb one file's
    // overlap rank first; dividing by the larger file would bury exactly
    // those against a huge, mostly indifferent partner.
    let pair = [*left, *right];
    let intersection_rows = estimated_interval_rows(intersection_min, intersection_max, &pair)?;
    let left_range_rows = estimated_interval_rows(left_min, left_max, &pair)?;
    let right_range_rows = estimated_interval_rows(right_min, right_max, &pair)?;
    let smaller_file_range_rows = left_range_rows.min(right_range_rows);
    (smaller_file_range_rows > 0.0)
        .then(|| ColumnOverlap::Score((intersection_rows / smaller_file_range_rows).min(1.0)))
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
    use crate::delta::manifest::FileStats;
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

    fn column_overlap_with_rows(
        left_min: &ArrayRef,
        left_max: &ArrayRef,
        left_rows: f64,
        right_min: &ArrayRef,
        right_max: &ArrayRef,
        right_rows: f64,
    ) -> Option<ColumnOverlap> {
        column_overlap(
            &UniformRange {
                min: left_min,
                max: left_max,
                rows: left_rows,
            },
            &UniformRange {
                min: right_min,
                max: right_max,
                rows: right_rows,
            },
        )
    }

    #[test]
    fn uniform_numeric_overlap_uses_estimated_rows() {
        let at_threshold = score(column_overlap_with_rows(
            &int_stat(0),
            &int_stat(17),
            1.0,
            &int_stat(14),
            &int_stat(31),
            1.0,
        ))
        .unwrap();
        let below_threshold = score(column_overlap_with_rows(
            &int_stat(0),
            &int_stat(17),
            1.0,
            &int_stat(15),
            &int_stat(32),
            1.0,
        ))
        .unwrap();
        assert!((at_threshold - 0.3).abs() < 1e-12);
        assert!(at_threshold >= 0.3);
        assert!(below_threshold < 0.3);

        // Raw overlap is 50% for both files, but the first file carries ten
        // times as many rows. Estimated interval rows therefore make the
        // overlap 550 / min(1050, 600), not 50 / 100.
        let weighted = score(column_overlap_with_rows(
            &int_stat(0),
            &int_stat(100),
            1_000.0,
            &int_stat(50),
            &int_stat(150),
            100.0,
        ))
        .unwrap();
        assert!((weighted - 11.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn singleton_overlap_follows_sort_key_rules() {
        assert!(matches!(
            column_overlap_with_rows(
                &int_stat(7),
                &int_stat(7),
                1.0,
                &int_stat(7),
                &int_stat(7),
                1.0,
            ),
            Some(ColumnOverlap::NextSortColumn)
        ));
        assert_eq!(
            score(column_overlap_with_rows(
                &int_stat(7),
                &int_stat(7),
                1.0,
                &int_stat(8),
                &int_stat(8),
                1.0,
            )),
            Some(0.0)
        );
        assert_eq!(
            score(column_overlap_with_rows(
                &int_stat(7),
                &int_stat(7),
                1.0,
                &int_stat(0),
                &int_stat(10),
                1.0,
            )),
            Some(1.0)
        );
        assert_eq!(
            score(column_overlap_with_rows(
                &int_stat(11),
                &int_stat(11),
                1.0,
                &int_stat(0),
                &int_stat(10),
                1.0,
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
        let files = [&left, &right];
        let (overlap, _, _) =
            highest_scoring_pair(&files, &["region".into(), "id".into()], u64::MAX).unwrap();
        assert!((overlap - 2.0 / 3.0).abs() < 1e-12);
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
        let files = [&range, &pinned, &contained];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "range");
        assert_eq!(right.file.path.as_str(), "contained");
    }

    #[test]
    fn sweep_selects_the_highest_overlap_pair_among_many_files() {
        let wide = stats_entry("wide", &[("id", int_stat(0), int_stat(100))]);
        let contained = stats_entry("contained", &[("id", int_stat(40), int_stat(45))]);
        let shifted = stats_entry("shifted", &[("id", int_stat(90), int_stat(190))]);
        let disjoint = stats_entry("disjoint", &[("id", int_stat(500), int_stat(600))]);
        let files = [&shifted, &disjoint, &wide, &contained];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], u64::MAX).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "wide");
        assert_eq!(right.file.path.as_str(), "contained");
    }

    #[test]
    fn disjoint_files_produce_no_pair() {
        let low = stats_entry("low", &[("id", int_stat(0), int_stat(10))]);
        let high = stats_entry("high", &[("id", int_stat(20), int_stat(30))]);

        let best = highest_scoring_pair(&[&low, &high], &["id".into()], u64::MAX);

        assert!(best.is_none());
    }

    #[test]
    fn singleton_finds_its_strictly_containing_partner() {
        let sized_entry = |path: &str, min: i64, max: i64| {
            let mut entry = stats_entry(path, &[("id", int_stat(min), int_stat(max))]);
            entry.file.size = 60;
            entry
        };
        let containing = sized_entry("containing", 0, 12);
        let deep = sized_entry("deep", 10, 200);
        let singleton = sized_entry("singleton", 10, 10);
        let files = [&containing, &deep, &singleton];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "containing");
        assert_eq!(right.file.path.as_str(), "singleton");
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
        let files = [&first, &second, &small];

        let (score, left, right) = highest_scoring_pair(&files, &["id".into()], 100).unwrap();

        assert!((score - 1.0).abs() < 1e-12);
        assert_eq!(left.file.path.as_str(), "first");
        assert_eq!(right.file.path.as_str(), "small");
        assert!(highest_scoring_pair(&[&first, &second], &["id".into()], 100).is_none());
    }

    #[test]
    fn intersecting_string_ranges_score_a_flat_half_overlap() {
        assert_eq!(
            score(column_overlap_with_rows(
                &string_stat("customer-aaaaaaaa"),
                &string_stat("customer-zzzzzzzz"),
                1.0,
                &string_stat("customer-mmmmmmmm"),
                &string_stat("customer-tttttttt"),
                1.0,
            )),
            Some(STRING_OVERLAP_SCORE)
        );
        assert_eq!(
            score(column_overlap_with_rows(
                &string_stat("a"),
                &string_stat("f"),
                1.0,
                &string_stat("m"),
                &string_stat("z"),
                1.0,
            )),
            Some(0.0)
        );
        assert_eq!(
            score(column_overlap_with_rows(
                &string_stat("g"),
                &string_stat("g"),
                1.0,
                &string_stat("a"),
                &string_stat("z"),
                1.0,
            )),
            Some(1.0)
        );
    }
}
