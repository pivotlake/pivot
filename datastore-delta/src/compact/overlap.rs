use std::cmp::Ordering;

use arrow_arith::numeric::sub;
use arrow_array::{Array, ArrayRef, BinaryViewArray, Float64Array, StringArray, StringViewArray};
use arrow_cast::cast;
use arrow_ord::cmp;
use arrow_schema::DataType;

use crate::manifest::DeltaFileEntry;

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

/// Natural log of the distance between two ordered values. This distance is
/// only used to calculate what fraction of one file's uniform distribution
/// lies in an interval; it is not itself the rank-space width or score.
fn distance_ln(lower: &ArrayRef, upper: &ArrayRef) -> Option<f64> {
    match lower.data_type() {
        DataType::Utf8 | DataType::Utf8View | DataType::BinaryView => {
            lexicographic_width_ln(bytes(lower)?, bytes(upper)?)
        }
        _ => {
            let lower = cast(lower.as_ref(), &DataType::Float64).ok()?;
            let upper = cast(upper.as_ref(), &DataType::Float64).ok()?;
            let width = sub(&upper, &lower).ok()?;
            let width = width.as_any().downcast_ref::<Float64Array>()?.value(0);
            (width > 0.0 && width.is_finite()).then(|| width.ln())
        }
    }
}

fn bytes(array: &ArrayRef) -> Option<&[u8]> {
    match array.data_type() {
        DataType::Utf8 => Some(
            array
                .as_any()
                .downcast_ref::<StringArray>()?
                .value(0)
                .as_bytes(),
        ),
        DataType::Utf8View => Some(
            array
                .as_any()
                .downcast_ref::<StringViewArray>()?
                .value(0)
                .as_bytes(),
        ),
        DataType::BinaryView => Some(array.as_any().downcast_ref::<BinaryViewArray>()?.value(0)),
        _ => None,
    }
}

/// Find the pair with the greatest overlap in the estimated rank space built
/// from `files`. The caller has already grouped these large-file candidates by
/// partition, so every file here contributes its rows to the same distribution.
/// A pair in which one file holds a single value on the deciding key is not a
/// candidate: that file is already perfectly clustered, and the rewrite would
/// reproduce it.
pub(super) fn highest_scoring_pair<'a>(
    files: &[&'a DeltaFileEntry],
    sort_by: &[String],
) -> Option<(f64, &'a DeltaFileEntry, &'a DeltaFileEntry)> {
    let rank_space: Vec<Vec<UniformRange<'_>>> = sort_by
        .iter()
        .map(|column| {
            files
                .iter()
                .filter_map(|entry| uniform_range(entry, column))
                .collect()
        })
        .collect();

    let mut best = None;
    for left_index in 0..files.len() {
        for right in &files[left_index + 1..] {
            let left = files[left_index];
            let Some(score) = file_overlap(left, right, sort_by, &rank_space) else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|(best_score, _, _)| score > *best_score)
            {
                best = Some((score, left, *right));
            }
        }
    }
    best
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

/// Logarithmic width in the lexicographic value space. A byte string is mapped
/// to the base-257 fraction whose digits are each byte plus one, with zeroes
/// after the string ends. That mapping preserves byte-lexicographic ordering,
/// including the rule that a prefix sorts before an extension. It gives string
/// min/max ranges a continuous width on which the same uniform-distribution
/// overlap calculation used for numbers can operate.
fn lexicographic_width_ln(lower: &[u8], upper: &[u8]) -> Option<f64> {
    if lower >= upper {
        return None;
    }
    let common = lower
        .iter()
        .zip(upper)
        .take_while(|(left, right)| left == right)
        .count();
    let lower_digit = lower.get(common).map_or(0, |byte| usize::from(*byte) + 1);
    let upper_digit = upper.get(common).map_or(0, |byte| usize::from(*byte) + 1);
    let digit_gap = upper_digit.checked_sub(lower_digit)?;
    let lower_tail = lower.get(common + 1..).unwrap_or_default();
    let upper_tail = upper.get(common + 1..).unwrap_or_default();

    // gap + upper_tail - lower_tail is rearranged into positive terms, avoiding
    // cancellation when adjacent leading digits have nearly touching tails.
    let mut terms = vec![one_minus_base257_fraction_ln(lower_tail)];
    if digit_gap > 1 {
        terms.push(((digit_gap - 1) as f64).ln());
    }
    if !upper_tail.is_empty() {
        terms.push(base257_fraction_ln(upper_tail));
    }
    let scaled_width_ln = log_sum_exp(&terms);
    Some(scaled_width_ln - (common + 1) as f64 * 257_f64.ln())
}

fn base257_fraction_ln(bytes: &[u8]) -> f64 {
    let mut factor = 1.0;
    let mut scaled = 0.0;
    for byte in bytes {
        scaled += (f64::from(*byte) + 1.0) * factor;
        factor /= 257.0;
        if factor == 0.0 {
            break;
        }
    }
    scaled.ln() - 257_f64.ln()
}

fn one_minus_base257_fraction_ln(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let first_nonzero = bytes
        .iter()
        .position(|byte| *byte < u8::MAX)
        .unwrap_or(bytes.len() - 1);
    let mut factor = 1.0;
    let mut scaled = 0.0;
    for byte in &bytes[first_nonzero..] {
        scaled += f64::from(u8::MAX - *byte) * factor;
        factor /= 257.0;
        if factor == 0.0 {
            break;
        }
    }
    // The finite string is followed by zero digits. In the complement this is
    // the remaining all-maximum tail, worth one unit at the final exponent.
    scaled += factor * 257.0;
    scaled.ln() - (first_nonzero + 1) as f64 * 257_f64.ln()
}

fn log_sum_exp(terms: &[f64]) -> f64 {
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + terms
        .iter()
        .map(|term| (*term - max).exp())
        .sum::<f64>()
        .ln()
}

enum ColumnOverlap {
    /// Both files contain exactly the same value on this key, so the next key
    /// is the first one that can distinguish their layout.
    NextSortColumn,
    /// One of the files holds a single value on this key, so it is already as
    /// clustered as a rewrite could make it. Rewriting the pair would only
    /// produce the same single-valued file again, since the merged rows are cut
    /// back into target-sized files in key order.
    AlreadyClustered,
    Score(f64),
}

/// Estimated rows whose value lies in `interval_min..=interval_max`. Every
/// eligible large file contributes its row count uniformly across its own
/// min/max range. The result is a rank-space width: dividing it by the total
/// candidate row count would turn it into a percentile width, but that common
/// denominator cancels from the overlap score.
fn estimated_rank_width(
    interval_min: &ArrayRef,
    interval_max: &ArrayRef,
    rank_space: &[UniformRange<'_>],
) -> Option<f64> {
    let mut estimated_rows = 0.0;
    for range in rank_space {
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

        let covered_distance = distance_ln(intersection_min, intersection_max)?;
        let range_distance = distance_ln(range.min, range.max)?;
        let covered_fraction = (covered_distance - range_distance).exp().min(1.0);
        estimated_rows += range.rows * covered_fraction;
    }
    Some(estimated_rows)
}

fn column_overlap(
    left_min: &ArrayRef,
    left_max: &ArrayRef,
    right_min: &ArrayRef,
    right_max: &ArrayRef,
    rank_space: &[UniformRange<'_>],
) -> Option<ColumnOverlap> {
    if compare(left_min, left_max)? == Ordering::Greater
        || compare(right_min, right_max)? == Ordering::Greater
    {
        return None;
    }

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
            && compare(left_min, right_max)? != Ordering::Greater;
        return Some(if contained {
            ColumnOverlap::AlreadyClustered
        } else {
            ColumnOverlap::Score(0.0)
        });
    }
    if right_singleton {
        let contained = compare(right_min, left_min)? != Ordering::Less
            && compare(right_min, left_max)? != Ordering::Greater;
        return Some(if contained {
            ColumnOverlap::AlreadyClustered
        } else {
            ColumnOverlap::Score(0.0)
        });
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

    let intersection_width = estimated_rank_width(intersection_min, intersection_max, rank_space)?;
    let smaller_width = estimated_rank_width(left_min, left_max, rank_space)?
        .min(estimated_rank_width(right_min, right_max, rank_space)?);
    (smaller_width > 0.0)
        .then(|| ColumnOverlap::Score((intersection_width / smaller_width).min(1.0)))
}

fn file_overlap(
    left: &DeltaFileEntry,
    right: &DeltaFileEntry,
    sort_by: &[String],
    rank_space: &[Vec<UniformRange<'_>>],
) -> Option<f64> {
    let left_stats = left.stats.as_ref()?;
    let right_stats = right.stats.as_ref()?;
    // A file without a known row count cannot contribute its own uniform
    // distribution, so it cannot be scored as a layout candidate.
    if left_stats.num_records? <= 0 || right_stats.num_records? <= 0 {
        return None;
    }
    for (column_index, column) in sort_by.iter().enumerate() {
        match column_overlap(
            left_stats.min_values.get(column)?,
            left_stats.max_values.get(column)?,
            right_stats.min_values.get(column)?,
            right_stats.max_values.get(column)?,
            rank_space.get(column_index)?,
        )? {
            ColumnOverlap::NextSortColumn => continue,
            ColumnOverlap::AlreadyClustered => return None,
            ColumnOverlap::Score(score) => return Some(score),
        }
    }
    // Every sort key is the same singleton in both files: they hold one key
    // between them, and a rewrite would hand back the same two files.
    None
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, StringViewArray};

    use super::*;
    use crate::manifest::FileStats;
    use crate::store::{FileRef, ObjectPath};

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
            ColumnOverlap::NextSortColumn | ColumnOverlap::AlreadyClustered => None,
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
        let rank_space = [
            UniformRange {
                min: left_min,
                max: left_max,
                rows: left_rows,
            },
            UniformRange {
                min: right_min,
                max: right_max,
                rows: right_rows,
            },
        ];
        column_overlap(left_min, left_max, right_min, right_max, &rank_space)
    }

    #[test]
    fn uniform_numeric_overlap_uses_estimated_rank_width() {
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
        // times as many rows. Estimated rank widths therefore make the overlap
        // 550 / min(1050, 600), not 50 / 100.
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
        assert!(matches!(
            column_overlap_with_rows(
                &int_stat(7),
                &int_stat(7),
                1.0,
                &int_stat(0),
                &int_stat(10),
                1.0,
            ),
            Some(ColumnOverlap::AlreadyClustered)
        ));
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
            highest_scoring_pair(&files, &["region".into(), "id".into()]).unwrap();
        assert!((overlap - 2.0 / 3.0).abs() < 1e-12);
        assert!(highest_scoring_pair(&files, &["region".into(), "missing".into()]).is_none());
    }

    #[test]
    fn strings_use_uniform_lexicographic_range_widths() {
        let overlap = score(column_overlap_with_rows(
            &string_stat("customer-aaaaaaaa"),
            &string_stat("customer-zzzzzzzz"),
            1.0,
            &string_stat("customer-mmmmmmmm"),
            &string_stat("customer-tttttttt"),
            1.0,
        ))
        .unwrap();
        assert!((overlap - 1.0).abs() < 1e-12);
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
    }
}
