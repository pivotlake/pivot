//! Deterministic layout-compaction scenarios over modeled files.
//!
//! A modeled file has rows and encoded row-group bytes but no Parquet
//! object, since key bounds alone cannot predict compressed sizes. The
//! scenario drives the production overlap cache, group selection, and file
//! planner under the routine schedule at parallelism 1: observe the live
//! files, merge the group a routine round selects, fold the outputs back,
//! repeat. Smaller cliques stand, so the converged layout keeps its
//! pyramid; there is no final sweep. It prints a distribution report under
//! `cargo test -p datastore-pivot --lib uniform_1000 -- --nocapture`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::RangeInclusive;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use object_storage::{FileRef, ObjectPath};

use super::CompactionPass;
use super::overlap::{LayoutCandidate, OverlapCache};
use crate::manifest::{DeltaFileEntry, FileStats};

#[derive(Clone, Debug)]
struct ValueRun {
    value: i64,
    rows: usize,
}

#[derive(Clone, Debug)]
struct DistributionRowGroup {
    values: Vec<ValueRun>,
}

impl DistributionRowGroup {
    fn min(&self) -> i64 {
        self.values.first().expect("a row group has rows").value
    }

    fn max(&self) -> i64 {
        self.values.last().expect("a row group has rows").value
    }

    fn rows(&self) -> usize {
        self.values.iter().map(|run| run.rows).sum()
    }
}

#[derive(Clone)]
struct DistributionFile {
    entry: DeltaFileEntry,
    row_groups: Vec<DistributionRowGroup>,
}

impl DistributionFile {
    fn new(path: String, row_groups: Vec<DistributionRowGroup>, bytes_per_row: usize) -> Self {
        let rows = row_groups
            .iter()
            .map(DistributionRowGroup::rows)
            .sum::<usize>();
        let body_size = rows
            .checked_mul(bytes_per_row)
            .expect("the modeled file body size fits usize");
        let min = row_groups
            .first()
            .expect("a modeled file has row groups")
            .min();
        let max = row_groups
            .last()
            .expect("a modeled file has row groups")
            .max();
        let mut entry = DeltaFileEntry::new(FileRef {
            path: ObjectPath::new(path),
            size: body_size as u64,
        });
        entry.stats = Some(Arc::new(FileStats {
            num_records: Some(rows as i64),
            min_values: HashMap::from([("key".to_string(), int_stat(min))]),
            max_values: HashMap::from([("key".to_string(), int_stat(max))]),
            null_counts: HashMap::new(),
        }));
        Self { entry, row_groups }
    }

    fn path(&self) -> &str {
        self.entry.file.path.as_str()
    }

    fn rows(&self) -> usize {
        self.row_groups.iter().map(DistributionRowGroup::rows).sum()
    }

    fn min(&self) -> i64 {
        self.row_groups
            .first()
            .expect("a modeled file has row groups")
            .min()
    }

    fn max(&self) -> i64 {
        self.row_groups
            .last()
            .expect("a modeled file has row groups")
            .max()
    }
}

fn int_stat(value: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![value]))
}

/// Identical sorted files whose values are uniform inside each input
/// row-group range, compacted under the routine schedule until no full
/// clique remains. Files under half the target are not candidates, as in
/// production; small-file compaction is out of scope.
struct UniformOverlapScenario {
    input_file_count: usize,
    input_row_group_ranges: Vec<RangeInclusive<i64>>,
    rows_per_value: usize,
    encoded_body_bytes_per_row: usize,
    output_row_group_rows: usize,
    target_file_body_bytes: u64,
}

impl UniformOverlapScenario {
    fn new(
        input_file_count: usize,
        input_row_group_ranges: impl IntoIterator<Item = RangeInclusive<i64>>,
    ) -> Self {
        Self {
            input_file_count,
            input_row_group_ranges: input_row_group_ranges.into_iter().collect(),
            rows_per_value: 1,
            encoded_body_bytes_per_row: 1,
            output_row_group_rows: 128 * 1024,
            target_file_body_bytes: super::DEFAULT_COMPACT_BYTES,
        }
    }

    fn rows_per_value(mut self, rows: usize) -> Self {
        self.rows_per_value = rows;
        self
    }

    fn encoded_body_bytes_per_row(mut self, bytes: usize) -> Self {
        self.encoded_body_bytes_per_row = bytes;
        self
    }

    fn output_row_group_rows(mut self, rows: usize) -> Self {
        self.output_row_group_rows = rows;
        self
    }

    fn target_file_body_bytes(mut self, bytes: u64) -> Self {
        self.target_file_body_bytes = bytes;
        self
    }

    fn run(self) -> DistributionResult {
        assert!(self.input_file_count > 0, "a scenario needs input files");
        assert!(self.rows_per_value > 0, "values need rows");
        assert!(
            self.encoded_body_bytes_per_row > 0,
            "rows need a positive encoded-byte weight"
        );
        assert!(
            self.output_row_group_rows > 0,
            "output row groups need rows"
        );
        assert!(
            self.target_file_body_bytes > 0,
            "the file target must be positive"
        );
        validate_ranges(&self.input_row_group_ranges);

        let template = uniform_row_groups(&self.input_row_group_ranges, self.rows_per_value);
        let mut files = BTreeMap::new();
        for index in 0..self.input_file_count {
            let file = DistributionFile::new(
                format!("file-{index:08}.parquet"),
                template.clone(),
                self.encoded_body_bytes_per_row,
            );
            files.insert(file.path().to_string(), file);
        }
        let input_rows = files.values().map(DistributionFile::rows).sum::<usize>();
        let input_body_bytes = files.values().map(|file| file.entry.file.size).sum::<u64>();

        let mut cache = OverlapCache::default();
        let mut rewrites = 0usize;
        let mut next_output = self.input_file_count;
        while let Some(group) =
            select_routine_group(&mut cache, &files, self.target_file_body_bytes)
        {
            rewrites += 1;
            assert!(rewrites < 100_000, "layout scenario did not converge");
            let mut inputs = Vec::with_capacity(group.len());
            for path in &group {
                inputs.push(files.remove(path).expect("the selected input is live"));
            }
            for output in rewrite_sorted_files(
                &inputs,
                self.output_row_group_rows,
                self.encoded_body_bytes_per_row,
                usize::try_from(self.target_file_body_bytes).unwrap_or(usize::MAX),
                &mut next_output,
            ) {
                files.insert(output.path().to_string(), output);
            }
        }

        assert_eq!(
            files.values().map(DistributionFile::rows).sum::<usize>(),
            input_rows,
            "the model must conserve rows"
        );
        assert_eq!(
            files.values().map(|file| file.entry.file.size).sum::<u64>(),
            input_body_bytes,
            "the constant-byte-weight model must conserve body bytes"
        );

        DistributionResult {
            target_file_body_bytes: self.target_file_body_bytes,
            rewrites,
            files: files.into_values().collect(),
        }
    }
}

/// The group a routine round merges next out of `files`, the way the
/// compaction driver selects it: files under half the target are not
/// candidates, the cache observes the live candidates, and the routine pass
/// picks a full clique of [`super::DEFAULT_LAYOUT_CLIQUE_SIZE`] files.
fn select_routine_group(
    cache: &mut OverlapCache,
    files: &BTreeMap<String, DistributionFile>,
    target_file_body_bytes: u64,
) -> Option<Vec<String>> {
    let sort_by = ["key".to_string()];
    let candidates: Vec<LayoutCandidate<'_>> = files
        .values()
        .filter(|file| !super::is_small_file(file.entry.file.size, target_file_body_bytes))
        .map(layout_candidate)
        .collect();
    cache.observe(&[&candidates], &sort_by);
    let group = cache.select_group(
        &HashSet::new(),
        CompactionPass::Routine,
        super::DEFAULT_LAYOUT_CLIQUE_SIZE,
    )?;
    Some(group.iter().map(|path| path.to_string()).collect())
}

fn layout_candidate(file: &DistributionFile) -> LayoutCandidate<'_> {
    LayoutCandidate {
        entry: &file.entry,
        row_group_ranges: vec![
            file.row_groups
                .iter()
                .map(|group| Some((int_stat(group.min()), int_stat(group.max()))))
                .collect(),
        ],
    }
}

fn uniform_row_groups(
    ranges: &[RangeInclusive<i64>],
    rows_per_value: usize,
) -> Vec<DistributionRowGroup> {
    ranges
        .iter()
        .map(|range| DistributionRowGroup {
            values: range
                .clone()
                .map(|value| ValueRun {
                    value,
                    rows: rows_per_value,
                })
                .collect(),
        })
        .collect()
}

fn validate_ranges(ranges: &[RangeInclusive<i64>]) {
    assert!(!ranges.is_empty(), "a scenario needs input row groups");
    let mut prior_max = None;
    for range in ranges {
        assert!(range.start() <= range.end(), "row-group range is inverted");
        if let Some(prior_max) = prior_max {
            assert!(
                prior_max < *range.start(),
                "input row groups must be strictly ordered and disjoint"
            );
        }
        prior_max = Some(*range.end());
    }
}

/// Globally sort a selected batch, cut it into fixed-row-count row groups, and
/// feed their modeled encoded sizes to the production compaction file planner.
fn rewrite_sorted_files(
    inputs: &[DistributionFile],
    output_row_group_rows: usize,
    bytes_per_row: usize,
    target_file_body_bytes: usize,
    next_output: &mut usize,
) -> Vec<DistributionFile> {
    let mut values = BTreeMap::<i64, usize>::new();
    for file in inputs {
        for group in &file.row_groups {
            for run in &group.values {
                *values.entry(run.value).or_default() += run.rows;
            }
        }
    }

    let mut row_groups = Vec::new();
    let mut current_values = Vec::new();
    let mut current_rows = 0usize;
    for (value, mut rows) in values {
        while rows > 0 {
            let take = rows.min(output_row_group_rows - current_rows);
            current_values.push(ValueRun { value, rows: take });
            current_rows += take;
            rows -= take;
            if current_rows == output_row_group_rows {
                row_groups.push(DistributionRowGroup {
                    values: std::mem::take(&mut current_values),
                });
                current_rows = 0;
            }
        }
    }
    if !current_values.is_empty() {
        row_groups.push(DistributionRowGroup {
            values: current_values,
        });
    }

    let row_group_body_sizes = row_groups
        .iter()
        .map(|group| group.rows() * bytes_per_row)
        .collect::<Vec<_>>();
    parquet_engine::writing::compaction_file_ranges(&row_group_body_sizes, target_file_body_bytes)
        .into_iter()
        .map(|range| {
            let path = format!("output-{:06}.parquet", *next_output);
            *next_output += 1;
            DistributionFile::new(path, row_groups[range].to_vec(), bytes_per_row)
        })
        .collect()
}

struct DistributionResult {
    target_file_body_bytes: u64,
    rewrites: usize,
    files: Vec<DistributionFile>,
}

impl DistributionResult {
    fn range_widths(&self) -> BTreeMap<i64, usize> {
        let mut widths = BTreeMap::new();
        for file in &self.files {
            *widths.entry(file.max() - file.min() + 1).or_default() += 1;
        }
        widths
    }

    fn compact_report(&self) -> String {
        let mut classes = BTreeMap::<(i64, usize, usize, u64), usize>::new();
        for file in &self.files {
            let class = (
                file.max() - file.min() + 1,
                file.row_groups.len(),
                file.rows(),
                file.entry.file.size,
            );
            *classes.entry(class).or_default() += 1;
        }
        let mut lines = vec![
            "overlap distribution scenario (best group first, parallelism 1)".to_string(),
            format!(
                "output: {} files after {} rewrites",
                with_commas(self.files.len() as u64),
                with_commas(self.rewrites as u64)
            ),
        ];
        for ((key_width, row_groups, rows, body_bytes), files) in classes {
            let key_unit = if key_width == 1 {
                "key/range"
            } else {
                "keys/range"
            };
            lines.push(format!(
                "  {} files: {} {}, {} row groups, {} rows, {} body bytes ({:.1}% full)",
                with_commas(files as u64),
                key_width,
                key_unit,
                row_groups,
                with_commas(rows as u64),
                with_commas(body_bytes),
                body_bytes as f64 * 100.0 / self.target_file_body_bytes as f64,
            ));
        }
        lines.join("\n")
    }
}

fn with_commas(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }
    formatted
}

/// 1,000 identical files of ten 100-value row groups. Only full cliques of
/// six merge, so each level leaves its remainder standing (1,000 is 166
/// cliques plus 4) and the population converges to a pyramid over a
/// settled base.
#[test]
fn uniform_1000_files_reach_the_expected_range_distribution() {
    let result = UniformOverlapScenario::new(
        1_000,
        (0..10).map(|group| group * 100 + 1..=(group + 1) * 100),
    )
    .rows_per_value(1)
    .encoded_body_bytes_per_row(1)
    .output_row_group_rows(100)
    .target_file_body_bytes(1_000)
    .run();
    println!("{}", result.compact_report());

    assert_eq!(result.files.len(), 1_000);
    assert_eq!(result.rewrites, 472);
    assert_eq!(
        result.range_widths(),
        BTreeMap::from([
            (5, 352),
            (6, 512),
            (28, 36),
            (29, 72),
            (167, 16),
            (168, 8),
            (1_000, 4),
        ])
    );
    assert!(
        result.files.iter().all(|file| {
            file.row_groups.len() == 10
                && file.row_groups.iter().all(|group| group.rows() == 100)
                && file.rows() == 1_000
                && file.entry.file.size == 1_000
        }),
        "every output should be target-full with ten equal row groups"
    );

    let mut value_counts = BTreeMap::<i64, usize>::new();
    for file in &result.files {
        for group in &file.row_groups {
            for run in &group.values {
                *value_counts.entry(run.value).or_default() += run.rows;
            }
        }
    }
    assert_eq!(value_counts.len(), 1_000);
    for value in 1..=1_000 {
        assert_eq!(value_counts[&value], 1_000);
    }
}
