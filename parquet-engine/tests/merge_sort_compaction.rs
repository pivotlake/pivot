//! A compaction's read of its sorted input files in the order of their merge.

use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet_engine::ParquetTable;
use parquet_engine::writing::{
    MergeSortSettings, MergedBatchColumn, MergedRows, read_in_merge_order_spec,
};
use tempfile::TempDir;

/// A two-worker dispatch pool shared by this file's tests.
fn test_dispatcher() -> &'static DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(2, 64, None))
        .dispatcher()
}

/// The label of the row of `key`: one of a few, so the column repeats.
fn label_of(key: i64) -> String {
    format!("l{}", key % 7)
}

/// The text of the row of `key`, wide enough for a few hundred rows to fill
/// a page.
fn text_of(key: i64) -> String {
    format!("the text of key {key:08}, padded to fill pages sooner")
}

/// Write one file per item of `keys_per_file` into a fresh directory: a row
/// per key, in the order given, with the key's label and text, in row groups
/// of `rows_per_group` rows. Returns the files as one table.
fn write_files(keys_per_file: &[Vec<i64>], rows_per_group: usize) -> (TempDir, Arc<ParquetTable>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let dir = TempDir::new().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    for (file, keys) in keys_per_file.iter().enumerate() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(keys.clone())),
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(|&key| label_of(key)),
            )),
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(|&key| text_of(key)),
            )),
        ];
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_data_page_size_limit(4096)
            .set_write_batch_size(64)
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        let path = dir.path().join(format!("file{file}.parquet"));
        let mut writer = ArrowWriter::try_new(
            File::create(&path).unwrap(),
            schema.clone(),
            Some(properties),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        paths.push(path);
    }
    let table = ParquetTable::from_files(test_dispatcher(), &paths, &[]).unwrap();
    (dir, Arc::new(table))
}

/// What a read in merge order gave: the rows' keys, labels and texts, each
/// in the order the read gave the rows.
#[derive(Default)]
struct OutputRows {
    keys: Vec<i64>,
    labels: Vec<String>,
    texts: Vec<String>,
}

impl OutputRows {
    /// Read `table`, sorted by `sort_columns`, in merge order, for output
    /// row groups of [`OUTPUT_ROW_GROUP_ROWS`] rows. The rows are copied out
    /// of each merged batch inside the dataflow, on a worker, whose memory
    /// the decoded arrays are in. The merged batches leave the dataflow in
    /// any order, and are put in output order by their sequence numbers.
    fn read(table: Arc<ParquetTable>, sort_columns: &[&str], settings: MergeSortSettings) -> Self {
        let sort_columns: Vec<String> = sort_columns.iter().map(|name| name.to_string()).collect();
        let mut parts = read_in_merge_order_spec(
            test_dispatcher(),
            table,
            &sort_columns,
            OUTPUT_ROW_GROUP_ROWS,
            settings,
        )
        .unwrap()
        .map_each(|merged_batch: MergedBatchColumn| {
            let mut part = OutputRows::default();
            part.append_merged_batch(&merged_batch);
            Ok((merged_batch.sequence, part))
        })
        .collect()
        .unwrap();
        parts.sort_by_key(|(sequence, _)| *sequence);
        let mut merged = OutputRows::default();
        for (_, part) in parts {
            merged.keys.extend(part.keys);
            merged.labels.extend(part.labels);
            merged.texts.extend(part.texts);
        }
        merged
    }

    /// Append the rows of `merged_batch`, in the order its row list gives,
    /// to the column it holds.
    fn append_merged_batch(&mut self, merged_batch: &MergedBatchColumn) {
        // A text column is not decoded as the same arrow type in every
        // batch, so every source is cast to one type first.
        let column_type = match merged_batch.column {
            0 => DataType::Int64,
            _ => DataType::Utf8,
        };
        let sources: Vec<ArrayRef> = merged_batch
            .sources
            .iter()
            .map(|source| arrow_cast::cast(&source.values, &column_type).unwrap())
            .collect();
        let rows: ArrayRef = match &merged_batch.rows {
            MergedRows::Consecutive {
                source,
                first_row,
                row_count,
            } => sources[*source as usize].slice(*first_row as usize, *row_count),
            MergedRows::Listed {
                sources: source_of_each_row,
                rows: row_in_source,
            } => {
                let sources: Vec<&dyn Array> =
                    sources.iter().map(|source| source.as_ref()).collect();
                let source_and_row_of_each_row: Vec<(usize, usize)> = source_of_each_row
                    .values()
                    .iter()
                    .zip(row_in_source.values().iter())
                    .map(|(&source, &row)| (source as usize, row as usize))
                    .collect();
                arrow_select::interleave::interleave(&sources, &source_and_row_of_each_row).unwrap()
            }
        };
        match merged_batch.column {
            0 => self
                .keys
                .extend(rows.as_primitive::<Int64Type>().values().iter()),
            1 | 2 => {
                let texts = rows.as_string::<i32>().iter().map(|text| text.unwrap());
                match merged_batch.column {
                    1 => self.labels.extend(texts.map(str::to_string)),
                    _ => self.texts.extend(texts.map(str::to_string)),
                }
            }
            _ => unreachable!("the table has three columns"),
        }
    }
}

/// `keys_per_file` as one list in the order of their merge: sorted, with
/// equal keys in the order of their files.
fn merge_keys(keys_per_file: &[Vec<i64>]) -> Vec<i64> {
    let mut keys: Vec<i64> = keys_per_file.iter().flatten().copied().collect();
    keys.sort();
    keys
}

const SETTINGS: MergeSortSettings = MergeSortSettings {
    pull_rows: 1000,
    memory_bytes: 1 << 30,
};

/// Small enough that every test's output is several row groups.
const OUTPUT_ROW_GROUP_ROWS: usize = 3000;

#[test]
fn files_that_do_not_overlap_are_read_one_after_the_other() {
    let keys_per_file = [(5000..10000).collect(), (0..5000).collect()];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::read(table, &["key"], SETTINGS);

    assert_eq!(merged.keys, merge_keys(&keys_per_file));
}

#[test]
fn files_that_overlap_are_read_in_the_order_of_their_merge() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 3).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::read(table, &["key"], SETTINGS);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.texts,
        expected_keys
            .iter()
            .map(|&key| text_of(key))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        merged.labels,
        expected_keys
            .iter()
            .map(|&key| label_of(key))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_file_of_several_row_groups_is_read_through_all_of_them() {
    let keys: Vec<i64> = (0..10_000).map(|row| row / 3).collect();
    let keys_per_file = [keys.clone(), keys];
    let (_dir, table) = write_files(&keys_per_file, 1_500);

    let merged = OutputRows::read(table, &["key"], SETTINGS);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.texts,
        expected_keys
            .iter()
            .map(|&key| text_of(key))
            .collect::<Vec<_>>()
    );
}

/// The budget has room for nothing, so the columns are decoded one after
/// the other, a merged batch at a time.
#[test]
fn files_of_several_row_groups_are_read_within_a_memory_budget_with_no_room() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 2 + 1).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 2_000);
    let settings = MergeSortSettings {
        memory_bytes: 1,
        ..SETTINGS
    };

    let merged = OutputRows::read(table, &["key"], settings);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.texts,
        expected_keys
            .iter()
            .map(|&key| text_of(key))
            .collect::<Vec<_>>()
    );
}

/// Two files whose keys alternate, each one row group of 40,000 rows:
/// several times the rows that are decoded in one go.
fn write_long_row_groups() -> ([Vec<i64>; 2], TempDir, Arc<ParquetTable>) {
    let keys_per_file = [
        (0..40_000).map(|row| row * 2).collect(),
        (0..40_000).map(|row| row * 2 + 1).collect(),
    ];
    let (dir, table) = write_files(&keys_per_file, 40_000);
    (keys_per_file, dir, table)
}

#[test]
fn a_row_group_longer_than_the_rows_decoded_in_one_go_is_read_in_merge_order() {
    let (keys_per_file, _dir, table) = write_long_row_groups();

    let merged = OutputRows::read(table, &["key"], SETTINGS);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.texts,
        expected_keys
            .iter()
            .map(|&key| text_of(key))
            .collect::<Vec<_>>()
    );
}

/// The budget has room for nothing, so only the rows the output cannot go
/// on without are decoded, a merged batch of a column group at a time.
#[test]
fn a_long_row_group_is_read_within_a_memory_budget_with_no_room() {
    let (keys_per_file, _dir, table) = write_long_row_groups();
    let settings = MergeSortSettings {
        memory_bytes: 1,
        ..SETTINGS
    };

    let merged = OutputRows::read(table, &["key"], settings);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.labels,
        expected_keys
            .iter()
            .map(|&key| label_of(key))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_table_without_sort_columns_is_read_file_by_file() {
    let keys_per_file = [(0..3000).rev().collect(), (0..2000).collect()];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::read(table, &[], SETTINGS);

    assert_eq!(merged.keys, keys_per_file.concat());
}

#[test]
fn rows_with_equal_keys_are_ordered_by_the_next_sort_column() {
    let sort_by_label = |keys: std::ops::Range<i64>| {
        let mut keys: Vec<i64> = keys.collect();
        keys.sort_by_key(|&key| (label_of(key), key));
        keys
    };
    let keys_per_file = [sort_by_label(0..6000), sort_by_label(3000..9000)];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::read(table, &["label", "key"], SETTINGS);

    let mut expected: Vec<(String, i64)> = keys_per_file
        .iter()
        .flatten()
        .map(|&key| (label_of(key), key))
        .collect();
    expected.sort();
    let read: Vec<(String, i64)> = merged.labels.into_iter().zip(merged.keys).collect();
    assert_eq!(read, expected);
}
