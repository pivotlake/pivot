//! A compaction of sorted input files into one output file, in the order of
//! their merge.

use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use dispatch::memory::{BUFFER_SIZE, SlabAllocator, init_test_free_pool};
use dispatch::{
    BuildContext, Built, BuiltUnary, DataFlowDispatcher, Dispatch, OperatorIO, Sender, Unary,
    UnaryFactory, UnaryResult, to_single_worker_mpsc, values_input,
};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet_engine::ParquetTable;
use parquet_engine::writing::{
    AssembledFile, EncodedColumnChunks, MergeSortSettings, ParquetFileBuilder, compact_files_spec,
    encode_record_batches_spec, plan_compaction_shredding,
};
use parquet_variant_compute::{VariantArray, json_to_variant, unshred_variant};
use parquet_variant_json::VariantToJson;
use tempfile::TempDir;

/// A two-worker dispatch pool shared by this file's tests, with the memory
/// for all of them to run at once: every compaction keeps a few blocks per
/// worker whatever it reads.
fn test_dispatcher() -> &'static DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(2, 128, None))
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

/// Appends the column chunks of a compaction to one file held in memory, as
/// a writer would to a file in storage, and emits the file's bytes when the
/// compaction is done.
struct InMemoryFileWriter {
    schema: SchemaRef,
    builder: Option<ParquetFileBuilder>,
    file: Vec<u8>,
}

impl UnaryFactory<EncodedColumnChunks, Vec<u8>> for InMemoryFileWriter {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<EncodedColumnChunks, Vec<u8>> for InMemoryFileWriter {
    /// The column chunks' bytes are copied into the file at once, so their
    /// memory is given back to the compaction here.
    fn consume(
        &mut self,
        column_chunks: EncodedColumnChunks,
        _sender: &mut dyn Sender<Vec<u8>>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let builder = self
            .builder
            .get_or_insert_with(|| ParquetFileBuilder::new(self.schema.clone()));
        let bytes =
            builder.append_column_chunks(column_chunks.chunks, &mut SlabAllocator::new(false));
        self.file.extend(bytes.runs().flatten());
        if column_chunks.ends_row_group {
            builder.end_row_group();
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<Vec<u8>>) -> UnaryResult<bool> {
        if let Some(builder) = self.builder.take() {
            let (footer, _) = builder.finish(&mut SlabAllocator::new(false))?;
            self.file.extend(footer.runs().flatten());
            sender.send(std::mem::take(&mut self.file))?;
        }
        Ok(true)
    }
}

/// The rows of a compaction's output file: their keys, labels and texts,
/// each in the order of the file, and how many rows each of the file's row
/// groups holds.
#[derive(Default)]
struct OutputRows {
    keys: Vec<i64>,
    labels: Vec<String>,
    texts: Vec<String>,
    row_group_rows: Vec<usize>,
}

/// Compact `table`, sorted by `sort_columns`, into one file of row groups
/// of [`OUTPUT_ROW_GROUP_ROWS`] rows. Returns the file's bytes.
fn compact_into_one_file(
    table: Arc<ParquetTable>,
    sort_columns: &[&str],
    settings: MergeSortSettings,
) -> Bytes {
    let sort_columns: Vec<String> = sort_columns.iter().map(|name| name.to_string()).collect();
    let dispatcher = test_dispatcher();
    let shredding = plan_compaction_shredding(dispatcher, &table).unwrap();
    let schema = shredding.schema.clone();
    let compaction = compact_files_spec(
        dispatcher,
        table,
        shredding,
        &sort_columns,
        OUTPUT_ROW_GROUP_ROWS,
        settings,
    )
    .unwrap();
    let worker_count = dispatcher.worker_count();
    let files = compaction
        .column_chunks
        .chain(
            to_single_worker_mpsc::<EncodedColumnChunks>(worker_count, compaction.write_worker)
                .into_iter()
                .collect(),
            (0..worker_count)
                .map(|_| InMemoryFileWriter {
                    schema: schema.clone(),
                    builder: None,
                    file: Vec::new(),
                })
                .collect(),
        )
        .collect()
        .unwrap();
    let [file] = files.try_into().expect("the compaction wrote one file");
    Bytes::from(file)
}

impl OutputRows {
    /// Compact `table`, sorted by `sort_columns`, into one file, and read
    /// the file back.
    fn compact(
        table: Arc<ParquetTable>,
        sort_columns: &[&str],
        settings: MergeSortSettings,
    ) -> Self {
        let file = compact_into_one_file(table, sort_columns, settings);
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let row_groups = reader.metadata().row_groups().iter();
        let mut output = OutputRows {
            row_group_rows: row_groups
                .map(|row_group| row_group.num_rows() as usize)
                .collect(),
            ..OutputRows::default()
        };
        for batch in reader.build().unwrap() {
            let batch = batch.unwrap();
            let texts_of = |column: usize| -> Vec<String> {
                let texts = arrow_cast::cast(batch.column(column), &DataType::Utf8).unwrap();
                let texts = texts.as_string::<i32>().iter();
                texts.map(|text| text.unwrap().to_string()).collect()
            };
            output
                .keys
                .extend(batch.column(0).as_primitive::<Int64Type>().values().iter());
            output.labels.extend(texts_of(1));
            output.texts.extend(texts_of(2));
        }
        output
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
    write_batch_bytes: 8 << 20,
};

/// Small enough that every test's output is several row groups.
const OUTPUT_ROW_GROUP_ROWS: usize = 3000;

#[test]
fn files_that_do_not_overlap_are_written_one_after_the_other() {
    let keys_per_file = [(5000..10000).collect(), (0..5000).collect()];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

    assert_eq!(merged.keys, merge_keys(&keys_per_file));
}

#[test]
fn the_output_is_cut_into_row_groups_of_the_rows_asked_for() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 2 + 1).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

    assert_eq!(merged.row_group_rows, [3000, 3000, 3000, 1000]);
}

#[test]
fn files_that_overlap_are_written_in_the_order_of_their_merge() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 3).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

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
fn many_files_that_overlap_are_written_in_the_order_of_their_merge() {
    let keys_per_file: Vec<Vec<i64>> = (0..12)
        .map(|file| (0..5000).map(|row| row * 12 + file).collect())
        .collect();
    let (_dir, table) = write_files(&keys_per_file, 2_500);

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

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

#[test]
fn a_file_of_several_row_groups_is_compacted_through_all_of_them() {
    let keys: Vec<i64> = (0..10_000).map(|row| row / 3).collect();
    let keys_per_file = [keys.clone(), keys];
    let (_dir, table) = write_files(&keys_per_file, 1_500);

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

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
fn files_of_several_row_groups_are_compacted_within_a_memory_budget_with_no_room() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 2 + 1).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 2_000);
    let settings = MergeSortSettings {
        memory_bytes: 1,
        ..SETTINGS
    };

    let merged = OutputRows::compact(table, &["key"], settings);

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
fn a_row_group_longer_than_the_rows_decoded_in_one_go_is_written_in_merge_order() {
    let (keys_per_file, _dir, table) = write_long_row_groups();

    let merged = OutputRows::compact(table, &["key"], SETTINGS);

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
fn a_long_row_group_is_compacted_within_a_memory_budget_with_no_room() {
    let (keys_per_file, _dir, table) = write_long_row_groups();
    let settings = MergeSortSettings {
        memory_bytes: 1,
        ..SETTINGS
    };

    let merged = OutputRows::compact(table, &["key"], settings);

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

/// Every column chunk is a batch of its own, so each row group is written
/// in three pieces.
#[test]
fn a_row_group_written_a_column_chunk_at_a_time_is_whole() {
    let keys_per_file = [
        (0..5000).map(|row| row * 2).collect(),
        (0..5000).map(|row| row * 2 + 1).collect(),
    ];
    let (_dir, table) = write_files(&keys_per_file, 2_000);
    let settings = MergeSortSettings {
        write_batch_bytes: 1,
        ..SETTINGS
    };

    let merged = OutputRows::compact(table, &["key"], settings);

    let expected_keys = merge_keys(&keys_per_file);
    assert_eq!(merged.keys, expected_keys);
    assert_eq!(
        merged.texts,
        expected_keys
            .iter()
            .map(|&key| text_of(key))
            .collect::<Vec<_>>()
    );
    assert_eq!(merged.row_group_rows, [3000, 3000, 3000, 1000]);
}

#[test]
fn a_table_without_sort_columns_is_written_file_by_file() {
    let keys_per_file = [(0..3000).rev().collect(), (0..2000).collect()];
    let (_dir, table) = write_files(&keys_per_file, 10_000);

    let merged = OutputRows::compact(table, &[], SETTINGS);

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

    let merged = OutputRows::compact(table, &["label", "key"], SETTINGS);

    let mut expected: Vec<(String, i64)> = keys_per_file
        .iter()
        .flatten()
        .map(|&key| (label_of(key), key))
        .collect();
    expected.sort();
    let read: Vec<(String, i64)> = merged.labels.into_iter().zip(merged.keys).collect();
    assert_eq!(read, expected);
}

/// Write one file per item of `documents_per_file`, with this crate's own
/// writer: a row per document, with its key, sorted by key. The writer
/// shreds each file's documents the way that file's rows suggest.
fn write_document_files(documents_per_file: &[Vec<(i64, String)>]) -> (TempDir, Arc<ParquetTable>) {
    let dir = TempDir::new().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    for (file, documents) in documents_per_file.iter().enumerate() {
        let json: ArrayRef = Arc::new(StringArray::from_iter_values(
            documents.iter().map(|(_, document)| document.as_str()),
        ));
        let variants = json_to_variant(&json).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            variants.field("attrs"),
        ]));
        let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
            documents.iter().map(|(key, _)| *key),
        ));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![keys, Arc::new(variants.into_inner())])
                .unwrap();
        let rows = values_input(test_dispatcher(), vec![batch]).record_batches();
        let files = encode_record_batches_spec(
            rows,
            schema,
            Arc::from(Vec::new()),
            Arc::from(vec!["key".to_string()]),
            128 * 1024,
            usize::MAX,
        )
        .map_each(|file: AssembledFile| {
            Ok(file.bytes.runs().flatten().copied().collect::<Vec<u8>>())
        })
        .collect()
        .unwrap();
        let [bytes] = files.try_into().expect("the rows were written as one file");
        let path = dir.path().join(format!("documents{file}.parquet"));
        std::fs::write(&path, bytes).unwrap();
        paths.push(path);
    }
    let table = ParquetTable::from_files(test_dispatcher(), &paths, &[]).unwrap();
    (dir, Arc::new(table))
}

#[test]
fn documents_each_file_shredded_its_own_way_are_written_whole() {
    let named = |key: i64| (key, format!(r#"{{"id":{key},"name":"n{key}"}}"#));
    let flagged = |key: i64| (key, format!(r#"{{"flag":true,"id":{key}}}"#));
    let documents_per_file = [
        (0..4000).map(|row| named(row * 2)).collect(),
        (0..4000).map(|row| flagged(row * 2 + 1)).collect(),
    ];
    let (_dir, table) = write_document_files(&documents_per_file);

    let file = compact_into_one_file(table, &["key"], SETTINGS);

    let mut written: Vec<(i64, String)> = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let leaves = reader.metadata().file_metadata().schema_descr().columns();
    let leaf_paths: Vec<String> = leaves.iter().map(|leaf| leaf.path().string()).collect();
    for batch in reader.build().unwrap() {
        let batch = batch.unwrap();
        let keys = batch.column(0).as_primitive::<Int64Type>().values().iter();
        let documents = VariantArray::try_new(batch.column(1).as_ref()).unwrap();
        let documents = unshred_variant(&documents).unwrap();
        for (row, &key) in keys.enumerate() {
            let mut document = Vec::new();
            documents.value(row).to_json(&mut document).unwrap();
            written.push((key, String::from_utf8(document).unwrap()));
        }
    }
    let mut expected: Vec<(i64, String)> = documents_per_file.concat();
    expected.sort();
    assert_eq!(written, expected);
    assert!(leaf_paths.contains(&"attrs.typed_value.id.typed_value".to_string()));
}

#[test]
fn a_footer_larger_than_a_block_of_memory_is_written_whole() {
    init_test_free_pool(8);
    let column_count = 60_000;
    let fields: Vec<Field> = (0..column_count)
        .map(|column| {
            Field::new(
                format!("a column with a long name, number {column:06}"),
                DataType::Int64,
                false,
            )
        })
        .collect();
    let builder = ParquetFileBuilder::new(Arc::new(Schema::new(fields)));

    let (footer, _) = builder.finish(&mut SlabAllocator::new(false)).unwrap();
    let mut file = Vec::new();
    file.extend(footer.runs().flatten());

    assert!(file.len() > BUFFER_SIZE);
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(&Bytes::from(file))
        .unwrap();
    assert_eq!(
        metadata.file_metadata().schema_descr().num_columns(),
        column_count
    );
}
