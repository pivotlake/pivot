//! Reads of selected rows through a file's offset index, end to end: files
//! written by arrow-rs with small pages and by the engine's own writer are
//! loaded as tables and read back through the fetch pipeline.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, StringArray, StructArray, UInt32Array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
use dispatch::{DataFlowStats, Dispatch, Projection, values_input};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterPropertiesBuilder};
use tempfile::TempDir;

use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::writing::tests::{TaggedJsonItem, read_back, write};
use crate::{ParquetTable, fetch_row_groups, materialize, table_input};

const RING_BUFFERS: usize = 32;

/// Writer properties that cut a data page every couple of kilobytes, so a
/// few thousand rows span many pages and leaves of different widths span
/// different numbers of them.
fn small_pages() -> WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_size_limit(2048)
        .set_write_batch_size(16)
}

/// Write `batch` as one row group of a file in a fresh directory.
fn write_file(batch: &RecordBatch, properties: WriterPropertiesBuilder) -> (TempDir, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    let properties = properties
        .set_max_row_group_row_count(Some(batch.num_rows()))
        .build();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(properties)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    (dir, path)
}

fn load_table(dispatch: &Dispatch, path: &Path) -> Arc<ParquetTable> {
    Arc::new(ParquetTable::from_files(dispatch.dispatcher(), &[path], &[]).unwrap())
}

/// Read the rows `selection` of the table's first row group, every column,
/// with the IO the read did.
fn read_selection(
    dispatch: &Dispatch,
    table: &Arc<ParquetTable>,
    selection: RowSelection,
) -> (RecordBatch, DataFlowStats) {
    let request = QueryRowGroupMetadata::new(table, 0, selection);
    let projection = Projection::columns(0..table.schema().fields().len());
    let (batches, stats) = fetch_row_groups(
        values_input(dispatch.dispatcher(), vec![request]),
        table,
        projection,
        dispatch::RECORD_BATCH_SIZE,
        false,
    )
    .record_batches()
    .collect_with_stats()
    .unwrap();
    let rows = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
    (rows, stats)
}

/// Assert that `read` holds the rows `rows` of `batch`, column by column,
/// with each column of `batch` cast to the type the engine read it as.
fn assert_read_rows_are(read: &RecordBatch, batch: &RecordBatch, rows: &[u32]) {
    let rows = UInt32Array::from(rows.to_vec());
    for (column, field) in batch.schema().fields().iter().enumerate() {
        let read_column = read.column(column);
        let taken = arrow_select::take::take(batch.column(column), &rows, None).unwrap();
        let expected = arrow_cast::cast(&taken, read_column.data_type()).unwrap();
        assert_eq!(read_column, &expected, "column {}", field.name());
    }
}

/// A batch whose columns cover the leaf shapes a page can hold: plain and
/// dictionary encoded values, a nullable leaf, a leaf nested in a nullable
/// struct, and strings wide enough to fill pages sooner than the integers.
fn batch_of_every_leaf_shape(rows: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values((0..rows as i64).map(|row| row * 7919 % 100_003));
    let labels = StringArray::from_iter_values((0..rows).map(|row| format!("label-{}", row % 13)));
    let nullable = Int64Array::from_iter((0..rows as i64).map(|row| (row % 3 != 0).then_some(row)));
    // The engine reads a struct as absent on the rows where its leaves are,
    // so the struct here is absent exactly where its leaf is.
    let inner = Fields::from(vec![Field::new("inner", DataType::Int64, true)]);
    let nested = StructArray::new(
        inner.clone(),
        vec![Arc::new(Int64Array::from_iter(
            (0..rows as i64).map(|row| (row % 7 != 0).then_some(-row)),
        ))],
        Some(NullBuffer::from_iter((0..rows).map(|row| row % 7 != 0))),
    );
    let texts = StringArray::from_iter_values(
        (0..rows).map(|row| format!("text of row {row:08} padded out to fill pages sooner")),
    );
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("nullable", DataType::Int64, true),
        Field::new("nested", DataType::Struct(inner), true),
        Field::new("text", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(ids),
            Arc::new(labels),
            Arc::new(nullable),
            Arc::new(nested),
            Arc::new(texts),
        ],
    )
    .unwrap()
}

/// The properties for [`batch_of_every_leaf_shape`]: only `label` takes a
/// dictionary.
fn every_leaf_shape_properties() -> WriterPropertiesBuilder {
    small_pages()
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled("label".into(), true)
}

/// The first and last row of every data page of every leaf of the table's
/// first row group, sorted.
fn rows_at_page_edges(table: &ParquetTable) -> Vec<u32> {
    let row_group = &table.row_groups()[0];
    let mut rows: Vec<u32> = (0..row_group.columns.len())
        .flat_map(|leaf| row_group.locate_leaf_pages(leaf).into_iter().flatten())
        .flat_map(|page| [page.rows.start, page.rows.end - 1])
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// How many data pages each leaf of the table's first row group spans; one
/// for a leaf the index keeps no pages of.
fn page_counts(table: &ParquetTable) -> Vec<usize> {
    let row_group = &table.row_groups()[0];
    (0..row_group.columns.len())
        .map(|leaf| row_group.locate_leaf_pages(leaf).map_or(1, Iterator::count))
        .collect()
}

#[test]
fn rows_at_every_page_edge_of_every_leaf_shape_read_back() {
    let batch = batch_of_every_leaf_shape(4_000);
    let (_dir, path) = write_file(&batch, every_leaf_shape_properties());
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let rows = rows_at_page_edges(&table);

    let (read, _) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );

    let counts = page_counts(&table);
    assert!(counts.iter().all(|&count| count > 1), "{counts:?}");
    assert!(counts[0] != counts[4], "{counts:?}");
    assert_read_rows_are(&read, &batch, &rows);
    dispatch.exit();
}

#[test]
fn a_run_of_rows_spanning_several_pages_reads_back() {
    let batch = batch_of_every_leaf_shape(4_000);
    let (_dir, path) = write_file(&batch, every_leaf_shape_properties());
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let rows: Vec<u32> = (1_000..2_500).chain([3_999]).collect();

    let (read, _) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );

    assert_read_rows_are(&read, &batch, &rows);
    dispatch.exit();
}

#[test]
fn a_file_without_an_offset_index_reads_selected_rows_from_whole_chunks() {
    let batch = batch_of_every_leaf_shape(4_000);
    let properties = every_leaf_shape_properties()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_offset_index_disabled(true);
    let (_dir, path) = write_file(&batch, properties);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let rows = vec![0, 17, 2_048, 3_999];

    let (read, _) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );

    assert!(table.row_groups()[0].page_index.is_none());
    assert_read_rows_are(&read, &batch, &rows);
    dispatch.exit();
}

/// A batch of `rows` rows of an id and a hundred bytes of text that does not
/// compress, written uncompressed with pages of about 64 KiB, so the bytes
/// a read takes from the file say how many pages it read.
fn write_file_of_incompressible_text(rows: usize) -> (TempDir, PathBuf, RecordBatch) {
    let ids = Int64Array::from_iter_values(0..rows as i64);
    let texts = StringArray::from_iter_values((0..rows as u64).map(|row| {
        let mut state = row.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..100)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (b'a' + (state % 26) as u8) as char
            })
            .collect::<String>()
    }));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(texts)]).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_data_page_size_limit(64 * 1024)
        .set_write_batch_size(64);
    let (dir, path) = write_file(&batch, properties);
    (dir, path, batch)
}

const KIB: u64 = 1024;

#[test]
fn a_read_of_a_few_rows_takes_only_their_pages_from_the_file() {
    let (_dir, path, batch) = write_file_of_incompressible_text(50_000);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let rows = vec![20_000, 20_001, 45_000];

    let (read, stats) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert!(file_size > 5_000 * KIB, "{file_size}");
    assert!(stats.disk_bytes < 400 * KIB, "{}", stats.disk_bytes);
    assert_read_rows_are(&read, &batch, &rows);
    dispatch.exit();
}

#[test]
fn a_read_of_every_row_takes_whole_chunks_from_the_file() {
    let (_dir, path, _) = write_file_of_incompressible_text(50_000);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);

    let (read, stats) = read_selection(&dispatch, &table, RowSelection::All);

    assert_eq!(read.num_rows(), 50_000);
    assert!(stats.disk_bytes > 5_000 * KIB, "{}", stats.disk_bytes);
    dispatch.exit();
}

/// Rows of a page that an earlier read brought in are served from memory,
/// which is what lets a caller walk a row group a range of rows at a time.
#[test]
fn a_later_read_of_other_rows_of_a_page_already_read_does_no_io() {
    let (_dir, path, batch) = write_file_of_incompressible_text(50_000);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(vec![20_000].into()),
    );

    let (read, stats) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(vec![20_001].into()),
    );

    assert_eq!(stats.disk_bytes, 0);
    assert_read_rows_are(&read, &batch, &[20_001]);
    dispatch.exit();
}

/// A dataflow may read one row group several times, for other rows each
/// time, as a caller walking a row group a range of rows at a time does.
#[test]
fn two_reads_of_one_row_group_in_one_dataflow_return_the_rows_of_each() {
    let (_dir, path, batch) = write_file_of_incompressible_text(50_000);
    let dispatch = Dispatch::spin_up(1, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let requests = [vec![100, 101], vec![40_000]]
        .map(|rows| QueryRowGroupMetadata::new(&table, 0, RowSelection::Indices(rows.into())));

    let batches = fetch_row_groups(
        values_input(dispatch.dispatcher(), requests),
        &table,
        Projection::columns([0, 1]),
        dispatch::RECORD_BATCH_SIZE,
        false,
    )
    .record_batches()
    .collect()
    .unwrap();

    let read = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
    let ids = read_ids(&read);
    assert_eq!(ids.len(), 3);
    assert!([100, 101, 40_000].iter().all(|id| ids.contains(id)));
    assert_read_rows_are(&read, &batch, &ids);
    dispatch.exit();
}

/// The `id` column of rows read from [`write_file_of_incompressible_text`],
/// which is each row's position in the file.
fn read_ids(read: &RecordBatch) -> Vec<u32> {
    read.column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .map(|id| id.unwrap() as u32)
        .collect()
}

/// Late materialization reads the columns of the rows that survived a filter,
/// here three rows the scan above it kept.
#[test]
fn materializing_a_few_rows_takes_only_their_pages_from_the_file() {
    let (_dir, path, batch) = write_file_of_incompressible_text(50_000);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let surviving_rows =
        table_input(dispatch.dispatcher(), &table, Projection::columns([]), true).limit(3, 30_000);

    let (batches, stats) = materialize(surviving_rows, table.clone(), Projection::columns([0, 1]))
        .collect_with_stats()
        .unwrap();

    let read = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
    let ids = read_ids(&read);
    assert_eq!(ids.len(), 3);
    assert!(stats.disk_bytes < 600 * KIB, "{}", stats.disk_bytes);
    assert_read_rows_are(&read, &batch, &ids);
    dispatch.exit();
}

/// A file the engine wrote, with a shredded variant column: its leaves span
/// different numbers of pages, and the rows around the page edges of its
/// widest leaf read back as arrow-rs reads them.
#[test]
fn selected_rows_of_an_engine_written_variant_column_read_back() {
    let documents = (0..60_000)
        .map(|row| format!(r#"{{"user": {row}, "message": "message number {row:012} of a log"}}"#))
        .collect();
    let file = write(
        vec![TaggedJsonItem {
            file_tag: 7,
            rows: documents,
        }],
        100_000,
    )
    .remove(0);
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    std::fs::write(&path, &file).unwrap();
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let table = load_table(&dispatch, &path);
    let rows = rows_at_page_edges(&table);

    let (read, _) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );

    assert!(rows.len() > 2, "{rows:?}");
    assert_read_rows_are(&read, &read_back(&file), &rows);
    dispatch.exit();
}

/// A file of many small pages has an offset index larger than the stretch of
/// the file's end that is read to find the footer.
#[test]
fn an_offset_index_larger_than_the_footer_probe_is_loaded() {
    let ids = Int64Array::from_iter_values(0..200_000);
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids)]).unwrap();
    let properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_data_page_row_count_limit(20)
        .set_write_batch_size(20);
    let (_dir, path) = write_file(&batch, properties);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);

    let table = load_table(&dispatch, &path);

    assert_eq!(page_counts(&table), vec![10_000]);
    let rows = vec![19, 20, 199_999];
    let (read, _) = read_selection(
        &dispatch,
        &table,
        RowSelection::Indices(rows.clone().into()),
    );
    assert_read_rows_are(&read, &batch, &rows);
    dispatch.exit();
}

/// A file of many columns has a footer larger than the stretch of the file's
/// end that is read to find it, and its offset index sits before that.
#[test]
fn the_offset_index_of_a_footer_larger_than_the_probe_is_loaded() {
    let column_count = 1_500;
    let fields: Vec<Field> = (0..column_count)
        .map(|column| Field::new(format!("column_{column}"), DataType::Int64, false))
        .collect();
    let columns: Vec<ArrayRef> = (0..column_count as i64)
        .map(|column| Arc::new(Int64Array::from_iter_values(column..column + 60)) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_data_page_row_count_limit(20)
        .set_write_batch_size(20);
    let (_dir, path) = write_file(&batch, properties);
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);

    let table = load_table(&dispatch, &path);

    assert_eq!(page_counts(&table), vec![3; column_count]);
    dispatch.exit();
}
