//! End-to-end tests for writing JSON as a shredded variant column: drive the
//! whole pipeline the ingest sink drives ([`encode_items`]) and read the finished
//! files back with arrow-rs's strict reader — the oracle for "another engine can
//! read this".
//!
//! The two paths that write variants both come through here. Ingest hands the
//! pipeline unshredded documents ([`JsonItem`]); compaction hands it batches read
//! back out of files that each shredded differently ([`ShreddedItem`]). The
//! pipeline is the same either way, which is the point of normalizing on the way
//! in.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{ArrowError, Schema};
use dispatch::{BUFFER_SIZE, Dispatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet_variant_compute::{
    ShreddedSchemaBuilder, VariantArray, json_to_variant, shred_variant, unshred_variant,
};
use parquet_variant_json::VariantToJson;

use super::{EncodedFile, ToRecordBatch, encode_items};

/// A 64 MiB file-cache ring, as the other in-crate write tests use.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

/// The column name every test writes its documents under.
const COLUMN: &str = "attrs";

/// One flush's worth of JSON documents, converted to an unshredded variant
/// column — what a producer feeding JSON into the pipeline looks like.
struct JsonItem(Vec<String>);

impl ToRecordBatch for JsonItem {
    fn num_rows(&self) -> usize {
        self.0.len()
    }

    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        Ok(Some(variant_batch(&self.0, None)?))
    }
}

/// Documents already shredded on `paths` — what compaction reads back out of a
/// file that made its own shredding choice.
struct ShreddedItem {
    rows: Vec<String>,
    paths: Vec<&'static str>,
}

impl ToRecordBatch for ShreddedItem {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        Ok(Some(variant_batch(&self.rows, Some(&self.paths))?))
    }
}

/// A one-column batch of `rows` as a variant, shredded on `paths` if given.
fn variant_batch(rows: &[String], paths: Option<&[&str]>) -> Result<RecordBatch, ArrowError> {
    let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
    let mut variants = json_to_variant(&json)?;
    if let Some(paths) = paths {
        let mut builder = ShreddedSchemaBuilder::new();
        for path in paths {
            builder = builder.with_path(*path, &arrow_schema::DataType::Int64)?;
        }
        variants = shred_variant(&variants, &builder.build())?;
    }
    let schema = Schema::new(vec![variants.field(COLUMN)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(variants.into_inner())])
}

/// Run the write pipeline over `items` (one unpartitioned, unsorted file stream)
/// and return the finished files' bytes. `rows_per_file` caps a file's rows, so a
/// test can force more than one file.
fn write<T: ToRecordBatch>(items: Vec<T>, rows_per_file: usize) -> Vec<Vec<u8>> {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let files: Vec<EncodedFile> = encode_items(
        dispatch.dispatcher(),
        items,
        Arc::from([]),
        Arc::from([]),
        rows_per_file,
        1,
    )
    .collect()
    .unwrap();
    dispatch.exit();
    files.into_iter().map(|f| f.bytes).collect()
}

/// Read a file back through arrow-rs's strict reader, as another engine would.
fn read_back(bytes: &[u8]) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// A file's first row group's statistics per leaf, keyed by dotted path.
fn leaf_stats(bytes: &[u8]) -> Vec<(String, Option<u64>, bool)> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    let row_group = reader.metadata().row_group(0);
    (0..row_group.num_columns())
        .map(|i| {
            let column = row_group.column(i);
            let stats = column.statistics();
            (
                column.column_path().string(),
                stats.and_then(|s| s.null_count_opt()),
                stats.is_some_and(|s| s.min_bytes_opt().is_some()),
            )
        })
        .collect()
}

/// A file's leaf column paths, dotted — the shape the writer actually stamped
/// into the footer.
fn leaf_paths(bytes: &[u8]) -> Vec<String> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|c| c.path().string())
        .collect()
}

/// Render the variant column back to one JSON string per row, exactly as a query
/// does: fold the typed leaves back into the binary value, then render each row.
/// The documents must survive shredding whole.
fn documents(batch: &RecordBatch) -> Vec<String> {
    let variants = VariantArray::try_new(batch.column(0).as_ref()).unwrap();
    let variants = unshred_variant(&variants).unwrap();
    (0..variants.len())
        .map(|row| {
            let mut text = Vec::new();
            variants.value(row).to_json(&mut text).unwrap();
            String::from_utf8(text).unwrap()
        })
        .collect()
}

fn rows(docs: &[&str]) -> Vec<String> {
    docs.iter().map(|d| d.to_string()).collect()
}

/// JSON fed in as a variant column comes out as a real shredded VARIANT: the
/// footer carries a typed leaf per agreed path beside the binary fallback, and a
/// strict reader accepts the file.
#[test]
fn json_writes_as_a_shredded_variant_column() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"id": 1, "name": "a"}"#,
            r#"{"id": 2, "name": "b"}"#,
        ]))],
        usize::MAX,
    );

    assert_eq!(files.len(), 1);
    assert_eq!(
        leaf_paths(&files[0]),
        vec![
            "attrs.metadata",
            "attrs.value",
            "attrs.typed_value.id.value",
            "attrs.typed_value.id.typed_value",
            "attrs.typed_value.name.value",
            "attrs.typed_value.name.typed_value",
        ]
    );
}

/// Shredding never loses a document: the rows read back are the rows written,
/// both the shredded fields and the ones left in the binary fallback.
#[test]
fn a_shredded_document_reads_back_whole() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"id": 1, "extra": true}"#,
            r#"{"id": 2, "extra": false}"#,
        ]))],
        usize::MAX,
    );

    let batch = read_back(&files[0]);

    // `id` shredded into a typed leaf, `extra` (a boolean, which has no leaf
    // type) stayed in `value` — and both come back.
    assert_eq!(
        documents(&batch),
        vec![r#"{"extra":true,"id":1}"#, r#"{"extra":false,"id":2}"#]
    );
}

/// A row whose value at a shredded path is of another type keeps that value in
/// the binary fallback rather than losing it — which is why one file can shred a
/// path its rows only mostly agree on.
#[test]
fn a_row_that_disagrees_on_a_shredded_types_falls_back() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"v": 1}"#,
            r#"{"v": 2}"#,
            r#"{"v": "not a number"}"#,
        ]))],
        usize::MAX,
    );

    let batch = read_back(&files[0]);

    assert!(leaf_paths(&files[0]).contains(&"attrs.typed_value.v.typed_value".to_string()));
    assert_eq!(
        documents(&batch),
        vec![r#"{"v":1}"#, r#"{"v":2}"#, r#"{"v":"not a number"}"#]
    );
}

/// Each file shreds to its own rows: two files whose documents disagree get
/// different layouts, which the read path resolves per file.
#[test]
fn each_file_shreds_to_its_own_rows() {
    // Two rows per file, so each item lands in its own file with its own shape.
    let files = write(
        vec![
            JsonItem(rows(&[r#"{"id": 1}"#, r#"{"id": 2}"#])),
            JsonItem(rows(&[r#"{"host": "a"}"#, r#"{"host": "b"}"#])),
        ],
        2,
    );

    assert_eq!(files.len(), 2);
    let layouts: Vec<Vec<String>> = files.iter().map(|f| leaf_paths(f)).collect();
    assert!(
        layouts
            .iter()
            .any(|l| l.contains(&"attrs.typed_value.id.typed_value".to_string()))
    );
    assert!(
        layouts
            .iter()
            .any(|l| l.contains(&"attrs.typed_value.host.typed_value".to_string()))
    );
    assert_ne!(layouts[0], layouts[1]);
}

/// Documents with no path worth shredding stay as the plain `{metadata, value}`
/// pair — a legal variant column, and still readable.
#[test]
fn a_column_with_nothing_worth_shredding_stays_unshredded() {
    let files = write(
        vec![JsonItem(rows(&[r#"{"ok": true}"#, r#"{"ok": false}"#]))],
        usize::MAX,
    );

    assert_eq!(leaf_paths(&files[0]), vec!["attrs.metadata", "attrs.value"]);
    assert_eq!(
        documents(&read_back(&files[0])),
        vec![r#"{"ok":true}"#, r#"{"ok":false}"#]
    );
}

/// The re-shredding compaction depends on: batches that arrive already shredded
/// — each on a different path, as separate files would be — merge into one file
/// that shreds to what its own rows favour, not to what any input file chose.
#[test]
fn already_shredded_input_re_shreds_to_the_merged_rows() {
    let files = write(
        vec![
            ShreddedItem {
                rows: rows(&[r#"{"id": 1, "n": 10}"#, r#"{"id": 2, "n": 20}"#]),
                paths: vec!["id"],
            },
            ShreddedItem {
                rows: rows(&[r#"{"id": 3, "n": 30}"#, r#"{"id": 4, "n": 40}"#]),
                paths: vec!["n"],
            },
        ],
        usize::MAX,
    );

    // One file, and it shreds *both* paths — neither input's layout survived as
    // such; the output decided afresh from all four rows.
    assert_eq!(files.len(), 1);
    assert_eq!(
        leaf_paths(&files[0]),
        vec![
            "attrs.metadata",
            "attrs.value",
            "attrs.typed_value.id.value",
            "attrs.typed_value.id.typed_value",
            "attrs.typed_value.n.value",
            "attrs.typed_value.n.typed_value",
        ]
    );

    let mut got = documents(&read_back(&files[0]));
    got.sort();
    assert_eq!(
        got,
        vec![
            r#"{"id":1,"n":10}"#,
            r#"{"id":2,"n":20}"#,
            r#"{"id":3,"n":30}"#,
            r#"{"id":4,"n":40}"#,
        ]
    );
}

/// Every leaf carries footer statistics, shredded or not, so a reader can prune
/// row groups by any of them. The typed leaves get a range; the binary
/// `metadata`/`value` leaves have no orderable type and get a null count alone.
#[test]
fn every_leaf_carries_footer_statistics() {
    let files = write(
        vec![JsonItem(rows(&[r#"{"id": 1}"#, r#"{"id": 5}"#]))],
        usize::MAX,
    );

    let stats = leaf_stats(&files[0]);

    assert!(
        stats.iter().all(|(_, null_count, _)| null_count.is_some()),
        "every leaf needs a null count: {stats:?}"
    );
    let typed = stats
        .iter()
        .find(|(path, ..)| path == "attrs.typed_value.id.typed_value")
        .unwrap();
    assert_eq!(typed.1, Some(0), "every row shredded into the typed leaf");
    assert!(typed.2, "a typed leaf carries a min/max to prune on");
}

/// A shredded path may only be pruned by its typed leaf when the `value`
/// fallback beside it holds nothing, and a reader establishes that from the
/// fallback's null count against the row count. So an unused fallback has to
/// record one, even though it is all-null and has no range to report.
#[test]
fn an_unused_fallback_leaf_records_a_full_null_count() {
    let documents = rows(&[r#"{"id": 1}"#, r#"{"id": 5}"#, r#"{"id": 9}"#]);
    let files = write(vec![JsonItem(documents.clone())], usize::MAX);

    let stats = leaf_stats(&files[0]);

    let fallback = stats
        .iter()
        .find(|(path, ..)| path == "attrs.typed_value.id.value")
        .unwrap();
    assert_eq!(
        fallback.1,
        Some(documents.len() as u64),
        "every row shredded, so the fallback is null throughout: {stats:?}"
    );
    assert!(!fallback.2, "an all-null leaf has no min/max to report");
}

/// A nested path shreds into nested groups, and the document still reads back
/// whole through them.
#[test]
fn a_nested_path_shreds_and_reads_back() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"user": {"id": 1}}"#,
            r#"{"user": {"id": 2}}"#,
        ]))],
        usize::MAX,
    );

    assert!(
        leaf_paths(&files[0])
            .contains(&"attrs.typed_value.user.typed_value.id.typed_value".to_string())
    );
    assert_eq!(
        documents(&read_back(&files[0])),
        vec![r#"{"user":{"id":1}}"#, r#"{"user":{"id":2}}"#]
    );
}
