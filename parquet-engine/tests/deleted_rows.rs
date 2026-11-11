//! Rows a table format's row-level deletes removed from a file: marked on
//! the row groups read from its footer, and left out of what a scan returns.

use std::fs::File;
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use object_storage::{FileRef, ObjectPath};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet_engine::{FileRowGroups, ParquetTable};
use tempfile::TempDir;

/// A one-worker dispatch pool shared by this file's tests, for the footer
/// reads and the scans they drive.
fn test_dispatcher() -> &'static DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 32, None))
        .dispatcher()
}

/// Write `batches` as one Parquet file, a row group per batch, and read its
/// footer.
fn write_parquet(batches: &[RecordBatch]) -> (TempDir, ParquetTable) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    let properties = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        batches[0].schema(),
        Some(properties),
    )
    .unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    let table = ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap();
    (dir, table)
}

/// The footer of a file of row groups holding `rows` rows each.
fn load_file_of_row_groups(rows: &[i64]) -> (TempDir, FileRowGroups) {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batches: Vec<RecordBatch> = rows
        .iter()
        .map(|row_count| {
            let values: ArrayRef = Arc::new(Int64Array::from(vec![0; *row_count as usize]));
            RecordBatch::try_new(schema.clone(), vec![values]).unwrap()
        })
        .collect();
    let (dir, table) = write_parquet(&batches);
    let file = FileRowGroups {
        file: FileRef {
            path: ObjectPath::new("data.parquet"),
            size: 0,
        },
        row_groups: table.row_groups().to_vec(),
    };
    (dir, file)
}

/// The deleted rows of each of `file`'s row groups, by position in the
/// row group; `None` for a row group handed none.
fn deleted_per_row_group(file: &FileRowGroups) -> Vec<Option<Vec<u32>>> {
    file.row_groups
        .iter()
        .map(|row_group| {
            row_group.deleted_rows.as_ref().map(|_| {
                row_group
                    .deleted_rows_in(0..row_group.num_rows as u32)
                    .collect()
            })
        })
        .collect()
}

/// A table of one row group whose `doc` variant column holds `documents`
/// unshredded, with the rows `deleted` marked deleted.
fn write_documents_with_deleted(
    documents: &[&str],
    deleted: &[u32],
) -> (TempDir, Arc<ParquetTable>) {
    let json: ArrayRef = Arc::new(StringArray::from(documents.to_vec()));
    let variant = parquet_variant_compute::json_to_variant(&json).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![variant.field("doc")])),
        vec![Arc::new(variant.into_inner()) as ArrayRef],
    )
    .unwrap();
    let (dir, mut table) = write_parquet(&[batch]);
    let row_group = Arc::get_mut(&mut table.row_groups_mut()[0]).unwrap();
    row_group.deleted_rows = Some(Arc::new(
        deleted
            .iter()
            .map(|row| row_group.file_row_offset + *row as u64)
            .collect(),
    ));
    (dir, Arc::new(table))
}

#[test]
fn deleted_rows_marked_on_shared_row_groups_stay_off_the_ones_shared() {
    let container = 1 << 16;
    let (_dir, shared) = load_file_of_row_groups(&[container, container]);
    let mut marked = FileRowGroups {
        file: shared.file.clone(),
        row_groups: shared.row_groups.clone(),
    };

    marked.mark_deleted_rows([1].into_iter().collect());

    assert_eq!(deleted_per_row_group(&marked), vec![Some(vec![1]), None]);
    assert_eq!(deleted_per_row_group(&shared), vec![None, None]);
    assert!(Arc::ptr_eq(&marked.row_groups[1], &shared.row_groups[1]));
}

#[test]
fn an_extracted_sub_variant_is_read_for_the_rows_left_live() {
    let (_dir, table) = write_documents_with_deleted(
        &[r#"{"a":{"x":1}}"#, r#"{"a":{"x":2}}"#, r#"{"b":3}"#],
        &[1],
    );
    let projection = dispatch::Projection::columns_with_extracts(
        vec![0, 0],
        vec![
            Some(dispatch::VariantExtract {
                path: vec!["a".to_string()],
                as_type: None,
            }),
            Some(dispatch::VariantExtract {
                path: vec!["a".to_string(), "x".to_string()],
                as_type: Some(DataType::Int64),
            }),
        ],
    );

    let batches = parquet_engine::table_input(test_dispatcher(), &table, projection, false)
        .collect()
        .unwrap();

    let batch = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert!(matches!(batch.column(0).data_type(), DataType::Struct(_)));
    assert!(batch.column(0).is_valid(0));
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![Some(1), None])
    );
}
