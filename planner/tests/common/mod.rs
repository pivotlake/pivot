use std::collections::HashMap;
use std::sync::{Arc, Once};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_json::ArrayWriter;
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::Value;
use tempfile::TempDir;

use dispatch::ParquetTable;
use planner::Planner;
use planner::catalog::{Catalog, Column, Table};
use planner::types::Type;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| dispatch::init(core_affinity::get_core_ids().unwrap().len()));
}

#[derive(Debug)]
struct TestTable {
    _dir: TempDir,
    parquet_table: Arc<ParquetTable>,
    columns: Vec<Column>,
}

impl TestTable {
    fn new(columns: &[(&str, Type, ArrayRef)]) -> Self {
        let dir = TempDir::new().unwrap();

        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, _, array)| Field::new(*name, array.data_type().clone(), false))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = columns.iter().map(|(_, _, array)| array.clone()).collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

        let file_path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = std::fs::File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let parquet_table = Arc::new(ParquetTable::from_directory(dir.path()).unwrap());
        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();

        TestTable {
            _dir: dir,
            parquet_table,
            columns: cols,
        }
    }
}

impl Table for TestTable {
    fn parquet_table(&self) -> Arc<ParquetTable> {
        self.parquet_table.clone()
    }
    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }
}

#[derive(Debug)]
struct TestCatalog {
    tables: HashMap<String, Arc<dyn Table>>,
}

impl Catalog for TestCatalog {
    fn table(&self, name: &str) -> Option<Arc<dyn Table>> {
        self.tables.get(name).cloned()
    }

    fn create_table(&self, _request: planner::catalog::CreateTableRequest) {
        unreachable!("test helper catalog does not support CREATE TABLE")
    }
}

pub fn make_planner_with_table(table_name: &str, columns: &[(&str, Type, ArrayRef)]) -> Planner {
    let test_table = TestTable::new(columns);
    let table: Arc<dyn Table> = Arc::new(test_table);
    let mut tables = HashMap::new();
    tables.insert(table_name.to_string(), table);
    Planner::new(Arc::new(TestCatalog { tables }))
}

// Used only by compile tests, but this module is shared across test binaries
#[allow(dead_code)]
pub fn batches_to_json(batches: &[RecordBatch]) -> Vec<Value> {
    let mut writer = ArrayWriter::new(Vec::new());
    writer
        .write_batches(&batches.iter().collect::<Vec<_>>())
        .unwrap();
    writer.finish().unwrap();
    serde_json::from_slice(&writer.into_inner()).unwrap()
}

pub fn int_table() -> Planner {
    make_planner_with_table(
        "test",
        &[
            ("a", Type::Int32, Arc::new(Int32Array::from(vec![1, 2, 3]))),
            (
                "b",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ),
            (
                "c",
                Type::Int32,
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ),
        ],
    )
}

pub fn string_table() -> Planner {
    make_planner_with_table(
        "test",
        &[
            (
                "name",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "alice", "bob", "charlie", "dave", "alice",
                ])),
            ),
            (
                "value",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
            ),
        ],
    )
}
