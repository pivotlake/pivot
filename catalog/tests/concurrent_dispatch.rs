//! Two catalogs compiling and running plain queries concurrently over one
//! shared worker pool. Regression for a lost-dataflow race: two dataflows
//! dispatched while a worker was parked coalesce into one wake, and a worker
//! that received only one builder per pass re-parked with the second query's
//! builder stranded in its queue, hanging that query (its sibling operators
//! wait forever for the stranded worker's part).

mod common;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use arrow_array::{Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::ParquetCatalog;
use planner::Planner;
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest};
use planner::types::Type;

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(8, 512, None))
        .dispatcher()
        .clone()
}

fn make_catalog() -> (TempDir, Arc<ParquetCatalog>) {
    let dir = TempDir::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = std::fs::File::create(dir.path().join("data.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
    writer
        .write(
            &RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(StringViewArray::from(vec!["a", "b", "c"])),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    writer.close().unwrap();

    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    let mut options = HashMap::new();
    options.insert(
        "path".to_string(),
        dir.path().to_string_lossy().into_owned(),
    );
    catalog
        .create_table(
            CreateTableRequest {
                name: "t".to_string(),
                columns: vec![
                    Column {
                        name: "id".to_string(),
                        col_type: Type::Int64,
                    },
                    Column {
                        name: "name".to_string(),
                        col_type: Type::Utf8,
                    },
                ],
                options,
                if_not_exists: false,
            },
            &dispatcher(),
        )
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    (dir, catalog)
}

fn run_count(catalog: &Arc<ParquetCatalog>) -> usize {
    let mut planner = Planner::new(catalog.clone() as Arc<dyn PlannerCatalog>);
    planner
        .plan("SELECT id FROM t")
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

#[test]
fn concurrent_plain_queries_on_two_catalogs() {
    let (_dir_a, catalog_a) = make_catalog();
    let (_dir_b, catalog_b) = make_catalog();

    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            for _ in 0..200 {
                assert_eq!(run_count(&catalog_a), 3);
            }
        });
        let b = scope.spawn(|| {
            for _ in 0..200 {
                assert_eq!(run_count(&catalog_b), 3);
            }
        });
        a.join().unwrap();
        b.join().unwrap();
    });
}
