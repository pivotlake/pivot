use std::collections::HashMap;
use std::sync::Arc;

use catalog::ParquetCatalog;
use dispatch::Dispatch;
use planner::catalog::{Catalog, Column, CreateTableRequest};
use planner::types::Type;
use tempfile::TempDir;

#[test]
fn rollback_deletes_insert_parquet_files() {
    let directory = TempDir::new().unwrap();
    let dispatch = Dispatch::spin_up(2, 64, None);
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    let request = CreateTableRequest {
        name: "rollback_rows".to_string(),
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
        options: HashMap::from([(
            "path".to_string(),
            directory.path().to_str().unwrap().to_string(),
        )]),
        if_not_exists: false,
    };
    catalog
        .create_table(request, dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    let transaction = catalog.begin_transaction();
    let mut planner = planner::Planner::new(catalog.clone());
    let plan = planner
        .plan(
            "INSERT INTO rollback_rows VALUES (1, 'one'), (2, 'two')",
            transaction.clone(),
        )
        .unwrap();

    let counts = plan
        .compile(dispatch.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    catalog.rollback_transaction(transaction).unwrap();

    assert_eq!(
        counts.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
    assert!(catalog.table_files("rollback_rows").unwrap().is_empty());
    assert_eq!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "parquet"))
            .count(),
        0
    );
    dispatch.exit();
}
