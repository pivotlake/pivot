//! Text indexes over a table: built for the files a table already has, then
//! used by scans with a token predicate to skip row groups that cannot match.

mod common;

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch, Projection};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::datastore::{Datastore, DatastoreTransaction};
use catalog::{DEFAULT_DATASTORE_NAME, PivotCatalog};
use common::commit_datastore_transaction;
use datastore_pivotlake::PivotlakeDatastore;
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};
use planner::operator::{Input, Operator};
use planner::types::Type;
use planner::{PlanNode, Planner};

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(2, 128, None))
        .dispatcher()
        .clone()
}

/// A datastore holding `logs (id INT, message VARCHAR)` over one adopted
/// Parquet file whose row groups hold two rows each:
///
/// ```text
/// row group 0   0  GET /users ok          1  POST /users ok
/// row group 1   2  GET /orders timeout    3  DELETE /orders ok
/// row group 2   4  GET /users ok          5  NULL
/// ```
struct Logs {
    datastore: Arc<PivotlakeDatastore>,
    _database: TempDir,
    _adopted_files: TempDir,
}

fn logs() -> Logs {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("message", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..6)) as ArrayRef,
            Arc::new(StringViewArray::from(vec![
                Some("GET /users ok"),
                Some("POST /users ok"),
                Some("GET /orders timeout"),
                Some("DELETE /orders ok"),
                Some("GET /users ok"),
                None,
            ])) as ArrayRef,
        ],
    )
    .unwrap();
    let adopted_files = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let file = File::create(adopted_files.path().join("logs.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let database = TempDir::new().unwrap();
    let datastore =
        PivotlakeDatastore::open(&database.path().to_string_lossy(), &dispatcher()).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "logs".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    col_type: Type::Int32,
                },
                Column {
                    name: "message".to_string(),
                    col_type: Type::Utf8,
                },
            ],
            options: HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                adopted_files.path().to_string_lossy().into_owned(),
            )]),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    Logs {
        datastore,
        _database: database,
        _adopted_files: adopted_files,
    }
}

impl Logs {
    fn create_text_index(&self, column: &str) -> usize {
        self.datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("logs"))
            .unwrap()
            .create_text_index(column)
            .unwrap()
    }

    fn plan(&self, sql: &str) -> (planner::Plan, Arc<dyn planner::catalog::CatalogTransaction>) {
        let catalog = Arc::new(
            PivotCatalog::new(
                HashMap::from([(
                    DEFAULT_DATASTORE_NAME.to_string(),
                    self.datastore.clone() as Arc<dyn Datastore>,
                )]),
                DEFAULT_DATASTORE_NAME.to_string(),
                common::trust_metastore(),
            )
            .unwrap(),
        );
        let transaction = catalog.begin_transaction();
        let mut planner = Planner::from_datastore_names(
            vec![DEFAULT_DATASTORE_NAME.to_string()],
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .unwrap();
        let plan = planner.plan(sql, transaction.clone()).unwrap();
        (plan, transaction)
    }

    /// The `id`s `sql` returns, ascending.
    fn ids(&self, sql: &str) -> Vec<i32> {
        let (plan, transaction) = self.plan(sql);
        let batches = plan
            .compile(&dispatcher(), transaction.as_ref())
            .unwrap()
            .collect()
            .unwrap();
        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// How many rows the scan under `sql` reads, before the query's own filter
    /// sees them: what is left after row groups are skipped.
    fn rows_scanned(&self, sql: &str) -> usize {
        let (plan, _transaction) = self.plan(sql);
        let input = first_input(&plan.root).expect("the query scans a table");
        input
            .table
            .compile_scan(&dispatcher(), Projection::all(0), Vec::new(), false)
            .unwrap()
            .collect()
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
    }
}

fn first_input(node: &PlanNode) -> Option<&Input> {
    if let Operator::Input(input) = &node.operator {
        return Some(input);
    }
    node.inputs.iter().find_map(first_input)
}

const TIMEOUTS: &str = "SELECT id FROM logs WHERE has_token(message, 'timeout')";

#[test]
fn creating_an_index_builds_one_for_each_existing_file() {
    let logs = logs();

    let built = logs.create_text_index("message");

    assert_eq!(built, 1);
}

#[test]
fn creating_an_index_again_builds_nothing_new() {
    let logs = logs();
    logs.create_text_index("message");

    let built_again = logs.create_text_index("message");

    assert_eq!(built_again, 0);
}

#[test]
fn a_token_query_skips_the_row_groups_its_index_rules_out() {
    let logs = logs();
    let scanned_without_index = logs.rows_scanned(TIMEOUTS);
    logs.create_text_index("message");

    let scanned_with_index = logs.rows_scanned(TIMEOUTS);

    assert_eq!(scanned_without_index, 6);
    assert_eq!(scanned_with_index, 2, "only row group 1 holds `timeout`");
}

#[test]
fn a_token_query_returns_the_same_rows_with_and_without_an_index() {
    let logs = logs();
    let without_index = logs.ids(TIMEOUTS);
    logs.create_text_index("message");

    let with_index = logs.ids(TIMEOUTS);

    assert_eq!(without_index, vec![2]);
    assert_eq!(with_index, without_index);
}

#[test]
fn all_and_any_token_queries_skip_row_groups_too() {
    let logs = logs();
    logs.create_text_index("message");

    let all =
        logs.rows_scanned("SELECT id FROM logs WHERE has_all_tokens(message, 'DELETE orders')");
    let any = logs.rows_scanned("SELECT id FROM logs WHERE has_any_token(message, 'POST timeout')");
    let absent = logs.rows_scanned("SELECT id FROM logs WHERE has_token(message, 'missing')");

    assert_eq!(all, 2, "row group 1");
    assert_eq!(any, 4, "row groups 0 and 1");
    assert_eq!(absent, 0);
}

#[test]
fn a_query_on_a_common_token_still_returns_every_match() {
    let logs = logs();
    logs.create_text_index("message");

    let ids = logs.ids("SELECT id FROM logs WHERE has_all_tokens(message, 'GET users')");

    assert_eq!(ids, vec![0, 4]);
}

#[test]
fn an_index_on_a_column_that_is_not_text_is_refused() {
    let logs = logs();

    let error = logs
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("logs"))
        .unwrap()
        .create_text_index("id")
        .unwrap_err();

    assert!(error.to_string().contains("not a text column"));
}
