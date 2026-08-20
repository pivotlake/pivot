//! `pivot open` over an object-store URI: the shell opens a datastore in a
//! bucket with the credentials the environment carries, and a table written
//! through it is still there when the datastore is reopened. Returns early when
//! Docker is unreachable, so the suite stays green offline.

use arrow_array::{Array, Int64Array};
use datastore_delta::test_support;
use engine::{ExecuteOptions, StatementOutput};
use shell::ShellInstance;

#[test]
fn a_datastore_is_opened_on_an_s3_uri() {
    let Some(backend) = test_support::s3("shell-open-s3") else {
        return;
    };

    let ids = write_then_reopen(&backend.root);

    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn a_datastore_is_opened_on_a_gcs_uri() {
    let Some(backend) = test_support::gcs("shell-open-gcs") else {
        return;
    };

    let ids = write_then_reopen(&backend.root);

    assert_eq!(ids, vec![1, 2]);
}

/// Create a table at `location` and insert two rows, then reopen the datastore
/// and read the ids back, so what is asserted came off the store rather than out
/// of the writing instance's memory.
fn write_then_reopen(location: &str) -> Vec<i64> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let instance = runtime
        .block_on(async { ShellInstance::open_with_resources(location, 1, 32) })
        .unwrap();
    assert_eq!(instance.location(), location);
    runtime.block_on(async {
        execute(
            instance.engine(),
            "CREATE TABLE people (id BIGINT, name VARCHAR)",
        )
        .await;
        execute(
            instance.engine(),
            "INSERT INTO people VALUES (1, 'alice'), (2, 'bob')",
        )
        .await;
    });
    drop(instance);

    let reopened = runtime
        .block_on(async { ShellInstance::open_with_resources(location, 1, 32) })
        .unwrap();
    let output = runtime
        .block_on(reopened.engine().execute(
            "SELECT id FROM people ORDER BY id".to_string(),
            ExecuteOptions::default(),
        ))
        .unwrap()
        .output;

    let StatementOutput::Rows { batches, .. } = output else {
        panic!("SELECT did not return rows from the datastore at {location}");
    };
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

async fn execute(engine: &engine::Engine, statement: &str) {
    engine
        .execute(statement.to_string(), ExecuteOptions::default())
        .await
        .unwrap();
}
