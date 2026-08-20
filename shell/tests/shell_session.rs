use arrow_array::{Array, Int64Array, StringViewArray};
use engine::{Command, ExecuteOptions, StatementOutput};
use shell::ShellInstance;

#[test]
fn a_datastore_is_created_at_the_requested_path_and_persists() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let parent = tempfile::tempdir().unwrap();
    let path = parent.path().join("data");
    let location = path.to_str().unwrap();
    let instance = runtime
        .block_on(async { ShellInstance::open_with_resources(location, 1, 32) })
        .unwrap();
    let engine = instance.engine();

    let (create, insert, select) = runtime.block_on(async {
        let create = engine
            .execute(
                "CREATE TABLE people (id BIGINT, name VARCHAR)".to_string(),
                ExecuteOptions::default(),
            )
            .await
            .unwrap();
        let insert = engine
            .execute(
                "INSERT INTO people VALUES (1, 'alice'), (2, 'bob')".to_string(),
                ExecuteOptions::default(),
            )
            .await
            .unwrap();
        let select = engine
            .execute(
                "SELECT id, name FROM people ORDER BY id".to_string(),
                ExecuteOptions::default(),
            )
            .await
            .unwrap();
        (create.output, insert.output, select.output)
    });

    assert!(matches!(
        create,
        StatementOutput::Command(Command::CreateTable)
    ));
    assert!(matches!(
        insert,
        StatementOutput::Command(Command::Insert { rows: 2 })
    ));
    let StatementOutput::Rows { batches, .. } = select else {
        panic!("SELECT did not return rows");
    };
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let names = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert_eq!((ids.value(0), names.value(0)), (1, "alice"));
    assert_eq!((ids.value(1), names.value(1)), (2, "bob"));

    let error = match ShellInstance::open_with_resources(location, 1, 32) {
        Ok(_) => panic!("a second instance opened the locked datastore"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("already in use"));
    assert!(error.to_string().contains(&std::process::id().to_string()));

    drop(instance);

    assert!(path.exists());
    assert!(path.join(".pivot.lock").is_file());

    let reopened = runtime
        .block_on(async { ShellInstance::open_with_resources(location, 1, 32) })
        .unwrap();
    let select = runtime.block_on(reopened.engine().execute(
        "SELECT id, name FROM people ORDER BY id".to_string(),
        ExecuteOptions::default(),
    ));
    let StatementOutput::Rows { batches, .. } = select.unwrap().output else {
        panic!("SELECT did not return rows after reopening the datastore");
    };
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let names = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert_eq!((ids.value(0), names.value(0)), (1, "alice"));
    assert_eq!((ids.value(1), names.value(1)), (2, "bob"));
}
