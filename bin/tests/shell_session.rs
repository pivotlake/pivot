use arrow_array::{Array, Int64Array, StringViewArray};
use bin::execution::{Command, ExecuteOptions, StatementOutput};
use bin::shell::{ShellInstance, ShellLimits};
use datastore_pivot::DEFAULT_REFRESH_INTERVAL;
use dispatch::BUFFER_SIZE;

#[test]
fn shell_limits_require_at_least_one_worker() {
    let result = ShellInstance::open_with_limits(
        "unused",
        ShellLimits {
            memory_bytes: Some(BUFFER_SIZE as u64),
            workers: Some(0),
        },
    );
    let error = match result {
        Ok(_) => panic!("shell accepted zero workers"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("worker count must be at least 1")
    );
}

#[test]
fn shell_limits_require_at_least_one_memory_slot() {
    let result = ShellInstance::open_with_limits(
        "unused",
        ShellLimits {
            memory_bytes: Some(BUFFER_SIZE as u64 - 1),
            workers: Some(1),
        },
    );
    let error = match result {
        Ok(_) => panic!("shell accepted less than one pool slot"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("must be at least 2 MiB"));
}

#[test]
fn shell_limits_reject_a_pool_larger_than_available_memory() {
    let result = ShellInstance::open_with_limits(
        "unused",
        ShellLimits {
            memory_bytes: Some(u64::MAX),
            workers: Some(1),
        },
    );
    let error = match result {
        Ok(_) => panic!("shell accepted a pool larger than available memory"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("the buffer pool needs"));
    assert!(error.to_string().contains("available right now"));
}

#[test]
fn a_table_can_be_created_in_a_relative_datastore_path() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let current_dir = std::env::current_dir().unwrap();
    let parent = tempfile::tempdir_in(&current_dir).unwrap();
    let path = parent.path().join("data");
    let location = path.strip_prefix(&current_dir).unwrap().to_str().unwrap();
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_limits(
                location,
                ShellLimits {
                    memory_bytes: Some((BUFFER_SIZE * 32) as u64),
                    workers: Some(1),
                },
            )
        })
        .unwrap();

    let create = runtime.block_on(instance.executor().execute(
        "CREATE TABLE test (value INT)".to_string(),
        ExecuteOptions::default(),
    ));

    assert!(matches!(
        create.unwrap().output,
        StatementOutput::Command(Command::CreateTable)
    ));
    assert!(path.join("_pivot_manifest.json").is_file());
}

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
        .block_on(async {
            ShellInstance::open_with_resources(location, 1, 32, DEFAULT_REFRESH_INTERVAL)
        })
        .unwrap();
    let executor = instance.executor();

    let (create, insert, select) = runtime.block_on(async {
        let create = executor
            .execute(
                "CREATE TABLE people (id BIGINT, name VARCHAR)".to_string(),
                ExecuteOptions::default(),
            )
            .await
            .unwrap();
        let insert = executor
            .execute(
                "INSERT INTO people VALUES (1, 'alice'), (2, 'bob')".to_string(),
                ExecuteOptions::default(),
            )
            .await
            .unwrap();
        let select = executor
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

    let error = match runtime.block_on(async {
        ShellInstance::open_with_resources(location, 1, 32, DEFAULT_REFRESH_INTERVAL)
    }) {
        Ok(_) => panic!("a second instance opened the locked datastore"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("already in use"));
    assert!(error.to_string().contains(&std::process::id().to_string()));

    drop(instance);

    assert!(path.exists());
    assert!(path.join(".pivot.lock").is_file());

    let reopened = runtime
        .block_on(async {
            ShellInstance::open_with_resources(location, 1, 32, DEFAULT_REFRESH_INTERVAL)
        })
        .unwrap();
    let select = runtime.block_on(reopened.executor().execute(
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
