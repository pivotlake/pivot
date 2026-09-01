//! `SET`/`RESET` through the shell's executor: the statement is parsed, not
//! applied, and hands the frontend the variable name and value it toggles its
//! session state by (the shell reads `pivot_stats` this way).

use bin::execution::{ExecuteOptions, StatementOutput};
use bin::shell::ShellInstance;

#[test]
fn set_and_reset_hand_back_the_variable_name_and_value() {
    let directory = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_resources(directory.path().to_str().unwrap(), 1, 32)
        })
        .unwrap();

    let (set, reset) = runtime.block_on(async {
        (
            execute(instance.executor(), "SET pivot_stats = 1").await,
            execute(instance.executor(), "RESET pivot_stats").await,
        )
    });

    assert_set(set, "pivot_stats", Some("1"));
    assert_set(reset, "pivot_stats", None);
}

async fn execute(
    executor: &bin::execution::Executor,
    sql: &str,
) -> StatementOutput<arrow_array::RecordBatch> {
    executor
        .execute(sql.to_string(), ExecuteOptions::default())
        .await
        .unwrap()
        .output
}

fn assert_set(output: StatementOutput<arrow_array::RecordBatch>, name: &str, value: Option<&str>) {
    match output {
        StatementOutput::Set {
            name: actual_name,
            value: actual_value,
        } => {
            assert_eq!(actual_name, name);
            assert_eq!(actual_value.as_deref(), value);
        }
        other => panic!("expected a SET output, got {other:?}"),
    }
}
