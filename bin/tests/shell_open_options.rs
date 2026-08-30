use bin::execution::{Command, ExecuteOptions, StatementOutput};
use bin::shell::{DiskCacheOptions, OpenOptions, ShellInstance};

#[test]
fn open_honours_explicit_memory_workers_and_disk_cache() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let parent = tempfile::tempdir().unwrap();
    let datastore = parent.path().join("data");
    let cache_dir = parent.path().join("cache");
    std::fs::create_dir(&cache_dir).unwrap();
    let options = OpenOptions {
        memory_bytes: Some(64 * 1024 * 1024),
        workers: Some(1),
        disk_cache: Some(DiskCacheOptions {
            dir: cache_dir,
            size_bytes: Some(16 * 1024 * 1024),
        }),
    };

    let instance = runtime
        .block_on(async { ShellInstance::open(datastore.to_str().unwrap(), options) })
        .unwrap();
    let create = runtime.block_on(instance.executor().execute(
        "CREATE TABLE test (value INT)".to_string(),
        ExecuteOptions::default(),
    ));

    assert!(matches!(
        create.unwrap().output,
        StatementOutput::Command(Command::CreateTable)
    ));
}

#[test]
fn open_refuses_a_memory_budget_beyond_available_memory() {
    let parent = tempfile::tempdir().unwrap();
    let datastore = parent.path().join("data");
    let options = OpenOptions {
        memory_bytes: Some(u64::MAX),
        ..OpenOptions::default()
    };

    let error = match ShellInstance::open(datastore.to_str().unwrap(), options) {
        Ok(_) => panic!("an impossible memory budget opened anyway"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("available"), "{error}");
    assert!(
        !datastore.exists(),
        "the datastore was created before the budget check"
    );
}

#[test]
fn open_refuses_an_unopenable_disk_cache() {
    let parent = tempfile::tempdir().unwrap();
    let datastore = parent.path().join("data");
    let cache_path = parent.path().join("cache");
    std::fs::write(&cache_path, b"not a directory").unwrap();
    let options = OpenOptions {
        memory_bytes: Some(64 * 1024 * 1024),
        workers: Some(1),
        disk_cache: Some(DiskCacheOptions {
            dir: cache_path,
            size_bytes: None,
        }),
    };

    let error = match ShellInstance::open(datastore.to_str().unwrap(), options) {
        Ok(_) => panic!("a disk cache path occupied by a file opened anyway"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("disk cache"), "{error}");
}
