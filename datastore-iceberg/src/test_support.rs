//! The reusable object-store test harness (MinIO, fake-gcs), re-exported for
//! this crate's integration tests. Gated behind the `test-support` feature,
//! so it never enters a normal build.

pub use object_storage::test_support::*;
