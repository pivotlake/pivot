//! Re-export shim: the Thrift compact-protocol codec and Parquet metadata
//! structures used to live here, but were extracted into the standalone
//! `thriftparquet` crate so the Parquet **writer** (in `ingest`) can share the
//! exact same definitions as this **reader**. The submodule paths are preserved
//! (`thrift::footer`, `thrift::headers`, `thrift::general`,
//! `thrift::parquet_thrift`) so the reader's existing imports are unchanged.

pub use thriftparquet::{footer, general, headers, parquet_thrift};
