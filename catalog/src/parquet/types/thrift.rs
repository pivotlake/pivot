//! Re-export shim: the Thrift compact-protocol codec and Parquet metadata
//! structures live in the standalone `thriftparquet` crate so the Parquet
//! **writer** shares the exact same definitions as this
//! **reader**. The submodule paths (`thrift::footer`, `thrift::headers`,
//! `thrift::general`, `thrift::parquet_thrift`) are preserved so the reader's
//! imports are unchanged.

pub use thriftparquet::{footer, general, headers, parquet_thrift};
