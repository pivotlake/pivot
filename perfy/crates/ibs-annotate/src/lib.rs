//! AMD IBS / cache-level annotation toolkit, native Rust.
//!
//! Reads `perf.data` files directly via `linux-perf-data` (no `perf script`
//! subprocess), decodes the `PERF_SAMPLE_DATA_SRC` bitfield and the AMD
//! `IBS_OP_DATA{,3}` MSRs, resolves symbols via mmap2 records + `object`'s
//! ELF symbol map, and exposes a unified [`Profile`].
//!
//! The data model in [`model`] is the same shape as the Python `ibs_annotate`
//! tool; downstream consumers (the source+asm view, the function-summary
//! sorter) can work against this crate without semantic changes.

pub mod cmd;
pub mod data_src;
pub mod ibs_msrs;
pub mod model;
pub mod parse;
pub mod reader;
pub mod sample;
pub mod symbols;

pub use model::{
    AnnotatedLine, CacheLevel, FunctionHeader, FunctionSummary, IBSRaw, InsnStats,
    InstructionLine, JumpArrow, JumpGraph, OpType, PrefetchCounts, SeparatorLine,
    SkippedSamples, SnoopStatus, SourceLine, TlbLevel,
};
pub use reader::{read_perf_data, EventClass, EventDesc, ParsedSample, Profile};
pub use symbols::{AddressSpaces, CacheStats, Mapping, ResolveTrace, SymbolCache};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("perf {0} failed: {1}")]
    Perf(String, String),
    #[error("objdump failed: {0}")]
    Objdump(String),
    #[error("readelf failed: {0}")]
    Readelf(String),
    #[error("nm failed: {0}")]
    Nm(String),
    #[error("could not auto-detect binary from perf buildid-list; pass --binary explicitly")]
    NoBinary,
    #[error("parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, Error>;
