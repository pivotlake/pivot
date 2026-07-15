//! [`Join`] — an inner hash equi-join of two inputs.
//!
//! The first input is the probe side and the second the build side, mirroring
//! DuckDB's convention of building the hash table from the right child (its
//! cost model puts the smaller relation there — see the planner's table
//! statistics hook). The output is every probe column followed by every build
//! column; DuckDB's join projection maps, when present, are replayed as a
//! `Projection` above this node during the plan walk, so the operator itself
//! always emits the full concatenation.

use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;

/// Inner hash equi-join on a single key column per side.
#[derive(Debug)]
pub struct Join {
    /// The key's column index in the probe (first) input's output.
    pub probe_key: usize,
    /// The key's column index in the build (second) input's output.
    pub build_key: usize,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Join(probe_key: {}, build_key: {})",
            self.probe_key, self.build_key
        )
    }
}

impl Join {
    pub(crate) fn compile(
        &self,
        probe: RecordBatchOperatorSpec,
        build: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Ok(probe.join(build, self.build_key, self.probe_key))
    }
}
