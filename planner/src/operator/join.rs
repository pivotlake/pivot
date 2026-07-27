//! [`Join`] — an inner hash equi-join of two inputs.
//!
//! The first input is the probe side and the second the build side, mirroring
//! DuckDB's convention of building the hash table from the right child (its
//! cost model puts the smaller relation there — see the planner's table
//! statistics hook). The output is the listed probe columns followed by the
//! listed build columns; DuckDB's join projection maps are folded into these
//! lists during the plan walk, and a later rewrite pass empties them when
//! nothing above the join reads any column (a bare `COUNT(*)`), letting the
//! dispatch probe skip materialization entirely.

use crate::compile::Error;
use crate::types::{Type, physical_arrow_type};
use dispatch::{JoinOutputColumns, RecordBatchOperatorSpec};
use std::fmt;

/// Inner hash equi-join on a single key column per side.
#[derive(Debug)]
pub struct Join {
    /// The key's column index in the probe (first) input's output.
    pub probe_key: usize,
    /// The key's column index in the build (second) input's output.
    pub build_key: usize,
    /// The type both key columns arrive as (DuckDB casts mismatched sides to
    /// a common type before the join); picks the dispatch join's key
    /// instantiation.
    pub key_type: Type,
    /// The probe input columns the join emits, in output order.
    pub probe_output: Vec<usize>,
    /// The build input columns the join emits after the probe columns.
    pub build_output: Vec<usize>,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Join(probe_key: {}, build_key: {}, probe_output: {:?}, build_output: {:?})",
            self.probe_key, self.build_key, self.probe_output, self.build_output
        )
    }
}

impl Join {
    pub(crate) fn compile(
        &self,
        probe: RecordBatchOperatorSpec,
        build: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let output_columns = JoinOutputColumns {
            probe: self.probe_output.clone(),
            build: self.build_output.clone(),
        };
        Ok(probe.join(
            build,
            self.build_key,
            self.probe_key,
            &physical_arrow_type(&self.key_type),
            output_columns,
        ))
    }
}
