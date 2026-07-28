//! [`Join`] — a hash equi-join of two inputs, inner or outer on the build side.
//!
//! The first input is the probe side and the second the build side, mirroring
//! DuckDB's convention of building the hash table from the right child (its
//! cost model puts the smaller relation there — see the planner's table
//! statistics hook). The output is the listed probe columns followed by the
//! listed build columns; DuckDB's join projection maps are folded into these
//! lists during the plan walk, and a later rewrite pass empties them when
//! nothing above the join reads any column (a bare `COUNT(*)`), letting the
//! dispatch probe skip materialization entirely.
//!
//! That same convention is why an outer join arrives here outer on the *build*
//! side: DuckDB writes `RIGHT` for a `LEFT JOIN` whose preserved relation is
//! the smaller one and therefore ends up as its right child.

use crate::compile::Error;
use crate::types::{Type, physical_arrow_type};
use arrow_schema::Field;
use dispatch::{JoinOutputColumns, JoinSpec, RecordBatchOperatorSpec};
use std::fmt;

/// Hash equi-join on a single key column per side.
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
    /// The types of [`probe_output`](Self::probe_output), which shape the NULLs
    /// an unmatched build row gets in place of probe values. Present exactly
    /// when every build row must reach the output, matched or not (DuckDB's
    /// `RIGHT`), which is the only case that has such rows to shape.
    pub outer_probe_types: Option<Vec<Type>>,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.outer_probe_types {
            Some(_) => "Join[build outer]",
            None => "Join",
        };
        write!(
            f,
            "{kind}(probe_key: {}, build_key: {}, probe_output: {:?}, build_output: {:?})",
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
        // The probe side's output fields, named by position: an unmatched build
        // row's probe values are NULLs the join synthesizes, so their names are
        // the join's to pick rather than any input column's.
        let outer_probe_fields = self.outer_probe_types.as_ref().map(|types| {
            types
                .iter()
                .enumerate()
                .map(|(i, col_type)| {
                    Field::new(format!("probe_{i}"), physical_arrow_type(col_type), true)
                })
                .collect()
        });
        let spec = JoinSpec {
            build_key_column: self.build_key,
            probe_key_column: self.probe_key,
            output_columns: JoinOutputColumns {
                probe: self.probe_output.clone(),
                build: self.build_output.clone(),
            },
            outer_probe_fields,
        };
        Ok(probe.join(build, &physical_arrow_type(&self.key_type), spec))
    }
}
