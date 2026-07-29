//! [`Join`] — a hash equi-join of two inputs: inner, outer on the build side,
//! or semi on the probe side.
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
//! the smaller one and therefore ends up as its right child. A semi join
//! arrives the other way round — DuckDB's `SEMI` keeps rows of its left child,
//! the probe — because its cost model wants the subquery it came from, being
//! the smaller side, on the build side.

use crate::compile::Error;
use crate::types::{Type, physical_arrow_type};
use arrow_schema::Field;
use dispatch::JoinKind as DispatchJoinKind;
use dispatch::{JoinOutputColumns, JoinSpec, RecordBatchOperatorSpec};
use std::fmt;

/// Which rows a join emits.
#[derive(Debug)]
pub enum JoinKind {
    /// One output row per matching (probe row, build row) pair.
    Inner,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus every build row no
    /// probe row matched, its probe columns NULL.
    BuildOuter {
        /// The types of the join's probe output columns, which shape the NULLs
        /// an unmatched build row gets in place of probe values.
        probe_types: Vec<Type>,
    },
    /// One output row per probe row the build side holds the key of, and no
    /// build columns (see [`Join::build_output`]).
    ProbeSemi,
}

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
    /// The build input columns the join emits after the probe columns. Always
    /// empty for a [`ProbeSemi`](JoinKind::ProbeSemi) join, which emits a probe
    /// row once however many build rows it matched, so no one build row is
    /// there to take values from.
    pub build_output: Vec<usize>,
    /// Which rows reach the output.
    pub kind: JoinKind,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            JoinKind::Inner => "Join",
            JoinKind::BuildOuter { .. } => "Join[build outer]",
            JoinKind::ProbeSemi => "Join[probe semi]",
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
        let kind = match &self.kind {
            JoinKind::Inner => DispatchJoinKind::Inner,
            // The probe side's output fields, named by position: an unmatched
            // build row's probe values are NULLs the join synthesizes, so their
            // names are the join's to pick rather than any input column's.
            JoinKind::BuildOuter { probe_types } => DispatchJoinKind::BuildOuter {
                probe_fields: probe_types
                    .iter()
                    .enumerate()
                    .map(|(i, col_type)| {
                        Field::new(format!("probe_{i}"), physical_arrow_type(col_type), true)
                    })
                    .collect(),
            },
            JoinKind::ProbeSemi => DispatchJoinKind::ProbeSemi,
        };
        let spec = JoinSpec {
            build_key_column: self.build_key,
            probe_key_column: self.probe_key,
            output_columns: JoinOutputColumns {
                probe: self.probe_output.clone(),
                build: self.build_output.clone(),
            },
            kind,
        };
        Ok(probe.join(build, &physical_arrow_type(&self.key_type), spec))
    }
}
