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
    BuildOuter,
    /// One output row per probe row the build side holds the key of, and no
    /// build columns (see [`Join::build_output`]).
    ProbeSemi,
}

/// Hash equi-join on one or more key columns per side, one per equality
/// condition.
#[derive(Debug)]
pub struct Join {
    /// The keys' column indices in the probe (first) input's output, aligned
    /// condition by condition with `build_keys`.
    pub probe_keys: Vec<usize>,
    /// The keys' column indices in the build (second) input's output.
    pub build_keys: Vec<usize>,
    /// The type each condition's key columns arrive as (DuckDB casts a
    /// condition's mismatched sides to a common type before the join);
    /// together they pick the dispatch join's key instantiation.
    pub key_types: Vec<Type>,
    /// The probe input columns the join emits, in output order.
    pub probe_output: Vec<usize>,
    /// The build input columns the join emits after the probe columns. Always
    /// empty for a [`ProbeSemi`](JoinKind::ProbeSemi) join, which emits a probe
    /// row once however many build rows it matched, so no one build row is
    /// there to take values from.
    pub build_output: Vec<usize>,
    /// The type and nullability of each listed probe output column, in
    /// `probe_output` order. The dispatch join shapes its output from these
    /// rather than from a probed batch, so every worker emits identical
    /// schemas whether or not it ever received a batch.
    pub probe_column_types: Vec<(Type, bool)>,
    /// The type and nullability of each listed build output column, in
    /// `build_output` order.
    pub build_column_types: Vec<(Type, bool)>,
    /// Which rows reach the output.
    pub kind: JoinKind,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            JoinKind::Inner => "Join",
            JoinKind::BuildOuter => "Join[build outer]",
            JoinKind::ProbeSemi => "Join[probe semi]",
        };
        write!(
            f,
            "{kind}(probe_keys: {:?}, build_keys: {:?}, probe_output: {:?}, build_output: {:?})",
            self.probe_keys, self.build_keys, self.probe_output, self.build_output
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
            JoinKind::BuildOuter => DispatchJoinKind::BuildOuter,
            JoinKind::ProbeSemi => DispatchJoinKind::ProbeSemi,
        };
        // Both sides' output fields, named by position: an outer join
        // synthesizes NULL probe values, so field names and nullability are
        // the join's to pick rather than any input column's, and every kind
        // uses the same convention.
        let outer = matches!(self.kind, JoinKind::BuildOuter);
        let probe_fields = self
            .probe_column_types
            .iter()
            .enumerate()
            .map(|(i, (col_type, nullable))| {
                Field::new(
                    format!("probe_{i}"),
                    physical_arrow_type(col_type),
                    *nullable || outer,
                )
            })
            .collect();
        let build_fields = self
            .build_column_types
            .iter()
            .enumerate()
            .map(|(i, (col_type, nullable))| {
                Field::new(
                    format!("build_{i}"),
                    physical_arrow_type(col_type),
                    *nullable,
                )
            })
            .collect();
        let spec = JoinSpec {
            build_key_columns: self.build_keys.clone(),
            probe_key_columns: self.probe_keys.clone(),
            output_columns: JoinOutputColumns {
                probe: self.probe_output.clone(),
                build: self.build_output.clone(),
            },
            probe_fields,
            build_fields,
            kind,
        };
        let key_types: Vec<_> = self.key_types.iter().map(physical_arrow_type).collect();
        Ok(probe.join(build, &key_types, spec))
    }
}
