//! [`Join`] — a hash equi-join of two inputs: inner, outer on either side, or
//! semi on the probe side.
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
//! That same convention decides which side an outer join preserves: DuckDB
//! writes `RIGHT` for a `LEFT JOIN` whose preserved relation is the smaller
//! one and therefore ends up as its right child (the build), and keeps `LEFT`
//! when the preserved relation is the larger one (the probe). A semi join
//! always keeps rows of its left child, the probe, because its cost model
//! wants the subquery it came from, being the smaller side, on the build side.

use crate::compile::{Error, ExprEvalFn};
use crate::expression::Expression;
use crate::types::{Type, physical_arrow_type};
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_schema::Field;
use dispatch::JoinKind as DispatchJoinKind;
use dispatch::{JoinResidual, JoinSpec, RangeCompare, RangeJoinSpec, RecordBatchOperatorSpec};
use std::fmt;
use std::sync::Arc;

/// Which rows a join emits.
#[derive(Debug)]
pub enum JoinKind {
    /// One output row per matching (probe row, build row) pair.
    Inner,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus every build row no
    /// probe row matched, its probe columns NULL.
    BuildOuter,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus every probe row
    /// nothing matched, its build columns NULL.
    ProbeOuter,
    /// One output row per probe row the build side holds the key of, and no
    /// build columns (see [`Join::build_output`]).
    ProbeSemi,
    /// Inner join on one `<`/`<=`/`>`/`>=` comparison (`probe key OP build
    /// key`) instead of equalities: the build side is sorted by key and each
    /// probe row matches a contiguous run of it. Always a single condition,
    /// so `probe_keys`/`build_keys`/`key_types` hold exactly one entry.
    Range(RangeCompare),
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
    /// Conditions of the join that are not left/right equalities (e.g. an OR
    /// referencing both sides), implicitly ANDed; a key-matched pair they
    /// reject is not a match. Their column refs are bound against the
    /// concatenation of both inputs' full outputs (probe columns first), the
    /// layout the dispatch join evaluates them in.
    pub residual_filters: Vec<Expression>,
    /// Which rows reach the output.
    pub kind: JoinKind,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            JoinKind::Inner => "Join",
            JoinKind::BuildOuter => "Join[build outer]",
            JoinKind::ProbeOuter => "Join[probe outer]",
            JoinKind::ProbeSemi => "Join[probe semi]",
            JoinKind::Range(RangeCompare::Less) => "Join[range <]",
            JoinKind::Range(RangeCompare::LessEq) => "Join[range <=]",
            JoinKind::Range(RangeCompare::Greater) => "Join[range >]",
            JoinKind::Range(RangeCompare::GreaterEq) => "Join[range >=]",
        };
        write!(
            f,
            "{kind}(probe_keys: {:?}, build_keys: {:?}, probe_output: {:?}, build_output: {:?}",
            self.probe_keys, self.build_keys, self.probe_output, self.build_output
        )?;
        if !self.residual_filters.is_empty() {
            let conditions = self
                .residual_filters
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" AND ");
            write!(f, ", residual: {conditions}")?;
        }
        write!(f, ")")
    }
}

impl Join {
    pub(crate) fn compile(
        &self,
        probe: RecordBatchOperatorSpec,
        build: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Both sides' output fields, named by position: an outer join
        // synthesizes NULL values for its non-preserved side, so field names
        // and nullability are the join's to pick rather than any input
        // column's, and every kind uses the same convention.
        let build_outer = matches!(self.kind, JoinKind::BuildOuter);
        let probe_outer = matches!(self.kind, JoinKind::ProbeOuter);
        let probe_fields = self
            .probe_column_types
            .iter()
            .enumerate()
            .map(|(i, (col_type, nullable))| {
                Field::new(
                    format!("probe_{i}"),
                    physical_arrow_type(col_type),
                    *nullable || build_outer,
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
                    *nullable || probe_outer,
                )
            })
            .collect();
        if let JoinKind::Range(compare) = self.kind {
            let spec = RangeJoinSpec {
                probe_key_index: self.probe_keys[0],
                build_key_index: self.build_keys[0],
                compare,
                probe_output_indices: self.probe_output.clone(),
                build_output_indices: self.build_output.clone(),
                probe_fields,
                build_fields,
            };
            return Ok(probe.range_join(build, &physical_arrow_type(&self.key_types[0]), spec));
        }
        let kind = match &self.kind {
            JoinKind::Inner => DispatchJoinKind::Inner,
            JoinKind::BuildOuter => DispatchJoinKind::BuildOuter,
            JoinKind::ProbeOuter => DispatchJoinKind::ProbeOuter,
            JoinKind::ProbeSemi => DispatchJoinKind::ProbeSemi,
            JoinKind::Range(_) => unreachable!("compiled above"),
        };
        let spec = JoinSpec {
            probe_key_indices: self.probe_keys.clone(),
            build_key_indices: self.build_keys.clone(),
            probe_output_indices: self.probe_output.clone(),
            build_output_indices: self.build_output.clone(),
            probe_fields,
            build_fields,
            kind,
            residual_filters: self.compile_residual_filters()?,
        };
        let key_types: Vec<_> = self.key_types.iter().map(physical_arrow_type).collect();
        Ok(probe.join(build, &key_types, spec))
    }

    /// Compile the residual conditions into the dispatch join's pair
    /// predicate: each condition evaluated over the gathered pair batch, the
    /// verdicts ANDed with SQL null semantics (the dispatch side rejects a
    /// NULL verdict, like any non-TRUE condition).
    fn compile_residual_filters(&self) -> Result<Option<JoinResidual>, Error> {
        if self.residual_filters.is_empty() {
            return Ok(None);
        }
        let factories = Arc::new(
            self.residual_filters
                .iter()
                .map(|condition| condition.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(Some(JoinResidual(Arc::new(move || {
            let mut evals: Vec<ExprEvalFn> = factories.iter().map(|factory| factory()).collect();
            Box::new(move |pairs: &RecordBatch| {
                let mut combined: Option<BooleanArray> = None;
                for eval in &mut evals {
                    let result = eval(pairs);
                    let (array, is_scalar) = result.as_datum().get();
                    let mask = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .expect("join residual conditions evaluate to booleans");
                    // A constant-folded condition comes back as one scalar
                    // verdict; broadcast it over the pairs.
                    let mask = if is_scalar {
                        let verdict = mask.is_valid(0).then(|| mask.value(0));
                        BooleanArray::from(vec![verdict; pairs.num_rows()])
                    } else {
                        mask.clone()
                    };
                    combined = Some(match combined {
                        None => mask,
                        Some(so_far) => arrow::compute::and_kleene(&so_far, &mask)
                            .expect("residual verdicts have one row per pair"),
                    });
                }
                combined.expect("a compiled residual has at least one condition")
            })
        }))))
    }
}
