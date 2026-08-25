//! [`Join`] — a hash equi-join of two inputs: inner, outer on either side,
//! or semi or anti on either side.
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

use crate::compile::{Error, ExprEvalFn, RuntimeFilterSlots};
use crate::dynamic_filter::JoinProducedFilter;
use crate::expression::Expression;
use crate::types::{Type, physical_arrow_type};
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_schema::Field;
use dispatch::JoinKind as DispatchJoinKind;
use dispatch::{
    JoinBuildFilter, JoinResidualSpec, JoinSpec, RangeCompare, RangeJoinSpec,
    RecordBatchOperatorSpec,
};
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
    /// One output row per probe row with no surviving match, and no build
    /// columns (see [`Join::build_output`]).
    ProbeAnti,
    /// One output row per build row no probe row matched, and no probe
    /// columns (see [`Join::probe_output`]).
    BuildAnti,
    /// One output row per build row some probe row matched, once however many
    /// matched it, and no probe columns (see [`Join::probe_output`]).
    BuildSemi,
    /// One output row per probe row, the listed probe columns plus a nullable
    /// boolean marker column after them: TRUE on a match, FALSE on a miss,
    /// NULL where SQL's three-valued `IN` cannot call the miss FALSE. No
    /// build columns (see [`Join::build_output`]).
    ProbeMark,
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
    /// The probe input columns the join emits, in output order. Always empty
    /// for a [`BuildAnti`](JoinKind::BuildAnti) join, whose output rows have
    /// no probe row to take values from, and for a
    /// [`BuildSemi`](JoinKind::BuildSemi) join, whose output rows have no
    /// single one.
    pub probe_output: Vec<usize>,
    /// The build input columns the join emits after the probe columns. Always
    /// empty for a [`ProbeSemi`](JoinKind::ProbeSemi) join, which emits a probe
    /// row once however many build rows it matched, so no one build row is
    /// there to take values from, and for a
    /// [`ProbeAnti`](JoinKind::ProbeAnti) join, whose output rows matched no
    /// build row at all.
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
    /// reject is not a match. Their refs address the compact concatenation of
    /// [`residual_probe_columns`](Self::residual_probe_columns) followed by
    /// [`residual_build_columns`](Self::residual_build_columns).
    pub residual_filters: Vec<Expression>,
    /// Probe input columns gathered for residual evaluation, in compact-batch
    /// order. Empty when the join has no residual or it reads no probe column.
    pub residual_probe_columns: Vec<usize>,
    /// Build input columns gathered after `residual_probe_columns` for residual
    /// evaluation.
    pub residual_build_columns: Vec<usize>,
    /// Which rows reach the output.
    pub kind: JoinKind,
    /// Filters this join's build side produces for probe-side scans: when the
    /// build seals, each listed key's min and max are published into the paired
    /// slots, and the consumer scans prune row groups outside those bounds.
    pub produced_filters: Vec<JoinProducedFilter>,
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            JoinKind::Inner => "Join",
            JoinKind::BuildOuter => "Join[build outer]",
            JoinKind::ProbeOuter => "Join[probe outer]",
            JoinKind::ProbeSemi => "Join[probe semi]",
            JoinKind::ProbeAnti => "Join[probe anti]",
            JoinKind::BuildAnti => "Join[build anti]",
            JoinKind::BuildSemi => "Join[build semi]",
            JoinKind::ProbeMark => "Join[probe mark]",
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
        if !self.produced_filters.is_empty() {
            let keys: Vec<String> = self
                .produced_filters
                .iter()
                .map(|filter| filter.key_position.to_string())
                .collect();
            write!(f, ", publishes key bounds: [{}]", keys.join(", "))?;
        }
        write!(f, ")")
    }
}

impl Join {
    pub(crate) fn compile(
        &self,
        probe: RecordBatchOperatorSpec,
        build: RecordBatchOperatorSpec,
        normalize_build_variants: bool,
        slots: &mut RuntimeFilterSlots,
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
            let key_type = physical_arrow_type(&self.key_types[0]);
            return Ok(if normalize_build_variants {
                probe.range_join_normalizing_build(build, &key_type, spec)
            } else {
                probe.range_join(build, &key_type, spec)
            });
        }
        let kind = match &self.kind {
            JoinKind::Inner => DispatchJoinKind::Inner,
            JoinKind::BuildOuter => DispatchJoinKind::BuildOuter,
            JoinKind::ProbeOuter => DispatchJoinKind::ProbeOuter,
            JoinKind::ProbeSemi => DispatchJoinKind::ProbeSemi,
            JoinKind::ProbeAnti => DispatchJoinKind::ProbeAnti,
            JoinKind::BuildAnti => DispatchJoinKind::BuildAnti,
            JoinKind::BuildSemi => DispatchJoinKind::BuildSemi,
            JoinKind::ProbeMark => DispatchJoinKind::ProbeMark,
            JoinKind::Range(_) => unreachable!("compiled above"),
        };
        // Resolve each produced filter's slots: the same registry hands the
        // consumer scans the same `Arc`s, which is the whole wiring.
        let build_filters = self
            .produced_filters
            .iter()
            .map(|filter| JoinBuildFilter {
                build_column: self.build_keys[filter.key_position],
                min_slot: slots.boundary_slot(filter.min_slot_id),
                max_slot: slots.boundary_slot(filter.max_slot_id),
            })
            .collect();
        let spec = JoinSpec {
            probe_key_indices: self.probe_keys.clone(),
            build_key_indices: self.build_keys.clone(),
            probe_output_indices: self.probe_output.clone(),
            build_output_indices: self.build_output.clone(),
            probe_fields,
            build_fields,
            kind,
            residual_filters: self.compile_residual_filters()?,
            build_filters,
        };
        let key_types: Vec<_> = self.key_types.iter().map(physical_arrow_type).collect();
        Ok(if normalize_build_variants {
            probe.join_normalizing_build(build, &key_types, spec)
        } else {
            probe.join(build, &key_types, spec)
        })
    }

    /// Compile the residual conditions into the dispatch join's pair
    /// predicate: each condition evaluated over the gathered pair batch, the
    /// verdicts ANDed with SQL null semantics (the dispatch side rejects a
    /// NULL verdict, like any non-TRUE condition).
    fn compile_residual_filters(&self) -> Result<Option<JoinResidualSpec>, Error> {
        if self.residual_filters.is_empty() {
            return Ok(None);
        }

        let factories = Arc::new(
            self.residual_filters
                .iter()
                .map(|condition| condition.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(Some(JoinResidualSpec::new(
            Arc::new(move || {
                let mut evals: Vec<ExprEvalFn> =
                    factories.iter().map(|factory| factory()).collect();
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
            }),
            self.residual_probe_columns.clone(),
            self.residual_build_columns.clone(),
        )))
    }
}
