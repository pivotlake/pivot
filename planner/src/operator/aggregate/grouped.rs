//! The general grouped aggregate: `GROUP BY k1, k2 …` with one or more of
//! COUNT(*)/SUM/COUNT/MIN/MAX.
//!
//! DuckDB lowers grouped `AVG(c)` to `sum(c)`+`count(c)` with a downstream
//! divide, so the node here only ever holds count/sum/min/max slots.
//!
//! Lowering is pure monomorphisation dispatch over two independent axes — the
//! group **key** and the **value** — neither of which can be chosen at runtime
//! (the group operator stores its cells inline as `[Acc; N]` and takes the key
//! extractor as a type parameter), so the code is a tree of small `match`es each
//! selecting one concrete type:
//!
//! * **key** — a single integer/string column gets its dedicated extractor
//!   ([`IntKeyExtractor`]/[`StringKeyExtractor`]); two integer keys pack into
//!   [`IntPairKeyExtractor`]; anything else (3+ keys, mixed types) byte-encodes
//!   the tuple with [`RowKeyExtractor`]. *Computed* keys (`date_trunc(...)`,
//!   `ClientIP - 1`, `CASE …`) are first materialised into leading columns (see
//!   [`Aggregate::keying`]), so from the dispatch's view every key is a column.
//! * **value** — recognised signatures lower to a branch-free [`Compiled`]
//!   tuple; every other shape folds each slot by kind in `Dynamic<N>`, in `i128`
//!   when a slot needs the width (see [`Aggregate`]'s rule) else the narrow `i64`.

use super::{Aggregate, aggregation_slots, row_key_schema, sum_reads_wide_column};
use crate::compile::{Error, ExprEvalFn, ExprFn, ExprResult};
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, CountSlot, Dynamic, IntKeyExtractor,
    IntPairKeyExtractor, RecordBatchOperatorSpec, RowKeyExtractor, StringKeyExtractor, SumSlot,
};
use std::sync::Arc;

/// The output of [`Aggregate::keying`]: the (possibly re-projected) input, the
/// group keys as `(column, type)` pairs, and the value slots.
type Keyed = (
    RecordBatchOperatorSpec,
    Vec<(usize, Type)>,
    Vec<AggregationSlot>,
);

/// The output of [`Aggregate::materialize_group_keys`]: the (possibly
/// re-projected) input, the resolved `(column, type)` group keys, and the
/// shift `n` applied to the original columns (the count of materialised
/// computed keys).
type MaterializedGroupKeys = (RecordBatchOperatorSpec, Vec<(usize, Type)>, usize);

impl Aggregate {
    pub(super) fn compile_grouped(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Resolve the group keys to `(column, type)` pairs over a (possibly
        // re-projected) input. Any computed key has been materialised into a
        // leading column by `keying`, so from here every key is just a column.
        let (input, keys, slots) = self.keying(input)?;
        let key_cols: Vec<usize> = keys.iter().map(|(col, _)| *col).collect();

        let sig = signatures(&self.expressions);
        let output_limit = self.output_limit;

        // MEASUREMENT A/B toggle: with `PIVOT_GB_COMPILED` set, the q30/q31/q32
        // signature keeps its hand-written `Compiled` lowering (the baseline);
        // unset, it falls through to the deferred `Dynamic` path being measured.
        let use_compiled = std::env::var_os("PIVOT_GB_COMPILED").is_some();
        // MEASUREMENT baseline: route the deferred `Dynamic` to its eager (original
        // in-place) consume instead.
        let use_eager = std::env::var_os("PIVOT_GB_OLDDYNAMIC").is_some();

        // Cell width: i128 when a string extreme needs its 128-bit `ArenaKey`
        // cell, or when a SUM reads a 64-bit column; else the narrow i64 entry.
        let wide = slots.iter().any(|s| s.kind.is_string_extreme())
            || sum_reads_wide_column(&self.expressions);

        // Whether every slot folds additively (COUNT/SUM, no MIN/MAX or string
        // extreme): the `ONLY_ADDITIVE` `Dynamic` drops the per-slot kind dispatch
        // to a branch-free `+`, recovering the additive fast path (~7% on
        // low-cardinality grouped aggregates).
        let all_additive = slots.iter().all(|s| {
            matches!(
                s.kind,
                AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum
            )
        });

        // `group!` is the single lowering primitive: every arm below picks one
        // concrete key type `$K`, value type `$V`, and key config `$cfg`, then
        // calls it. The surrounding `input`/`key_cols`/`slots`/`output_limit` are
        // captured from this scope — exactly one arm ever runs, so each
        // moved-once value is consumed at most once.
        macro_rules! group {
            ($K:ty, $V:ty, $cfg:expr) => {
                Ok(input.group_by_aggregate_config::<$K, $V>(key_cols, slots, output_limit, $cfg))
            };
        }
        // Fold each slot by kind (or branch-free `+` when `$add`) in
        // `Dynamic<N, acc, ADDITIVE>`, dispatched on the slot count N (the inline
        // cell-array length). Numeric, string (`acc = i128`), and mixed alike,
        // since `Dynamic` dispatches per slot.
        macro_rules! arity {
            ($K:ty, $acc:ty, $add:literal, $eager:literal, $cfg:expr) => {
                match slots.len() {
                    1 => group!($K, Dynamic<1, $acc, $add, $eager>, $cfg),
                    2 => group!($K, Dynamic<2, $acc, $add, $eager>, $cfg),
                    3 => group!($K, Dynamic<3, $acc, $add, $eager>, $cfg),
                    4 => group!($K, Dynamic<4, $acc, $add, $eager>, $cfg),
                    5 => group!($K, Dynamic<5, $acc, $add, $eager>, $cfg),
                    6 => group!($K, Dynamic<6, $acc, $add, $eager>, $cfg),
                    n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            };
        }
        // The generic value fallback: pick the accumulator width and the additive
        // flag, then dispatch by arity. The `EAGER` const picks the consume strategy
        // (deferred default; the `PIVOT_GB_OLDDYNAMIC` MEASUREMENT baseline folds in
        // place) — chosen here so each build monomorphises one straight-line path.
        macro_rules! arity_add {
            ($K:ty, $acc:ty, $add:literal, $cfg:expr) => {
                if use_eager {
                    arity!($K, $acc, $add, true, $cfg)
                } else {
                    arity!($K, $acc, $add, false, $cfg)
                }
            };
        }
        macro_rules! dynamic {
            ($K:ty, $cfg:expr) => {
                match (wide, all_additive) {
                    (true, true) => arity_add!($K, i128, true, $cfg),
                    (true, false) => arity_add!($K, i128, false, $cfg),
                    (false, true) => arity_add!($K, i64, true, $cfg),
                    (false, false) => arity_add!($K, i64, false, $cfg),
                }
            };
        }
        // The value dispatch for a given key: a few signatures are worth a
        // hand-written, branch-free `Compiled` tuple; everything else folds
        // per-slot in `Dynamic`. Add a signature here to specialise it.
        macro_rules! value {
            ($K:ty, $cfg:expr) => {
                match sig.as_slice() {
                    [Sig::Count] => group!($K, Compiled<(CountSlot,)>, $cfg),
                    // MEASUREMENT A/B: the (Count, Sum16, Sum16, Count) `Compiled`
                    // arm for q30/q31/q32 is the baseline, kept only when
                    // `PIVOT_GB_COMPILED` is set; otherwise these route through the
                    // deferred `Dynamic` path being measured.
                    [
                        Sig::Count,
                        Sig::Sum(Type::Int16),
                        Sig::Sum(Type::Int16),
                        Sig::Count,
                    ] if use_compiled => group!(
                        $K,
                        Compiled<(CountSlot, SumSlot<Int16Type>, SumSlot<Int16Type>, CountSlot)>,
                        $cfg
                    ),
                    _ => dynamic!($K, $cfg),
                }
            };
        }

        // A single column keys on its native value directly: the dedicated
        // int/string extractor is cheaper than byte-encoding one column into the
        // row key and, unlike the row encoder, radix-partitions.
        if let [(_, ty)] = keys.as_slice() {
            match ty {
                Type::Int8 => return value!(IntKeyExtractor<Int8Type>, ()),
                Type::Int16 => return value!(IntKeyExtractor<Int16Type>, ()),
                Type::Int32 => return value!(IntKeyExtractor<Int32Type>, ()),
                Type::Int64 => return value!(IntKeyExtractor<Int64Type>, ()),
                Type::Utf8 => return value!(StringKeyExtractor, ()),
                // Other single-key types fall through to the row encoder below.
                _ => {}
            }
        }

        // Two integer keys pack into the specialised u128 pair extractor.
        if let Some(pair) = int_pair_keys(&keys) {
            return match pair {
                (Type::Int64, Type::Int32) => value!(IntPairKeyExtractor<Int64Type, Int32Type>, ()),
                (Type::Int32, Type::Int32) => value!(IntPairKeyExtractor<Int32Type, Int32Type>, ()),
                (Type::Int16, Type::Int32) => value!(IntPairKeyExtractor<Int16Type, Int32Type>, ()),
                (Type::Int16, Type::Int16) => value!(IntPairKeyExtractor<Int16Type, Int16Type>, ()),
                (Type::Int64, Type::Int64) => value!(IntPairKeyExtractor<Int64Type, Int64Type>, ()),
                (Type::Int32, Type::Int64) => value!(IntPairKeyExtractor<Int32Type, Int64Type>, ()),
                _ => unreachable!("int_pair_keys only returns the arms above"),
            };
        }

        // The general fallback: encode the whole key tuple into one byte blob.
        // Handles a single non-int/string key, 3+ keys, or mixed types.
        let Some(schema) = row_key_schema(keys.iter().map(|(_, t)| t)) else {
            return Err(Error::DataTypeNotSupportedForGroupBy(keys[0].1.clone()));
        };
        value!(RowKeyExtractor, schema)
    }

    /// Resolve the group keys into `(column, type)` pairs over a (possibly
    /// re-projected) input, materialising any *computed* key (one that is not a
    /// plain column) into a leading column cast to a canonical `Int64`/`Utf8View`
    /// so its type is known to the extractor. Returns the shift `n` (the number
    /// of computed keys) applied to the original columns, so the caller can
    /// offset its value/aggregate columns; `n` is 0 when every key is already a
    /// plain column. A computed key mixed with plain keys works the same way: the
    /// plain keys shift right past the materialised ones.
    ///
    /// Shared by the general grouped path and the two-level `COUNT(DISTINCT)`
    /// lowering, so a computed group key is supported uniformly by both.
    pub(super) fn materialize_group_keys(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<MaterializedGroupKeys, Error> {
        let computed: Vec<&Expression> = self
            .groups
            .iter()
            .filter(|g| !matches!(g, Expression::Ref(_)))
            .collect();

        if computed.is_empty() {
            let keys = self
                .groups
                .iter()
                .map(|g| match g {
                    Expression::Ref(r) => (r.column_idx, r.return_type.clone()),
                    _ => unreachable!("no computed keys"),
                })
                .collect();
            return Ok((input, keys, 0));
        }

        // Canonical type of each computed key: string keys stay strings, anything
        // else is read as i64 (the type its column is cast to below). The order
        // matches `computed`, i.e. the order computed keys appear in the GROUP BY.
        let computed_types: Vec<Type> = computed
            .iter()
            .map(|g| match g.result_type() {
                Some(Type::Utf8) => Type::Utf8,
                _ => Type::Int64,
            })
            .collect();
        let n = computed.len();
        let input = project_keys(input, &computed, &computed_types)?;

        // Computed keys occupy the leading columns `0..n` (in GROUP BY order);
        // plain keys keep their column shifted right by `n`.
        let mut next = 0;
        let keys = self
            .groups
            .iter()
            .map(|g| match g {
                Expression::Ref(r) => (r.column_idx + n, r.return_type.clone()),
                _ => {
                    let key = (next, computed_types[next].clone());
                    next += 1;
                    key
                }
            })
            .collect();
        Ok((input, keys, n))
    }

    /// [`materialize_group_keys`](Self::materialize_group_keys) plus the value
    /// slots, each aggregate column shifted right past any materialised keys.
    fn keying(&self, input: RecordBatchOperatorSpec) -> Result<Keyed, Error> {
        let (input, keys, n) = self.materialize_group_keys(input)?;
        let slots = aggregation_slots(&self.expressions)?
            .into_iter()
            .map(|s| AggregationSlot::new(s.kind, s.column + n))
            .collect();
        Ok((input, keys, slots))
    }
}

/// Evaluate each `key` per batch, cast it to its canonical `Int64`/`Utf8View`
/// type, and prepend them as leading columns `k0, k1, …`. The original columns
/// follow, shifted right by `keys.len()`, so the aggregates can still read their
/// value columns.
fn project_keys(
    input: RecordBatchOperatorSpec,
    keys: &[&Expression],
    types: &[Type],
) -> Result<RecordBatchOperatorSpec, Error> {
    let builders: Arc<Vec<ExprFn>> =
        Arc::new(keys.iter().map(|k| k.compile()).collect::<Result<_, _>>()?);
    let targets: Arc<Vec<DataType>> = Arc::new(
        types
            .iter()
            .map(|t| match t {
                Type::Utf8 => DataType::Utf8View,
                _ => DataType::Int64,
            })
            .collect(),
    );
    Ok(input.project(move || {
        let mut evals: Vec<ExprEvalFn> = builders.iter().map(|b| b()).collect();
        let targets = targets.clone();
        move |batch: RecordBatch| {
            let width = evals.len() + batch.num_columns();
            let mut fields: Vec<Field> = Vec::with_capacity(width);
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(width);
            for (i, eval) in evals.iter_mut().enumerate() {
                let arr: ArrayRef = match eval(&batch) {
                    ExprResult::Array(a) => a,
                    // A constant key: broadcast it to the batch length so it can
                    // sit beside the per-row columns.
                    ExprResult::Scalar(s) => {
                        let indices = UInt32Array::from(vec![0u32; batch.num_rows()]);
                        arrow::compute::take(&s.into_inner(), &indices, None).unwrap()
                    }
                };
                let casted = arrow::compute::cast(&arr, &targets[i]).unwrap();
                fields.push(Field::new(format!("k{i}"), targets[i].clone(), true));
                columns.push(casted);
            }
            for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
                fields.push(field.as_ref().clone());
                columns.push(col.clone());
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
        }
    }))
}

/// Per-slot signature used to recognise the `Compiled` specialisations in
/// `value!`. `CountStar` and `Count` collapse to one `Count` (both add +1 per
/// row); a `Sum` carries its column type so the specialisation can fix the read
/// width.
enum Sig {
    Count,
    Sum(Type),
    /// MIN/MAX (and any kind with no compiled specialisation). A distinct
    /// variant so these never match a compiled Count/Sum shape — doing so would
    /// monomorphise the wrong op and silently compute a count/sum instead of the
    /// extreme.
    Other,
}

fn signatures(exprs: &[Expression]) -> Vec<Sig> {
    exprs
        .iter()
        .map(|e| match e {
            Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                Sig::Sum(a.column.return_type.clone())
            }
            Expression::AggregateFunc(AggregateFunc::CountStar(_) | AggregateFunc::Count(_)) => {
                Sig::Count
            }
            _ => Sig::Other,
        })
        .collect()
}

/// The two key types when the group is exactly two integer columns we've
/// monomorphised the pair extractor for; `None` routes to the row fallback.
fn int_pair_keys(keys: &[(usize, Type)]) -> Option<(Type, Type)> {
    let [(_, a), (_, b)] = keys else { return None };
    let pair = (a.clone(), b.clone());
    matches!(
        pair,
        (Type::Int64, Type::Int32)
            | (Type::Int32, Type::Int32)
            | (Type::Int16, Type::Int32)
            | (Type::Int16, Type::Int16)
            | (Type::Int64, Type::Int64)
            | (Type::Int32, Type::Int64)
    )
    .then_some(pair)
}
