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
//!   [`Aggregate::materialize_group_keys`]), so from the dispatch's view every
//!   key is a column.
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
    AggregationKind, AggregationSlot, Compiled, CountSlot, Distinct, Dynamic, GroupLimit,
    IntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor, RecordBatchOperatorSpec,
    RowKeyExtractor, StringKeyExtractor, SumSlot,
};
use std::sync::Arc;

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
        // leading column, so from here every key is just a column and each
        // aggregate's value column shifts right past the `n` materialised keys.
        let (input, keys, n) = self.materialize_group_keys(input)?;
        build_group_by_operator(input, &keys, &self.expressions, n, self.output_limit)
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

        // Canonical type of each computed key (the type its column is cast to
        // below), derived from the key's static result type. The order matches
        // `computed`, i.e. the order computed keys appear in the GROUP BY.
        let computed_types: Vec<Type> = computed
            .iter()
            .map(|g| canonical_group_key_type(g.result_type()?))
            .collect::<Result<_, _>>()?;
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
}

/// Build a grouped-aggregate operator from the aggregate `exprs`. The semantic
/// layer over [`dispatch_group_by`]: resolve `exprs` to value slots via the shared
/// [`aggregation_slots`] (the single place aggregate semantics map to an
/// [`AggregationKind`]), shifting each value column right past the `key_shift`
/// materialised group keys; coalesce duplicate-valued aggregates and re-expand the
/// output around the [`dispatch_group_by`] call. The entry point for any level that
/// has aggregates: the general grouped path and each level of the two-level
/// `COUNT(DISTINCT)` lowering (whose inner/outer levels are *synthetic* expression
/// lists, which is why this takes `exprs` rather than reading `Aggregate`).
pub(super) fn build_group_by_operator(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    exprs: &[Expression],
    key_shift: usize,
    output_limit: Option<GroupLimit>,
) -> Result<RecordBatchOperatorSpec, Error> {
    // Coalesce aggregates that fold to the same value so each is scattered/merged
    // once and the hash entry stays narrow: any COUNT/COUNT(*) is identical (pivot's
    // Count is +1 per row), a SUM/MIN/MAX of the same column is identical.
    // `to_unique[i]` is the deduped slot expression `i` folds into; the value
    // columns are re-expanded below so the output still has one per expression.
    let mut unique: Vec<&Expression> = Vec::new();
    let mut fold_keys: Vec<FoldKey> = Vec::new();
    let mut to_unique: Vec<usize> = Vec::with_capacity(exprs.len());
    for e in exprs {
        let key = fold_key(e)?;
        let idx = match fold_keys.iter().position(|k| *k == key) {
            Some(i) => i,
            None => {
                fold_keys.push(key);
                unique.push(e);
                unique.len() - 1
            }
        };
        to_unique.push(idx);
    }
    let unique_exprs: Vec<Expression> = unique.iter().map(|e| (*e).clone()).collect();

    let slots: Vec<AggregationSlot> = aggregation_slots(&unique_exprs)?
        .into_iter()
        .map(|s| AggregationSlot::new(s.kind, s.column + key_shift))
        .collect();
    let sig = signatures(&unique_exprs);

    // A pushed-down Top-K sorts by a value slot identified by expression index;
    // coalescing renumbers the slots, so remap it to the unique slot it folds into
    // (identity when nothing coalesced). The sorted expression may itself be a
    // coalesced duplicate, which folds to the same value, so the order is unchanged.
    let output_limit = match output_limit {
        Some(GroupLimit::TopK { slot, limit }) => Some(GroupLimit::TopK {
            slot: to_unique[slot],
            limit,
        }),
        other => other,
    };

    // Cell width: i128 when a string extreme needs its 128-bit `ArenaKey` cell, or
    // when a SUM reads a 64-bit column; else the narrow i64 entry.
    let wide =
        slots.iter().any(|s| s.kind.is_string_extreme()) || sum_reads_wide_column(&unique_exprs);

    let grouped = dispatch_group_by(input, keys, slots, &sig, wide, output_limit)?;

    // No two aggregates coalesced: the output already has one value column per
    // expression, in order.
    if unique_exprs.len() == exprs.len() {
        return Ok(grouped);
    }
    Ok(reexpand_coalesced(grouped, keys.len(), to_unique))
}

/// Re-expand a coalesced group output back to one value column per original
/// expression: keep the `key_count` key columns, then select each expression's
/// coalesced value column via `to_unique`. A zero-copy column select on the
/// (low-cardinality) group output, renaming the value columns back to `v0..` so a
/// value mapped to several expressions stays a distinct field, matching the
/// non-coalesced output shape.
fn reexpand_coalesced(
    input: RecordBatchOperatorSpec,
    key_count: usize,
    to_unique: Vec<usize>,
) -> RecordBatchOperatorSpec {
    let cols: Arc<Vec<usize>> = Arc::new(
        (0..key_count)
            .chain(to_unique.iter().map(|&u| key_count + u))
            .collect(),
    );
    input.project(move || {
        let cols = cols.clone();
        move |batch: RecordBatch| {
            let out = batch.project(&cols).unwrap();
            let fields: Vec<Field> = out
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    if i < key_count {
                        f.as_ref().clone()
                    } else {
                        Field::new(format!("v{}", i - key_count), f.data_type().clone(), true)
                    }
                })
                .collect();
            RecordBatch::try_new(Arc::new(Schema::new(fields)), out.columns().to_vec()).unwrap()
        }
    })
}

/// Identifies aggregates that fold to the same value, so [`build_group_by_operator`] scatters
/// and merges each once: any COUNT/COUNT(*) (pivot's Count is +1 per row, so it
/// equals COUNT(*)), or a SUM/MIN/MAX of a given column.
#[derive(PartialEq)]
enum FoldKey {
    Count,
    Sum(usize),
    Min(usize),
    Max(usize),
}

fn fold_key(e: &Expression) -> Result<FoldKey, Error> {
    let Expression::AggregateFunc(func) = e else {
        return Err(Error::UnsupportedAggregateExpression(e.clone()));
    };
    Ok(match func {
        AggregateFunc::CountStar(_) | AggregateFunc::Count(_) => FoldKey::Count,
        AggregateFunc::Sum(a) => FoldKey::Sum(a.column.column_idx),
        AggregateFunc::Min(a) => FoldKey::Min(a.column.column_idx),
        AggregateFunc::Max(a) => FoldKey::Max(a.column.column_idx),
        _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
    })
}

/// The column type a computed group key is materialised as. String keys group
/// on their own value; integer and temporal keys group on the `Int64` they are
/// cast to (the bit pattern preserves distinctness). Any other result type is
/// rejected rather than silently coerced to an integer.
fn canonical_group_key_type(result_type: Type) -> Result<Type, Error> {
    match result_type {
        Type::Utf8 => Ok(Type::Utf8),
        Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Date | Type::Timestamp => {
            Ok(Type::Int64)
        }
        other => Err(Error::DataTypeNotSupportedForGroupBy(other)),
    }
}

/// Evaluate each `key` per batch, cast it to its canonical `Int64`/`Utf8View`
/// type, and prepend them as leading columns `k0, k1, …`. The original columns
/// follow, shifted right by `keys.len()`, so the aggregates can still read their
/// value columns. `types` are the canonical key types from
/// [`canonical_group_key_type`], so only `Utf8` and `Int64` reach here.
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
                Type::Int64 => DataType::Int64,
                other => {
                    unreachable!(
                        "canonical_group_key_type yields only Utf8 or Int64, got {other:?}"
                    )
                }
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
/// `select_value`. `CountStar` and `Count` collapse to one `Count` (both add +1 per
/// row); a `Sum` carries its column type so the specialisation can fix the read
/// width.
pub(super) enum Sig {
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
    // DATE/TIMESTAMP pack as their physical integer width (days/seconds since
    // epoch); the pair extractor casts the column the same way the row encoder
    // does, so they reach the fast u128 path beside plain integer keys.
    let pack_as_int = |t: &Type| match t {
        Type::Date => Type::Int32,
        Type::Timestamp => Type::Int64,
        other => other.clone(),
    };
    let pair = (pack_as_int(a), pack_as_int(b));
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

/// Select the concrete group-key extractor for `$keys` and hand it (plus its key
/// config) to the `$with_key!` macro, which finishes the build. This is one half
/// of the two-axis monomorphisation: a `macro_rules!` cannot *return* the chosen
/// type, so the key choice cannot be a value combined later, it must call a
/// continuation that has the rest of the build. [`dispatch_group_by`] passes
/// `select_value` (pick the value container, then `build_group_by!`);
/// [`build_dedup_operator`] passes `emit_dedup` (no value container, keys only).
///
/// A single integer/string column gets its dedicated extractor; two integer keys
/// pack into [`IntPairKeyExtractor`]; one integer plus one string key use
/// [`IntStrKeyExtractor`]; anything else byte-encodes the tuple with
/// [`RowKeyExtractor`]. Every arm `return`s except the row fallback, which is the
/// tail expression, so the surrounding function returns from inside the macro.
macro_rules! select_key_extractor {
    ($keys:expr, $with_key:ident) => {{
        // A single column keys on its native value directly: the dedicated
        // int/string extractor is cheaper than byte-encoding one column into the
        // row key and, unlike the row encoder, radix-partitions.
        if let [(_, ty)] = $keys {
            match ty {
                Type::Int8 => return $with_key!(IntKeyExtractor<Int8Type>, ()),
                Type::Int16 => return $with_key!(IntKeyExtractor<Int16Type>, ()),
                Type::Int32 => return $with_key!(IntKeyExtractor<Int32Type>, ()),
                Type::Int64 => return $with_key!(IntKeyExtractor<Int64Type>, ()),
                Type::Utf8 => return $with_key!(StringKeyExtractor, ()),
                // Other single-key types fall through to the row encoder below.
                _ => {}
            }
        }

        // Two integer keys pack into the specialised u128 pair extractor.
        if let Some(pair) = int_pair_keys($keys) {
            return match pair {
                (Type::Int64, Type::Int32) => $with_key!(IntPairKeyExtractor<Int64Type, Int32Type>, ()),
                (Type::Int32, Type::Int32) => $with_key!(IntPairKeyExtractor<Int32Type, Int32Type>, ()),
                (Type::Int16, Type::Int32) => $with_key!(IntPairKeyExtractor<Int16Type, Int32Type>, ()),
                (Type::Int16, Type::Int16) => $with_key!(IntPairKeyExtractor<Int16Type, Int16Type>, ()),
                (Type::Int64, Type::Int64) => $with_key!(IntPairKeyExtractor<Int64Type, Int64Type>, ()),
                (Type::Int32, Type::Int64) => $with_key!(IntPairKeyExtractor<Int32Type, Int64Type>, ()),
                _ => unreachable!("int_pair_keys only returns the arms above"),
            };
        }

        // One integer key plus one string key, in either order: the dedicated
        // int+string extractor keys on the native integer beside the string's arena
        // handle, skipping the row encoder's byte-encode of the tuple. `STR_FIRST`
        // follows the key order so the leading output column stays the first key.
        match $keys {
            [(_, int_ty), (_, Type::Utf8)] => match int_ty {
                Type::Int8 => return $with_key!(IntStrKeyExtractor<Int8Type, false>, ()),
                Type::Int16 => return $with_key!(IntStrKeyExtractor<Int16Type, false>, ()),
                Type::Int32 => return $with_key!(IntStrKeyExtractor<Int32Type, false>, ()),
                Type::Int64 => return $with_key!(IntStrKeyExtractor<Int64Type, false>, ()),
                _ => {}
            },
            [(_, Type::Utf8), (_, int_ty)] => match int_ty {
                Type::Int8 => return $with_key!(IntStrKeyExtractor<Int8Type, true>, ()),
                Type::Int16 => return $with_key!(IntStrKeyExtractor<Int16Type, true>, ()),
                Type::Int32 => return $with_key!(IntStrKeyExtractor<Int32Type, true>, ()),
                Type::Int64 => return $with_key!(IntStrKeyExtractor<Int64Type, true>, ()),
                _ => {}
            },
            // Anything else (two non-int/string, 3+ keys) falls through to the row
            // encoder below.
            _ => {}
        }

        // The general fallback: encode the whole key tuple into one byte blob.
        // Handles a single non-int/string key, 3+ keys, or mixed types.
        let Some(schema) = row_key_schema($keys.iter().map(|(_, t)| t)) else {
            return Err(Error::DataTypeNotSupportedForGroupBy($keys[0].1.clone()));
        };
        $with_key!(RowKeyExtractor, schema)
    }};
}

/// The monomorphisation core: pick the concrete key extractor for `keys` and value
/// container for `slots`, then build the GROUP BY operator. The mechanical layer
/// under [`build_group_by_operator`] (its only caller); the keys-only dedup path
/// ([`build_dedup_operator`]) shares the key cascade but supplies its own value.
///
/// `keys`/`slots` are already resolved to input column indices and kinds.
/// `sig` selects a hand-written, branch-free [`Compiled`] tuple for the few
/// signatures worth specialising; any shape it doesn't list folds per-slot in
/// [`Dynamic`]. `wide` requests the `i128` cell (a string extreme needs its
/// 128-bit `ArenaKey`, or a `SUM` can overflow `i64`); `output_limit` is the
/// per-partition LIMIT pushed into this level, or `None` to emit every group.
pub(super) fn dispatch_group_by(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    slots: Vec<AggregationSlot>,
    sig: &[Sig],
    wide: bool,
    output_limit: Option<GroupLimit>,
) -> Result<RecordBatchOperatorSpec, Error> {
    let key_cols: Vec<usize> = keys.iter().map(|(col, _)| *col).collect();

    // Whether every slot folds additively (COUNT/SUM, no MIN/MAX or string
    // extreme): the `ONLY_ADDITIVE` `Dynamic` drops the per-slot kind dispatch to
    // a branch-free `+`, recovering the additive fast path (~7% on low-cardinality
    // grouped aggregates).
    let all_additive = slots.iter().all(|s| {
        matches!(
            s.kind,
            AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum
        )
    });

    // The leaf combiner: both monomorphisation axes meet here, building the GROUP
    // BY operator for one concrete key type `$K`, value type `$V`, and key config
    // `$cfg`. The surrounding `input`/`key_cols`/`slots`/`output_limit` are captured
    // from this scope; exactly one arm ever runs, so each moved-once value is
    // consumed at most once.
    macro_rules! build_group_by {
        ($K:ty, $V:ty, $cfg:expr) => {
            Ok(input.group_by_aggregate::<$K, $V>(key_cols, slots, output_limit, $cfg))
        };
    }
    // Fold each slot by kind (or branch-free `+` when `$add`) in
    // `Dynamic<N, acc, ADDITIVE>`, dispatched on the slot count N (the inline
    // cell-array length). Numeric, string (`acc = i128`), and mixed alike, since
    // `Dynamic` dispatches per slot.
    macro_rules! arity {
        ($K:ty, $acc:ty, $add:literal, $cfg:expr) => {
            match slots.len() {
                1 => build_group_by!($K, Dynamic<1, $acc, $add>, $cfg),
                2 => build_group_by!($K, Dynamic<2, $acc, $add>, $cfg),
                3 => build_group_by!($K, Dynamic<3, $acc, $add>, $cfg),
                4 => build_group_by!($K, Dynamic<4, $acc, $add>, $cfg),
                5 => build_group_by!($K, Dynamic<5, $acc, $add>, $cfg),
                6 => build_group_by!($K, Dynamic<6, $acc, $add>, $cfg),
                n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
            }
        };
    }
    // The generic value fallback: pick the accumulator width and the additive
    // flag, then dispatch by arity.
    macro_rules! dynamic {
        ($K:ty, $cfg:expr) => {
            match (wide, all_additive) {
                (true, true) => arity!($K, i128, true, $cfg),
                (true, false) => arity!($K, i128, false, $cfg),
                (false, true) => arity!($K, i64, true, $cfg),
                (false, false) => arity!($K, i64, false, $cfg),
            }
        };
    }
    // The value-container selection (the continuation `select_key_extractor!`
    // calls once it has picked a key): a few signatures are worth a hand-written,
    // branch-free `Compiled` tuple; everything else folds per-slot in `Dynamic`.
    // Add a signature here to specialise it. Ends at the `build_group_by!` leaf.
    macro_rules! select_value {
        ($K:ty, $cfg:expr) => {
            match sig {
                [Sig::Count] => build_group_by!($K, Compiled<(CountSlot,)>, $cfg),
                // `COUNT(*), SUM(i16), SUM(i16)` (e.g. an `AVG(i16)` whose count has
                // coalesced into the `COUNT(*)`): a branch-free, narrow entry.
                [Sig::Count, Sig::Sum(Type::Int16), Sig::Sum(Type::Int16)] => build_group_by!(
                    $K,
                    Compiled<(CountSlot, SumSlot<Int16Type>, SumSlot<Int16Type>)>,
                    $cfg
                ),
                _ => dynamic!($K, $cfg),
            }
        };
    }

    // Select the key extractor (the cascade shared with the keys-only dedup path);
    // `select_value` then picks the value container for it and reaches the
    // `build_group_by!` leaf. The two selections nest because a macro can't return a
    // chosen type to combine later.
    select_key_extractor!(keys, select_value)
}

/// Lower a keys-only GROUP BY that dedups the `keys` tuple and emits each distinct
/// key (no aggregates). Shares the [`select_key_extractor!`] extractor cascade with
/// [`dispatch_group_by`], so it supports every group-key shape the aggregating path
/// does. Used as the inner level of the two-level `COUNT(DISTINCT)` lowerings,
/// which then count the deduped rows per group.
pub(super) fn build_dedup_operator(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
) -> Result<RecordBatchOperatorSpec, Error> {
    let key_cols: Vec<usize> = keys.iter().map(|(col, _)| *col).collect();
    // Keys-only: `Distinct` holds no accumulator, so the slot list is empty and the
    // group emits the key columns themselves.
    macro_rules! emit_dedup {
        ($K:ty, $cfg:expr) => {
            Ok(input.group_by_aggregate::<$K, Distinct>(key_cols, Vec::new(), None, $cfg))
        };
    }
    select_key_extractor!(keys, emit_dedup)
}

/// Remap resolved group `keys` onto the leading output columns `0..n` of a
/// preceding group-by (which emits its key columns in order), keeping each key's
/// type. Shared by the two-level `COUNT(DISTINCT)` lowerings to group the deduped
/// inner output by its leading group columns.
pub(super) fn derive_outer_keys(groups: &[(usize, Type)]) -> Vec<(usize, Type)> {
    groups
        .iter()
        .enumerate()
        .map(|(col, (_, ty))| (col, ty.clone()))
        .collect()
}
