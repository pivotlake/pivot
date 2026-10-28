//! Aggregates remembered per row group.
//!
//! An aggregate over one table's scan is the merge of the same aggregate
//! taken over each of the table's row groups. Those per-row-group partials
//! depend only on the table's data and the query's shape, so once a query
//! has computed them they can serve every later run of that shape: the scan,
//! the decode and the first level of grouping are skipped, and the query
//! merges the remembered partials instead.
//!
//! A shape is remembered from its second run on. The first run only notes
//! having met it, so a query run once pays nothing. The second run groups
//! by the row group as an extra key, copies each row's partial as the
//! query finishes, and merges them for its own answer. That run also times
//! its two halves: the time to the first partial is about what the shape
//! costs to aggregate afresh, and the rest is what merging the partials
//! costs. The partials are kept only when the merge is clearly the cheaper
//! of the two; otherwise the shape is remembered as dense and left alone
//! from then on, as it is when its partials hold too many rows to copy at
//! all (a group per row, as with a unique key). A remembered shape whose
//! merge later takes as long as aggregating afresh is forgotten the same
//! way.
//!
//! Only decomposable aggregates qualify: `COUNT`, `SUM`, `MIN` and `MAX`,
//! whose merge is a sum or an extreme of the partials, and one
//! `COUNT(DISTINCT x)`, whose partials are each row group's distinct `x` per
//! group and whose merge counts them distinctly again.
//! The scan below must be a plain one (no join-produced filters), under at
//! most a filter and a projection reading columns deterministically, so a
//! partial means the same thing on every run. The table revision is part of
//! the key: a table that changed is read again.
//! `PIVOT_PARTIAL_AGGREGATE_CACHE_BYTES` bounds the cache (`0` turns it off).

use crate::Operator;
use crate::compile::{Error, RuntimeFilterSlots};
use crate::expression::{AggregateFunc, Cast, CountStar, Expression, NumericAggregate, Ref};
use crate::operator::{Aggregate, Filter, Input, Projection};
use crate::plan::PlanNode;
use crate::types::{Type, physical_arrow_type};
use arrow::array::MutableArrayData;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, RunArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{
    DataFlowDispatcher, OperatorIO, RecordBatchOperatorSpec, Sender, Unary, UnaryFactory,
    values_input,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const DEFAULT_BUDGET_BYTES: usize = 2 << 30;
const BUDGET_VARIABLE: &str = "PIVOT_PARTIAL_AGGREGATE_CACHE_BYTES";
/// Partials are copied only while they hold fewer rows than this share of
/// the scan's rows, one in four: a merge over more cannot pay off.
const DENSE_DIVISOR: u64 = 4;
/// Partials are kept when their merge took under this share, in percent, of
/// the time the shape took to aggregate afresh.
const MERGE_SHARE_TO_KEEP: u128 = 70;
/// Remembered partials are forgotten once a merge of them takes at least
/// this share, in percent, of the time the shape took to aggregate afresh.
pub(crate) const MERGE_SHARE_TO_FORGET: u128 = 100;
const ROW_GROUP_KEY_NAME: &str = "row_group";

/// One aggregate shape over one table revision.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartialsKey {
    pub(crate) table: String,
    pub(crate) version: String,
    /// The plan from the remembered node down, rendered.
    pub(crate) shape: String,
}

/// What the cache remembers of a shape.
#[derive(Clone)]
pub enum RememberedPartials {
    /// Met once, and not yet worth gathering.
    Met,
    /// Merging its partials does not pay off.
    Dense,
    Partials(Arc<Partials>),
}

/// A shape's remembered partials.
pub struct Partials {
    /// The per-row-group partials: group keys, the row group, then one
    /// column per aggregate.
    pub batches: Vec<RecordBatch>,
    /// How long the shape took to aggregate afresh: the time a merge of the
    /// partials has to beat.
    pub fresh: Duration,
}

impl RememberedPartials {
    fn bytes(&self) -> usize {
        match self {
            RememberedPartials::Partials(partials) => partials
                .batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum(),
            RememberedPartials::Met | RememberedPartials::Dense => size_of::<Self>(),
        }
    }
}

/// Whether a merge taking `merge` is worth keeping against aggregating afresh
/// in `fresh`: it must take under `share_percent` of that.
pub(crate) fn merge_pays_off(merge: Duration, fresh: Duration, share_percent: u128) -> bool {
    merge.as_nanos() * 100 < fresh.as_nanos() * share_percent
}

struct Entries {
    by_key: HashMap<PartialsKey, RememberedPartials>,
    insertion_order: VecDeque<PartialsKey>,
    bytes: usize,
}

/// The process-wide store of remembered partials.
pub struct PartialAggregateCache {
    entries: Mutex<Entries>,
    budget_bytes: usize,
}

impl PartialAggregateCache {
    pub fn global() -> &'static PartialAggregateCache {
        static CACHE: OnceLock<PartialAggregateCache> = OnceLock::new();
        CACHE.get_or_init(|| {
            PartialAggregateCache::with_budget(dispatch::env::get_env_var_with_default(
                BUDGET_VARIABLE,
                DEFAULT_BUDGET_BYTES,
            ))
        })
    }

    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            entries: Mutex::new(Entries {
                by_key: HashMap::new(),
                insertion_order: VecDeque::new(),
                bytes: 0,
            }),
            budget_bytes,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.budget_bytes > 0
    }

    pub fn get(&self, key: &PartialsKey) -> Option<RememberedPartials> {
        self.entries.lock().unwrap().by_key.get(key).cloned()
    }

    /// Remembers `remembered` for `key`, letting the oldest entries go while
    /// the cache is over its budget.
    pub fn insert(&self, key: PartialsKey, remembered: RememberedPartials) {
        let mut entries = self.entries.lock().unwrap();
        let bytes = remembered.bytes();
        if let Some(previous) = entries.by_key.insert(key.clone(), remembered) {
            entries.bytes -= previous.bytes();
        } else {
            entries.insertion_order.push_back(key);
        }
        entries.bytes += bytes;
        while entries.bytes > self.budget_bytes {
            let oldest = entries
                .insertion_order
                .pop_front()
                .expect("a cache over budget holds an entry");
            if let Some(evicted) = entries.by_key.remove(&oldest) {
                entries.bytes -= evicted.bytes();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compiles an aggregate over a plain scan from remembered partials, or,
/// the second time the shape is met, as a plan that gathers them; `None`
/// leaves the node to the ordinary compile.
pub(crate) fn try_compile_memoized(
    node: &PlanNode,
    dispatcher: &DataFlowDispatcher,
    slots: &mut RuntimeFilterSlots,
) -> Result<Option<RecordBatchOperatorSpec>, Error> {
    let cache = PartialAggregateCache::global();
    if !cache.is_enabled() {
        return Ok(None);
    }
    let Operator::Aggregate(aggregate) = &node.operator else {
        return Ok(None);
    };
    let [child] = node.inputs.as_slice() else {
        return Ok(None);
    };
    if !is_decomposable(aggregate) {
        return Ok(None);
    }
    let Some(shape) = ScanShape::recognize(child, None) else {
        return Ok(None);
    };
    if shape.input.emit_row_group_metadata {
        return Ok(None);
    }
    let revision = shape.input.table.table_revision();
    let key = PartialsKey {
        table: revision.identity,
        version: revision.version,
        shape: node.to_string(),
    };
    match cache.get(&key) {
        None => {
            cache.insert(key, RememberedPartials::Met);
            Ok(None)
        }
        Some(RememberedPartials::Dense) => Ok(None),
        Some(RememberedPartials::Partials(partials)) => {
            let started = Instant::now();
            let source = values_input(dispatcher, partials.batches.iter().cloned())
                .map_each(|batch: RecordBatch| batch)
                .record_batches();
            let rewrite = Rewrite::plan(aggregate, child)?;
            let merged = rewrite.compile_merge(source)?;
            let fresh = partials.fresh;
            let forget_a_slow_merge = move || {
                if !merge_pays_off(started.elapsed(), fresh, MERGE_SHARE_TO_FORGET) {
                    cache.insert(key.clone(), RememberedPartials::Dense);
                }
            };
            Ok(Some(merged.with_unary(MergeWatch::over_workers(
                dispatcher.worker_count(),
                forget_a_slow_merge,
            ))))
        }
        Some(RememberedPartials::Met) => {
            let gathering = Arc::new(Gathering::start(
                key,
                shape.input.table.estimate_row_count(),
                None,
            ));
            let rewrite = Rewrite::plan(aggregate, child)?;
            let partials = rewrite.compile_gathering(dispatcher, slots, &shape)?;
            let tapped =
                partials.with_unary((0..dispatcher.worker_count()).map(|_| PartialsTapFactory {
                    gathering: gathering.clone(),
                }));
            let merged = rewrite.compile_merge(tapped)?;
            let file_the_partials = move || gathering.file();
            Ok(Some(merged.with_unary(MergeWatch::over_workers(
                dispatcher.worker_count(),
                file_the_partials,
            ))))
        }
    }
}

/// Whether every aggregate merges from per-row-group partials, and every
/// group key reads deterministically. A count, sum or extreme merges by a
/// sum or an extreme over an argument the gathering can read; one
/// `COUNT(DISTINCT x)` merges by counting `x` distinctly again over the rows
/// each row group deduplicated, so `x` must be groupable.
fn is_decomposable(aggregate: &Aggregate) -> bool {
    let groups_deterministic = aggregate
        .groups
        .iter()
        .all(Expression::is_deterministic_over_columns);
    let distinct_count = aggregate
        .expressions
        .iter()
        .filter(|expression| {
            matches!(
                expression,
                Expression::AggregateFunc(AggregateFunc::CountDistinct(_))
            )
        })
        .count();
    let aggregates_decomposable = aggregate.expressions.iter().all(|expression| {
        let Expression::AggregateFunc(function) = expression else {
            return false;
        };
        match function {
            AggregateFunc::CountStar(_) => true,
            AggregateFunc::Count(a)
            | AggregateFunc::Sum(a)
            | AggregateFunc::Min(a)
            | AggregateFunc::Max(a) => {
                a.argument.is_deterministic_over_columns()
                    && a.argument
                        .result_type()
                        .is_ok_and(|argument| is_gatherable_argument(&argument))
            }
            AggregateFunc::CountDistinct(a) => {
                a.argument.is_deterministic_over_columns()
                    && a.argument
                        .result_type()
                        .is_ok_and(|argument| is_groupable_distinct_argument(&argument))
            }
            AggregateFunc::First(_) | AggregateFunc::Avg(_) => false,
        }
    });
    groups_deterministic
        && aggregates_decomposable
        && distinct_count <= 1
        && !aggregate.expressions.is_empty()
}

/// Whether `argument`, the column a `COUNT(DISTINCT)` counts, can be a group
/// key of the gathering aggregate.
fn is_groupable_distinct_argument(argument: &Type) -> bool {
    matches!(
        argument,
        Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::UInt8
            | Type::UInt16
            | Type::UInt32
            | Type::UInt64
            | Type::Decimal { .. }
            | Type::Utf8
            | Type::Date
            | Type::Timestamp
            | Type::TimestampTz
    )
}

/// The plan under a memoizable node: a plain scan, under at most a filter
/// and then projections, each reading columns deterministically.
pub(crate) struct ScanShape<'a> {
    pub(crate) input: &'a Input,
    /// The scan's output nullability, one per emitted column.
    pub(crate) scan_nullability: Vec<bool>,
    pub(crate) filter: Option<&'a Filter>,
    /// Innermost first.
    pub(crate) projections: Vec<&'a Projection>,
}

impl<'a> ScanShape<'a> {
    /// The shape under `child`, when it is one. The scan may consume dynamic
    /// filters only from the slot `dynamic_filter_slot` names, the one the
    /// node above publishes, which a run over remembered rows has no use for.
    /// The table has to tag rows with their row group when asked, as a
    /// gathering run asks.
    pub(crate) fn recognize(
        child: &'a PlanNode,
        dynamic_filter_slot: Option<usize>,
    ) -> Option<Self> {
        let mut projections = Vec::new();
        let mut node = child;
        while let Operator::Projection(projection) = &node.operator {
            if !projection
                .projections
                .iter()
                .all(Expression::is_deterministic_over_columns)
            {
                return None;
            }
            projections.push(projection);
            node = node.inputs.first()?;
        }
        projections.reverse();
        let (filter, scan_node) = match &node.operator {
            Operator::Filter(filter) => (Some(filter), node.inputs.first()?),
            _ => (None, node),
        };
        if let Some(filter) = filter
            && !filter
                .conditions
                .iter()
                .all(Expression::is_deterministic_over_columns)
        {
            return None;
        }
        let Operator::Input(input) = &scan_node.operator else {
            return None;
        };
        let dynamic_filters_allowed = input
            .dynamic_filters
            .iter()
            .all(|filter| Some(filter.slot_id) == dynamic_filter_slot);
        if !scan_node.inputs.is_empty()
            || !dynamic_filters_allowed
            || !input.key_bitset_filters.is_empty()
            || input.join_filter_info.is_some()
            || !input.table.supports_late_materialization()
        {
            return None;
        }
        Some(Self {
            input,
            scan_nullability: scan_node.output_nullability(),
            filter,
            projections,
        })
    }

    /// The scan compiled on its own, tagging each row with its row group and
    /// consuming no dynamic filter.
    pub(crate) fn compile_tagged_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        slots: &mut RuntimeFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let input = self.input;
        let scan = Input {
            table: input.table.clone_box(),
            columns: input.columns.clone(),
            dynamic_filters: Vec::new(),
            key_bitset_filters: Vec::new(),
            duckdb_table_binding_index: input.duckdb_table_binding_index,
            join_filter_info: None,
            emit_row_group_metadata: true,
        };
        scan.compile(dispatcher, slots)
    }
}

/// The two aggregates an aggregate shape becomes: the gathering one, grouped
/// by the row group as well, and the merging one over its partials.
struct Rewrite {
    gathering: Aggregate,
    /// Nullability of the gathering aggregate's input columns.
    gathering_input_nullability: Vec<bool>,
    merging: Aggregate,
    /// Nullability of the partials' columns, the merging aggregate's input.
    partials_nullability: Vec<bool>,
    /// The merge's output columns to cast back to the type the query
    /// declared: a count summed as a wide integer, an extreme taken over a
    /// widened argument.
    restore: Vec<(usize, DataType)>,
}

impl Rewrite {
    fn plan(aggregate: &Aggregate, child: &PlanNode) -> Result<Self, Error> {
        // The gathering aggregate reads what the original did, plus the row
        // group appended as the last column.
        let child_nullability = child.output_nullability();
        let row_group_position = child_nullability.len();
        let mut gathering_input_nullability = child_nullability;
        gathering_input_nullability.push(false);
        let row_group_key = Expression::Ref(Ref {
            column_idx: row_group_position,
            return_type: Type::Int32,
            name: Some(ROW_GROUP_KEY_NAME.to_string()),
        });
        let key_count = aggregate.groups.len();
        let mut gathering_groups = aggregate.groups.clone();
        gathering_groups.push(row_group_key);
        // A `COUNT(DISTINCT x)` makes `x` a key of the gathering, so each row
        // group's partial holds its distinct `x` per group, and the merge
        // counts `x` distinctly over them, right after the row group column.
        let distinct_column = key_count + 1;
        for expression in &aggregate.expressions {
            if let Expression::AggregateFunc(AggregateFunc::CountDistinct(a)) = expression {
                gathering_groups.push((*a.argument).clone());
            }
        }
        let mut gathering_expressions = Vec::with_capacity(aggregate.expressions.len());
        let mut merging_expressions = Vec::with_capacity(aggregate.expressions.len());
        let mut restore = Vec::new();
        let mut partial_column = gathering_groups.len();
        for (position, expression) in aggregate.expressions.iter().enumerate() {
            let Expression::AggregateFunc(function) = expression else {
                return Err(Error::UnsupportedAggregateExpression(expression.clone()));
            };
            if let AggregateFunc::CountDistinct(a) = function {
                merging_expressions.push(Expression::AggregateFunc(AggregateFunc::CountDistinct(
                    partial_of(
                        distinct_column,
                        a.argument.result_type()?,
                        a.return_type.clone(),
                    ),
                )));
                continue;
            }
            let pair = AggregatePair::of(function, partial_column)?;
            partial_column += 1;
            gathering_expressions.push(Expression::AggregateFunc(pair.gathering));
            merging_expressions.push(Expression::AggregateFunc(pair.merging));
            if let Some(declared) = pair.restore {
                restore.push((key_count + position, declared));
            }
        }
        if gathering_expressions.is_empty() {
            // A shape counting distinct values alone: its partials are the
            // key rows themselves, and the gathering just has to emit them.
            gathering_expressions.push(Expression::AggregateFunc(AggregateFunc::CountStar(
                CountStar {
                    params: Vec::new(),
                    return_type: Type::Int64,
                },
            )));
        }
        let gathering = Aggregate {
            groups: gathering_groups,
            expressions: gathering_expressions,
            output_limit: None,
        };
        let partials_nullability: Vec<bool> = gathering
            .groups
            .iter()
            .chain(gathering.expressions.iter())
            .map(|expression| expression.nullability(&gathering_input_nullability))
            .collect();

        // The merging aggregate groups by the partials' key columns and folds
        // each aggregate's partial column.
        let merging_groups: Vec<Expression> = aggregate
            .groups
            .iter()
            .enumerate()
            .map(|(position, group)| {
                Ok(Expression::Ref(Ref {
                    column_idx: position,
                    return_type: group.result_type()?,
                    name: None,
                }))
            })
            .collect::<Result<_, Error>>()?;
        let merging = Aggregate {
            groups: merging_groups,
            expressions: merging_expressions,
            output_limit: aggregate.output_limit.clone(),
        };
        Ok(Self {
            gathering,
            gathering_input_nullability,
            merging,
            partials_nullability,
            restore,
        })
    }

    /// The plan that reads the table and aggregates it per row group: the
    /// scan carrying its row group, the shape's filter and projection with the
    /// row group passed along, then the gathering aggregate.
    fn compile_gathering(
        &self,
        dispatcher: &DataFlowDispatcher,
        slots: &mut RuntimeFilterSlots,
        shape: &ScanShape<'_>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let scan_width = shape.scan_nullability.len();
        let mut spec = shape
            .compile_tagged_scan(dispatcher, slots)?
            .project(move || move |batch: RecordBatch| with_plain_row_group(batch, scan_width));
        if let Some(filter) = shape.filter {
            spec = filter.compile(spec)?;
        }
        let mut row_group_position = scan_width;
        for projection in &shape.projections {
            let mut projections = projection.projections.clone();
            projections.push(Expression::Ref(Ref {
                column_idx: row_group_position,
                return_type: Type::Int32,
                name: Some(ROW_GROUP_KEY_NAME.to_string()),
            }));
            row_group_position = projections.len() - 1;
            spec = Projection { projections }.compile(spec)?;
        }
        self.gathering
            .compile(spec, self.gathering_input_nullability.clone())
    }

    /// The merge of `partials` into the aggregate's answer, with every column
    /// back at the type the query declared.
    fn compile_merge(
        &self,
        partials: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let merged = self
            .merging
            .compile(partials, self.partials_nullability.clone())?;
        if self.restore.is_empty() {
            return Ok(merged);
        }
        let restore = Arc::new(self.restore.clone());
        Ok(merged.project(move || {
            let restore = restore.clone();
            move |batch: RecordBatch| cast_columns(batch, &restore)
        }))
    }
}

/// How one aggregate is taken per row group and merged back.
struct AggregatePair {
    gathering: AggregateFunc,
    merging: AggregateFunc,
    /// The declared type to cast the merged column back to, when the merge
    /// emits another.
    restore: Option<DataType>,
}

impl AggregatePair {
    /// The pair for `function`, whose partial the merge reads at
    /// `partial_column`. A count is summed as a wide integer and cast back to
    /// `Int64`; a sum is summed again at its declared type; an extreme is
    /// taken again over the partial extremes. An unsigned argument narrower
    /// than 64 bits is widened to `Int64` for the gathering, which groups by
    /// the row group and so reads it through the grouped path's readers;
    /// an extreme over a widened argument is cast back to the declared type.
    fn of(function: &AggregateFunc, partial_column: usize) -> Result<Self, Error> {
        let count_pair = |gathering: AggregateFunc| Self {
            gathering,
            merging: AggregateFunc::Sum(partial_of(partial_column, Type::Int64, Type::Int128)),
            restore: Some(DataType::Int64),
        };
        Ok(match function {
            AggregateFunc::CountStar(count) => count_pair(AggregateFunc::CountStar(count.clone())),
            AggregateFunc::Count(a) => count_pair(AggregateFunc::Count(NumericAggregate {
                argument: Box::new(widened(&a.argument)?.0),
                return_type: a.return_type.clone(),
            })),
            AggregateFunc::Sum(a) => Self {
                gathering: AggregateFunc::Sum(NumericAggregate {
                    argument: Box::new(widened(&a.argument)?.0),
                    return_type: a.return_type.clone(),
                }),
                merging: AggregateFunc::Sum(partial_of(
                    partial_column,
                    a.return_type.clone(),
                    a.return_type.clone(),
                )),
                restore: None,
            },
            AggregateFunc::Min(a) | AggregateFunc::Max(a) => {
                let (argument, was_widened) = widened(&a.argument)?;
                let partial_type = if was_widened {
                    Type::Int64
                } else {
                    a.return_type.clone()
                };
                let gathering = NumericAggregate {
                    argument: Box::new(argument),
                    return_type: partial_type.clone(),
                };
                let merging = partial_of(partial_column, partial_type.clone(), partial_type);
                let (gathering, merging) = if matches!(function, AggregateFunc::Min(_)) {
                    (AggregateFunc::Min(gathering), AggregateFunc::Min(merging))
                } else {
                    (AggregateFunc::Max(gathering), AggregateFunc::Max(merging))
                };
                Self {
                    gathering,
                    merging,
                    restore: was_widened.then(|| physical_arrow_type(&a.return_type)),
                }
            }
            AggregateFunc::First(a) => {
                return Err(Error::UnsupportedAggregateExpression(
                    Expression::AggregateFunc(AggregateFunc::First(a.clone())),
                ));
            }
            AggregateFunc::Avg(a) | AggregateFunc::CountDistinct(a) => {
                return Err(Error::UnsupportedAggregateExpression(
                    Expression::AggregateFunc(AggregateFunc::Avg(a.clone())),
                ));
            }
        })
    }
}

/// `argument` as the gathering aggregate reads it: cast to `Int64` when it
/// is an unsigned integer narrower than 64 bits, as itself otherwise. The
/// flag says whether it was cast.
fn widened(argument: &Expression) -> Result<(Expression, bool), Error> {
    match argument.result_type()? {
        Type::UInt8 | Type::UInt16 | Type::UInt32 => Ok((
            Expression::Cast(Cast {
                target: Type::Int64,
                target_arrow: DataType::Int64,
                source: Box::new(argument.clone()),
            }),
            true,
        )),
        _ => Ok((argument.clone(), false)),
    }
}

/// Whether an aggregate's argument of type `argument` can be gathered per row
/// group: what the grouped path reads, or an unsigned integer it is widened
/// for.
fn is_gatherable_argument(argument: &Type) -> bool {
    matches!(
        argument,
        Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::UInt8
            | Type::UInt16
            | Type::UInt32
            | Type::Float32
            | Type::Float64
            | Type::Decimal { .. }
            | Type::Utf8
            | Type::Date
            | Type::Timestamp
            | Type::TimestampTz
    )
}

fn partial_of(column: usize, column_type: Type, return_type: Type) -> NumericAggregate {
    NumericAggregate {
        argument: Box::new(Expression::Ref(Ref {
            column_idx: column,
            return_type: column_type,
            name: None,
        })),
        return_type,
    }
}

/// `batch` as the scan emitted it with its row-group metadata, keeping the
/// first `width` columns and, after them, the row group of every row as a
/// plain `Int32` column; the run-encoded row group and the row index go.
fn with_plain_row_group(batch: RecordBatch, width: usize) -> RecordBatch {
    let rows = batch.num_rows();
    let encoded = batch
        .column(width)
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .expect("a scan emitting row-group metadata run-encodes the row group");
    let groups = encoded
        .values()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("row groups are UInt32");
    let mut plain: Vec<i32> = Vec::with_capacity(rows);
    let mut run_start = 0usize;
    for (run, &run_end) in encoded.run_ends().values().iter().enumerate() {
        let run_end = run_end as usize;
        plain.extend(std::iter::repeat_n(
            groups.value(run) as i32,
            run_end - run_start,
        ));
        run_start = run_end;
    }
    let (schema, mut columns, _) = batch.into_parts();
    columns.truncate(width);
    columns.push(Arc::new(Int32Array::from(plain)) as ArrayRef);
    let mut fields: Vec<_> = schema.fields().iter().take(width).cloned().collect();
    fields.push(Arc::new(Field::new(
        ROW_GROUP_KEY_NAME,
        DataType::Int32,
        false,
    )));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .expect("the row group column is as long as the batch")
}

/// `batch` with each listed column cast to the type beside it: a merged
/// value back at the type the query declared.
fn cast_columns(batch: RecordBatch, targets: &[(usize, DataType)]) -> RecordBatch {
    let (schema, mut arrays, _) = batch.into_parts();
    let mut fields: Vec<_> = schema.fields().iter().cloned().collect();
    for (column, target) in targets {
        arrays[*column] = arrow::compute::cast(&arrays[*column], target)
            .expect("a merged value fits the type its query declared");
        fields[*column] = Arc::new(Field::new(
            fields[*column].name(),
            target.clone(),
            fields[*column].is_nullable(),
        ));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .expect("casting a column keeps the batch's shape")
}

/// A batch copied into memory of its own, so the cache holds nothing that
/// belongs to a query's buffers. A view column's values are gathered into
/// fresh buffers: copying its views alone would keep the page buffers they
/// point into alive, and those belong to the workers' memory.
pub(crate) fn copied(batch: &RecordBatch) -> RecordBatch {
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|column| match column.data_type() {
            DataType::Utf8View => Arc::new(column.as_string_view().gc()) as ArrayRef,
            DataType::BinaryView => Arc::new(column.as_binary_view().gc()) as ArrayRef,
            _ => {
                let data = column.to_data();
                let mut copy = MutableArrayData::new(vec![&data], false, data.len());
                copy.extend(0, 0, data.len());
                arrow_array::make_array(copy.freeze())
            }
        })
        .collect();
    RecordBatch::try_new(batch.schema(), columns).expect("a copy keeps the batch's shape")
}

/// A pass over gathered partials before they are filed, boiling down what
/// the workers kept separately.
pub(crate) type Condense = Box<dyn Fn(Vec<RecordBatch>) -> Vec<RecordBatch> + Send + Sync>;

/// One shape's gathering run while it is under way: its partials as they
/// are copied, and the moments that tell whether merging them pays off.
pub(crate) struct Gathering {
    key: PartialsKey,
    started: Instant,
    /// Nanoseconds from the start to the first partial emitted, which is
    /// about how long the shape takes to compute afresh; `u64::MAX` until
    /// one is.
    first_partial_nanos: AtomicU64,
    /// Rows the partials may hold before they are too many to keep, when the
    /// table can tell its row count.
    row_limit: Option<u64>,
    rows: AtomicU64,
    batches: Mutex<Vec<RecordBatch>>,
    /// Set once the partials grew past the row limit and were let go.
    abandoned: AtomicBool,
    condense: Option<Condense>,
}

impl Gathering {
    pub(crate) fn start(
        key: PartialsKey,
        scan_rows: Option<u64>,
        condense: Option<Condense>,
    ) -> Self {
        Self {
            key,
            started: Instant::now(),
            first_partial_nanos: AtomicU64::new(u64::MAX),
            row_limit: scan_rows.map(|rows| rows / DENSE_DIVISOR),
            rows: AtomicU64::new(0),
            batches: Mutex::new(Vec::new()),
            abandoned: AtomicBool::new(false),
            condense,
        }
    }

    /// Takes note of a batch of partials: the moment the first arrived, and
    /// a copy of it while the partials are few enough to keep.
    fn note(&self, batch: &RecordBatch) {
        self.first_partial_nanos
            .fetch_min(self.started.elapsed().as_nanos() as u64, Ordering::AcqRel);
        if self.abandoned.load(Ordering::Acquire) {
            return;
        }
        let rows = batch.num_rows() as u64;
        let rows_so_far = self.rows.fetch_add(rows, Ordering::AcqRel) + rows;
        if self.row_limit.is_some_and(|limit| rows_so_far > limit) {
            self.abandoned.store(true, Ordering::Release);
            self.batches.lock().unwrap().clear();
        } else {
            self.batches.lock().unwrap().push(copied(batch));
        }
    }

    /// Files what the run found with the cache, once its merge is done.
    pub(crate) fn file(&self) {
        let elapsed = self.started.elapsed();
        let first_partial = self.first_partial_nanos.load(Ordering::Acquire);
        let fresh = if first_partial == u64::MAX {
            elapsed
        } else {
            Duration::from_nanos(first_partial)
        };
        let merge = elapsed.saturating_sub(fresh);
        let batches = std::mem::take(&mut *self.batches.lock().unwrap());
        let kept = (!self.abandoned.load(Ordering::Acquire))
            .then_some(batches)
            .map(|batches| match &self.condense {
                Some(condense) => condense(batches),
                None => batches,
            });
        PartialAggregateCache::global().insert(self.key.clone(), judged(kept, fresh, merge));
    }
}

/// What a gathering run leaves behind: its partials, when it kept them and
/// their merge took clearly less than aggregating afresh, else a dense mark.
fn judged(kept: Option<Vec<RecordBatch>>, fresh: Duration, merge: Duration) -> RememberedPartials {
    match kept {
        Some(batches) if merge_pays_off(merge, fresh, MERGE_SHARE_TO_KEEP) => {
            RememberedPartials::Partials(Arc::new(Partials { batches, fresh }))
        }
        _ => RememberedPartials::Dense,
    }
}

pub(crate) struct PartialsTapFactory {
    pub(crate) gathering: Arc<Gathering>,
}

impl UnaryFactory<RecordBatch, RecordBatch> for PartialsTapFactory {
    type Unary = PartialsTap;

    fn build_unary(self) -> Self::Unary {
        PartialsTap {
            gathering: self.gathering,
        }
    }
}

/// Passes the gathered partials through, noting each batch with the run.
pub(crate) struct PartialsTap {
    gathering: Arc<Gathering>,
}

impl Unary<RecordBatch, RecordBatch> for PartialsTap {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        self.gathering.note(&batch);
        sender.send(batch)?;
        Ok(())
    }
}

/// Watches a merge to its end across every worker: the last worker to finish
/// runs what the watch was given.
pub(crate) struct MergeWatch {
    workers_left: AtomicUsize,
    when_done: Box<dyn Fn() + Send + Sync>,
}

impl MergeWatch {
    /// One watching pass-through per worker, running `when_done` once all of
    /// them have finished.
    pub(crate) fn over_workers(
        workers: usize,
        when_done: impl Fn() + Send + Sync + 'static,
    ) -> impl Iterator<Item = MergeWatchFactory> {
        let watch = Arc::new(MergeWatch {
            workers_left: AtomicUsize::new(workers),
            when_done: Box::new(when_done),
        });
        (0..workers).map(move |_| MergeWatchFactory {
            watch: watch.clone(),
        })
    }
}

pub(crate) struct MergeWatchFactory {
    watch: Arc<MergeWatch>,
}

impl UnaryFactory<RecordBatch, RecordBatch> for MergeWatchFactory {
    type Unary = MergeWatcher;

    fn build_unary(self) -> Self::Unary {
        MergeWatcher { watch: self.watch }
    }
}

pub(crate) struct MergeWatcher {
    watch: Arc<MergeWatch>,
}

impl Unary<RecordBatch, RecordBatch> for MergeWatcher {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        sender.send(batch)?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<bool> {
        if self.watch.workers_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            (self.watch.when_done)();
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;

    fn key(shape: &str) -> PartialsKey {
        PartialsKey {
            table: "t".to_string(),
            version: "1".to_string(),
            shape: shape.to_string(),
        }
    }

    fn one_row_batch(value: i64) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("c", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![value])) as ArrayRef],
        )
        .unwrap()
    }

    #[test]
    fn a_shape_is_met_then_remembered_until_the_budget_evicts_it() {
        let cache = PartialAggregateCache::with_budget(200);
        let first = key("a");

        cache.insert(first.clone(), RememberedPartials::Met);
        let met = matches!(cache.get(&first), Some(RememberedPartials::Met));
        cache.insert(first.clone(), remembered_partials(one_row_batch(1)));
        let remembered = matches!(cache.get(&first), Some(RememberedPartials::Partials(_)));
        cache.insert(key("b"), remembered_partials(one_row_batch(2)));
        let evicted = cache.get(&first).is_none();

        assert!(met);
        assert!(remembered);
        assert!(evicted);
        assert_eq!(cache.len(), 1);
    }

    fn remembered_partials(batch: RecordBatch) -> RememberedPartials {
        RememberedPartials::Partials(Arc::new(Partials {
            batches: vec![batch],
            fresh: Duration::from_millis(1),
        }))
    }

    #[test]
    fn a_gathering_keeps_its_partials_only_when_their_merge_was_clearly_the_cheaper_half() {
        let fresh = Duration::from_millis(100);
        let kept = || Some(vec![one_row_batch(1)]);

        let quick_merge = judged(kept(), fresh, Duration::from_millis(30));
        let slow_merge = judged(kept(), fresh, Duration::from_millis(90));
        let let_go = judged(None, fresh, Duration::from_millis(1));

        assert!(matches!(quick_merge, RememberedPartials::Partials(_)));
        assert!(matches!(slow_merge, RememberedPartials::Dense));
        assert!(matches!(let_go, RememberedPartials::Dense));
    }

    #[test]
    fn a_gathering_lets_its_partials_go_once_they_outgrow_the_row_limit() {
        let gathering = Gathering::start(key("a"), Some(2 * DENSE_DIVISOR), None);

        gathering.note(&one_row_batch(1));
        gathering.note(&one_row_batch(2));
        let kept_before_the_limit = gathering.batches.lock().unwrap().len();
        gathering.note(&one_row_batch(3));

        assert_eq!(kept_before_the_limit, 2);
        assert!(gathering.abandoned.load(Ordering::Acquire));
        assert!(gathering.batches.lock().unwrap().is_empty());
    }

    #[test]
    fn a_run_encoded_row_group_column_expands_to_one_value_per_row() {
        let values = UInt32Array::from(vec![7u32]);
        let run_ends = arrow_array::Int32Array::from(vec![3]);
        let encoded = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("c", DataType::Int64, false),
                Field::new("row_group_idx", encoded.data_type().clone(), false),
                Field::new("row_idx", DataType::UInt32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(encoded) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0u32, 1, 2])) as ArrayRef,
            ],
        )
        .unwrap();

        let plain = with_plain_row_group(batch, 1);

        assert_eq!(plain.num_columns(), 2);
        let groups = plain
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(groups.values().as_ref(), &[7i32, 7, 7]);
    }

    #[test]
    fn a_copied_batch_owns_its_memory_and_keeps_its_rows() {
        let batch = one_row_batch(42);

        let copy = copied(&batch);

        let original_buffer = batch.column(0).to_data().buffers()[0].as_ptr();
        let copied_buffer = copy.column(0).to_data().buffers()[0].as_ptr();
        assert_eq!(copy.num_rows(), 1);
        assert_ne!(copied_buffer, original_buffer);
    }

    #[test]
    fn a_copied_view_column_points_into_buffers_of_its_own() {
        let long = "a value long enough to live outside its view";
        let strings = arrow_array::StringViewArray::from(vec![long, "short"]);
        let original_block = strings.data_buffers()[0].as_ptr();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "s",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(strings) as ArrayRef],
        )
        .unwrap();

        let copy = copied(&batch);

        let copied_strings = copy.column(0).as_string_view();
        assert_eq!(copied_strings.value(0), long);
        assert_eq!(copied_strings.value(1), "short");
        assert!(
            copied_strings
                .data_buffers()
                .iter()
                .all(|block| block.as_ptr() != original_block)
        );
    }
}
