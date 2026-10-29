//! Top rows remembered per row group.
//!
//! An `ORDER BY … LIMIT` over one table's scan ranks the same rows whether it
//! sees the whole table or only each row group's own top rows, so those rows,
//! like an aggregate's partials, depend only on the table's data and the
//! query's shape, and can serve every later run of it: the scan is skipped
//! and the remembered rows are ranked instead. The shape is remembered on
//! the terms of the partial aggregates: met once, gathered on its second
//! run, kept only when ranking the remembered rows is clearly cheaper than
//! the scan, and forgotten when it stops being so.
//!
//! The gathering run tags each scanned row with its row group, and each
//! worker keeps the top rows of every row group it sees; what the workers
//! kept is boiled down to one set of top rows per row group when filed.

use crate::Operator;
use crate::compile::{Error, RuntimeFilterSlots};
use crate::expression::Expression;
use crate::operator::{OrderByDirection, TopN};
use crate::partial_aggregates::{
    BeginTapFactory, FreshSpan, Gathering, MERGE_SHARE_TO_FORGET, MergeWatch,
    PartialAggregateCache, PartialsKey, PartialsTapFactory, RememberedPartials, RunClock,
    ScanShape, merge_pays_off,
};
use crate::plan::PlanNode;
use arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, take_record_batch};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt32Type};
use arrow_array::{RecordBatch, RunArray};
use arrow_schema::SortOptions;
use dispatch::{
    BoundarySlot, DataFlowDispatcher, OperatorIO, OrderBy, RecordBatchOperatorSpec, Sender, Unary,
    UnaryFactory, trailing_metadata_columns, values_input,
};
use std::collections::HashMap;
use std::sync::Arc;

/// A top-N keeping more rows than this per row group is left to the scan.
const ROWS_PER_GROUP_LIMIT: usize = 1 << 12;

/// Compiles a top-N over a plain scan from each row group's remembered top
/// rows, or, the second time the shape is met, as a plan that gathers them;
/// `None` leaves the node to the ordinary compile.
pub(crate) fn try_compile_memoized(
    node: &PlanNode,
    dispatcher: &DataFlowDispatcher,
    slots: &mut RuntimeFilterSlots,
) -> Result<Option<RecordBatchOperatorSpec>, Error> {
    let cache = PartialAggregateCache::global();
    if !cache.is_enabled() {
        return Ok(None);
    }
    let Operator::TopN(top_n) = &node.operator else {
        return Ok(None);
    };
    let [child] = node.inputs.as_slice() else {
        return Ok(None);
    };
    let Some(ranking) = Ranking::of(top_n) else {
        return Ok(None);
    };
    let published_slot = top_n
        .produces_dynamic_filter
        .as_ref()
        .map(|filter| filter.slot_id);
    let Some(shape) = ScanShape::recognize(child, published_slot) else {
        return Ok(None);
    };
    let revision = shape.input.table.table_revision();
    let key = PartialsKey {
        table: revision.identity,
        version: revision.version,
        shape: node.to_string(),
    };
    let workers = dispatcher.worker_count();
    match cache.get(&key) {
        None => {
            cache.insert(key, RememberedPartials::Met);
            Ok(None)
        }
        Some(RememberedPartials::Dense) => Ok(None),
        Some(RememberedPartials::Partials(partials)) => {
            let clock = Arc::new(RunClock::new());
            let source = values_input(dispatcher, partials.batches.iter().cloned())
                .map_each(|batch: RecordBatch| batch)
                .record_batches()
                .with_unary(BeginTapFactory::over_workers(workers, &clock));
            let ranked = ranking.rank(source, &shape, None);
            let fresh = partials.fresh;
            let forget_a_slow_ranking = move || {
                let ranking_took = clock.since_began();
                let pays_off = merge_pays_off(ranking_took, fresh, MERGE_SHARE_TO_FORGET);
                tracing::debug!(
                    shape = %key.shape,
                    fresh_ms = fresh.as_secs_f64() * 1e3,
                    ranking_ms = ranking_took.as_secs_f64() * 1e3,
                    kept = pays_off,
                    "top rows ranked"
                );
                if !pays_off {
                    cache.insert(key.clone(), RememberedPartials::Dense);
                }
            };
            Ok(Some(ranked.with_unary(MergeWatch::over_workers(
                workers,
                forget_a_slow_ranking,
            ))))
        }
        Some(RememberedPartials::Met) => {
            let condensing = ranking.clone();
            let gathering = Arc::new(Gathering::start(
                key,
                shape.input.table.estimate_row_count(),
                Some(Box::new(move |batches| {
                    condensing.top_rows_per_row_group(batches)
                })),
                FreshSpan::WholeRun,
            ));
            // The gathering's own ranking publishes its boundary like the
            // top-N would, so the scan prunes the row groups it can; a row
            // group pruned away holds no top rows for the shape, and leaves
            // none behind.
            let boundary = top_n
                .produces_dynamic_filter
                .as_ref()
                .map(|filter| slots.boundary_slot(filter.slot_id));
            let mut spec = shape
                .compile_tagged_scan(dispatcher, slots)?
                .with_unary(BeginTapFactory::over_workers(workers, gathering.clock()));
            if let Some(filter) = shape.filter {
                spec = filter.compile(spec)?;
            }
            for projection in &shape.projections {
                spec = projection.compile(spec)?;
            }
            let per_row_group = spec.with_unary((0..workers).map(|_| RowGroupTopFactory {
                ranking: ranking.clone(),
            }));
            let tapped = per_row_group.with_unary((0..workers).map(|_| PartialsTapFactory {
                gathering: gathering.clone(),
            }));
            let ranked = ranking.rank(tapped, &shape, boundary);
            let file_the_rows = move || gathering.file();
            Ok(Some(ranked.with_unary(MergeWatch::over_workers(
                workers,
                file_the_rows,
            ))))
        }
    }
}

/// How a top-N ranks rows: the columns it orders by, and how many rows it
/// keeps.
#[derive(Clone)]
struct Ranking {
    orders: Vec<(usize, SortOptions)>,
    limit: usize,
    offset: usize,
}

impl Ranking {
    /// The ranking of `top_n`, when it orders by columns alone and keeps few
    /// enough rows to remember per row group.
    fn of(top_n: &TopN) -> Option<Self> {
        let keep = top_n.limit + top_n.offset;
        if keep == 0 || keep > ROWS_PER_GROUP_LIMIT {
            return None;
        }
        let orders = top_n
            .order_bys
            .iter()
            .map(|order| {
                let Expression::Ref(column) = &order.expression else {
                    return None;
                };
                let options = SortOptions {
                    descending: matches!(order.direction, OrderByDirection::Desc),
                    nulls_first: order.nulls_first,
                };
                Some((column.column_idx, options))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            orders,
            limit: top_n.limit,
            offset: top_n.offset,
        })
    }

    fn keep(&self) -> usize {
        self.limit + self.offset
    }

    /// `input` ranked as the top-N would be, publishing its running boundary
    /// into `boundary` when a scan below consumes it, and then, when the
    /// shape's own scan carried no row-group tags, with the tags dropped
    /// again.
    fn rank(
        &self,
        input: RecordBatchOperatorSpec,
        shape: &ScanShape<'_>,
        boundary: Option<Arc<BoundarySlot>>,
    ) -> RecordBatchOperatorSpec {
        let orders = self
            .orders
            .iter()
            .map(|(column, options)| OrderBy::new(*column, options.descending, options.nulls_first))
            .collect();
        let ranked = input.order_by_limit_offset(orders, self.limit, self.offset, boundary);
        if shape.input.emit_row_group_metadata {
            ranked
        } else {
            ranked.project(|| without_row_group_tags)
        }
    }

    /// The rows of `batch` ranking first, as many as the top-N keeps.
    fn top_rows(&self, batch: &RecordBatch) -> RecordBatch {
        let columns: Vec<SortColumn> = self
            .orders
            .iter()
            .map(|(column, options)| SortColumn {
                values: batch.column(*column).clone(),
                options: Some(*options),
            })
            .collect();
        let indices =
            lexsort_to_indices(&columns, Some(self.keep())).expect("the order columns sort");
        take_record_batch(batch, &indices).expect("sorted indices index the batch")
    }

    /// One set of top rows per row group out of what every worker kept.
    fn top_rows_per_row_group(&self, kept: Vec<RecordBatch>) -> Vec<RecordBatch> {
        let mut by_row_group: HashMap<u32, Vec<RecordBatch>> = HashMap::new();
        for batch in kept.into_iter().filter(|batch| batch.num_rows() > 0) {
            by_row_group
                .entry(row_group_of(&batch))
                .or_default()
                .push(batch);
        }
        let mut row_groups: Vec<_> = by_row_group.into_iter().collect();
        row_groups.sort_by_key(|(row_group, _)| *row_group);
        row_groups
            .into_iter()
            .map(|(_, batches)| {
                let all = concat_batches(&batches[0].schema(), &batches)
                    .expect("one shape's rows share a schema");
                self.top_rows(&all)
            })
            .collect()
    }
}

/// The row group of a batch whose rows all share one.
fn row_group_of(batch: &RecordBatch) -> u32 {
    row_groups_column(batch)
        .values()
        .as_primitive::<UInt32Type>()
        .value(0)
}

/// `batch` cut at its row-group boundaries, each piece with its row group.
fn row_group_slices(batch: &RecordBatch) -> Vec<(u32, RecordBatch)> {
    let encoded = row_groups_column(batch);
    let row_groups = encoded.values().as_primitive::<UInt32Type>();
    let run_ends = encoded.run_ends();
    let offset = run_ends.offset();
    let rows = run_ends.len();
    let mut slices = Vec::new();
    let mut start = 0usize;
    for (run, &run_end) in run_ends.values().iter().enumerate() {
        let end = (run_end as usize).saturating_sub(offset).min(rows);
        if end > start {
            slices.push((row_groups.value(run), batch.slice(start, end - start)));
            start = end;
        }
        if start >= rows {
            break;
        }
    }
    slices
}

/// The run-encoded row group tag a tagged batch carries beside its rows.
fn row_groups_column(batch: &RecordBatch) -> &RunArray<Int32Type> {
    let tags = trailing_metadata_columns(batch.schema_ref());
    assert_eq!(tags, 2, "a gathering scan tags each row with its row group");
    batch
        .column(batch.num_columns() - tags)
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .expect("a scan emitting row-group metadata run-encodes the row group")
}

/// `batch` without the trailing row-group tags.
fn without_row_group_tags(batch: RecordBatch) -> RecordBatch {
    let own = batch.num_columns() - trailing_metadata_columns(batch.schema_ref());
    batch
        .project(&(0..own).collect::<Vec<_>>())
        .expect("the leading columns are the batch's own")
}

struct RowGroupTopFactory {
    ranking: Ranking,
}

impl UnaryFactory<RecordBatch, RecordBatch> for RowGroupTopFactory {
    type Unary = RowGroupTop;

    fn build_unary(self) -> Self::Unary {
        RowGroupTop {
            ranking: self.ranking,
            kept: HashMap::new(),
        }
    }
}

/// Keeps, per row group, the top rows of the tagged batches one worker sees.
/// A row group's rows are emitted as soon as the worker moves on to another
/// row group, so the ranking after it can tighten its boundary while the
/// scan is still going; rows of a row group met again are emitted again, and
/// the ranking takes the union. Whatever is left is emitted once the input
/// is drained.
struct RowGroupTop {
    ranking: Ranking,
    kept: HashMap<u32, RecordBatch>,
}

impl RowGroupTop {
    /// Folds `batch` into the kept rows and hands back the top rows of the
    /// row groups the worker has moved on from.
    fn note(&mut self, batch: &RecordBatch) -> Vec<RecordBatch> {
        let mut done = Vec::new();
        for (row_group, slice) in row_group_slices(batch) {
            let left_behind: Vec<u32> = self
                .kept
                .keys()
                .filter(|&&kept| kept != row_group)
                .copied()
                .collect();
            for kept in left_behind {
                done.push(self.kept.remove(&kept).expect("a listed row group is kept"));
            }
            let candidates = match self.kept.remove(&row_group) {
                Some(kept) => concat_batches(&batch.schema(), [&kept, &slice])
                    .expect("a worker's kept rows share the batch's schema"),
                None => slice,
            };
            self.kept
                .insert(row_group, self.ranking.top_rows(&candidates));
        }
        done
    }
}

impl Unary<RecordBatch, RecordBatch> for RowGroupTop {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        for done in self.note(&batch) {
            sender.send(done)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<bool> {
        for (_, batch) in std::mem::take(&mut self.kept) {
            sender.send(batch)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int64Type;
    use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{ROW_GROUP_IDX_FIELD, ROW_IDX_FIELD};

    /// Rows of `values` tagged with row groups: each entry of `run_ends`
    /// closes the run of the matching entry of `row_groups`.
    fn tagged(values: &[i64], run_ends: &[i32], row_groups: &[u32]) -> RecordBatch {
        let encoded = RunArray::<Int32Type>::try_new(
            &Int32Array::from(run_ends.to_vec()),
            &UInt32Array::from(row_groups.to_vec()),
        )
        .unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::Int64, false),
                Field::new(ROW_GROUP_IDX_FIELD, encoded.data_type().clone(), false),
                Field::new(ROW_IDX_FIELD, DataType::UInt32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
                Arc::new(encoded) as ArrayRef,
                Arc::new(UInt32Array::from_iter_values(0..values.len() as u32)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn ascending(keep: usize) -> Ranking {
        Ranking {
            orders: vec![(
                0,
                SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            )],
            limit: keep,
            offset: 0,
        }
    }

    fn values(batch: &RecordBatch) -> Vec<i64> {
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn a_worker_keeps_each_row_groups_top_rows_across_batches() {
        let mut top = RowGroupTop {
            ranking: ascending(2),
            kept: HashMap::new(),
        };

        let mut emitted = top.note(&tagged(&[9, 1, 5, 7], &[3, 4], &[0, 1]));
        emitted.extend(top.note(&tagged(&[4, 2], &[2], &[1])));

        assert_eq!(emitted.len(), 1);
        assert_eq!(values(&emitted[0]), vec![1, 5]);
        assert_eq!(values(&top.kept[&1]), vec![2, 4]);
    }

    #[test]
    fn what_workers_kept_separately_condenses_to_one_set_of_top_rows_per_row_group() {
        let ranking = ascending(1);
        let kept = vec![
            tagged(&[4], &[1], &[0]),
            tagged(&[2], &[1], &[0]),
            tagged(&[8], &[1], &[1]),
        ];

        let condensed = ranking.top_rows_per_row_group(kept);

        assert_eq!(condensed.len(), 2);
        assert_eq!(values(&condensed[0]), vec![2]);
        assert_eq!(values(&condensed[1]), vec![8]);
    }

    #[test]
    fn dropping_the_row_group_tags_leaves_the_rows_own_columns() {
        let batch = tagged(&[3, 1], &[2], &[0]);

        let plain = without_row_group_tags(batch);

        assert_eq!(plain.num_columns(), 1);
        assert_eq!(values(&plain), vec![3, 1]);
    }
}
