//! Optional normalization stage for pipeline-breaker batch groups.
//!
//! An operation-specific consumer produces one [`NormalizationBatches`] per worker. In
//! the common path its normal outputter gathers those groups directly. When
//! variant columns may have per-file physical layouts, the same consumer can
//! instead use [`Normalizer`] as its outputter. The normalizer gathers every
//! group, normalizes mismatched batches in parallel, and emits the groups into
//! a second pipeline breaker whose [`Collector`] initializes the original
//! operation-specific outputter.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use crossbeam_deque::{Injector, Steal};

use crate::GatherBarrier;
use crate::operations::channels::Sender;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, Unary};

/// A worker-local result containing the record batches an operation will
/// combine after its consume phase.
pub(crate) trait NormalizationBatches: Send + 'static {
    /// Visit every batch mutably. Implementations must visit each stored batch
    /// exactly once and must not move their storage after this call returns.
    fn visit_batches_mut(&mut self, visit: &mut dyn FnMut(&mut RecordBatch));
}

/// An outputter that accepts the batch group produced by its own consumer.
///
/// Direct operation outputters gather these groups themselves. [`Normalizer`]
/// implements the same interface, which is what makes the two outputters
/// interchangeable on the first pipeline breaker.
pub(crate) trait BatchesOutputter<G, O>: Outputter<O> {
    fn accept(&mut self, group: G) -> unary::Result<()>;
}

/// An operation outputter that can be initialized from all worker groups.
///
/// [`Collector`] uses this in the second pipeline breaker of the normalized
/// path. The same outputter also implements [`BatchesOutputter`] for the direct
/// one-breaker path.
pub(crate) trait InitializableOutputter<G, O>: Outputter<O> {
    fn initialize(&mut self, groups: Vec<G>) -> unary::Result<()>;
}

/// A stable pointer to one batch inside the gathered groups.
///
/// `NormalizerState::initialize` creates exactly one slot per batch after the
/// groups have reached their final allocation. Jobs never move the groups and
/// each slot is assigned to exactly one job, so concurrent writes are disjoint.
#[derive(Clone, Copy)]
struct BatchSlot(NonNull<RecordBatch>);

unsafe impl Send for BatchSlot {}

struct NormalizationJob {
    slot: BatchSlot,
    variant_columns: Arc<[usize]>,
    remaining: Arc<AtomicUsize>,
}

impl NormalizationJob {
    fn run(self) -> unary::Result<()> {
        // The old batch is cloned before replacement so the normalization
        // kernels can read it without borrowing the slot being written.
        let batch = unsafe { self.slot.0.as_ref() }.clone();
        let result = crate::arrays::variant::unshred_variant_columns(&batch, &self.variant_columns)
            .map(|normalized| {
                // Replace rather than `write`: the old RecordBatch owns
                // the array references this slot is exchanging and must
                // be dropped.
                drop(unsafe { self.slot.0.as_ptr().replace(normalized) });
            })
            .map_err(unary::Error::from);
        // Release-publish the replacement. The emitting outputter's Acquire
        // load observes every replacement before taking the groups.
        self.remaining.fetch_sub(1, Ordering::Release);
        result
    }
}

struct NormalizerState<G> {
    gather: GatherBarrier<G>,
    jobs: Injector<NormalizationJob>,
    groups: UnsafeCell<Option<Vec<G>>>,
    remaining: Arc<AtomicUsize>,
    jobs_injected: AtomicBool,
    output_claimed: AtomicBool,
}

// Before publication, only the final gather arrival can access `groups`.
// Afterwards normalization jobs write distinct BatchSlots. Once `remaining`
// reaches zero, exactly one outputter takes the groups. Those phases are
// ordered by the atomics documented above.
unsafe impl<G: NormalizationBatches> Sync for NormalizerState<G> {}

impl<G: NormalizationBatches> NormalizerState<G> {
    fn new(worker_count: usize) -> Self {
        Self {
            gather: GatherBarrier::new(worker_count),
            jobs: Injector::new(),
            groups: UnsafeCell::new(None),
            remaining: Arc::new(AtomicUsize::new(0)),
            jobs_injected: AtomicBool::new(false),
            output_claimed: AtomicBool::new(false),
        }
    }

    fn initialize(&self, groups: Vec<G>) {
        unsafe { *self.groups.get() = Some(groups) };

        let mut slots = Vec::new();
        let groups = unsafe {
            (*self.groups.get())
                .as_mut()
                .expect("groups were just stored")
        };
        for group in groups {
            group.visit_batches_mut(&mut |batch| {
                slots.push(BatchSlot(NonNull::from(batch)));
            });
        }

        let batches: Vec<RecordBatch> = slots
            .iter()
            .map(|slot| unsafe { slot.0.as_ref() }.clone())
            .collect();
        let variant_columns = crate::arrays::variant::mismatched_variant_columns(&batches);
        if !variant_columns.is_empty() {
            self.remaining.store(slots.len(), Ordering::Relaxed);
            let variant_columns: Arc<[usize]> = variant_columns.into();
            for slot in slots {
                self.jobs.push(NormalizationJob {
                    slot,
                    variant_columns: variant_columns.clone(),
                    remaining: self.remaining.clone(),
                });
            }
        }
        self.jobs_injected.store(true, Ordering::Release);
    }
}

/// Outputter that normalizes a gathered set of [`NormalizationBatches`] and emits the
/// groups unchanged in shape once every normalization job has landed.
pub(crate) struct Normalizer<G> {
    shared: Arc<NormalizerState<G>>,
}

impl<G: NormalizationBatches> Normalizer<G> {
    pub(crate) fn create_for_workers(worker_count: usize) -> Vec<Self> {
        let shared = Arc::new(NormalizerState::new(worker_count));
        (0..worker_count)
            .map(|_| Self {
                shared: shared.clone(),
            })
            .collect()
    }
}

impl<G: NormalizationBatches> BatchesOutputter<G, G> for Normalizer<G> {
    fn accept(&mut self, group: G) -> unary::Result<()> {
        let shared = self.shared.clone();
        self.shared.gather.arrive(group, |groups| {
            shared.initialize(groups);
        });
        Ok(())
    }
}

impl<G: NormalizationBatches> Outputter<G> for Normalizer<G> {
    fn output(&mut self, sender: &mut dyn Sender<G>) -> unary::Result<bool> {
        match self.shared.jobs.steal() {
            Steal::Success(job) => {
                job.run()?;
                return Ok(false);
            }
            Steal::Retry => return Ok(false),
            Steal::Empty => {}
        }

        if !self.shared.jobs_injected.load(Ordering::Acquire)
            || self.shared.remaining.load(Ordering::Acquire) != 0
        {
            return Ok(false);
        }

        if !self.shared.output_claimed.swap(true, Ordering::AcqRel) {
            let groups = unsafe {
                (*self.shared.groups.get())
                    .take()
                    .expect("the claimed normalizer output exists")
            };
            for group in groups {
                sender.send(group)?;
            }
        }
        Ok(true)
    }
}

/// Consumer for the second pipeline breaker in the normalized path.
///
/// Its channel funnels every group to one selected worker. That worker passes
/// all groups to the final outputter; every worker then returns its copy of the
/// shared outputter so operation jobs remain parallel.
pub(crate) struct Collector<G, O> {
    groups: Vec<G>,
    outputter: O,
    /// Whether this is the one worker the collectors' `to_single_worker_mpsc`
    /// channel delivers every group to, and so the one that initializes the
    /// outputter. The flag is decided up front rather than inferred from
    /// having received groups, because an empty input legitimately delivers
    /// no groups at all, and the initialization must still run on exactly one
    /// worker: some outputters, such as the join's build table, can only be
    /// set up once.
    is_collector_worker: bool,
}

impl<G, O, Out> Consumer<G, Out> for Collector<G, O>
where
    O: InitializableOutputter<G, Out>,
{
    type Outputter = O;

    fn consume(&mut self, group: G, _sender: &mut dyn Sender<Out>) -> unary::Result<()> {
        debug_assert!(
            self.is_collector_worker,
            "the collectors' channel delivers groups to the collector worker alone"
        );
        self.groups.push(group);
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        if self.is_collector_worker {
            self.outputter.initialize(self.groups)?;
        } else {
            debug_assert!(
                self.groups.is_empty(),
                "a worker other than the collector worker received groups"
            );
        }
        Ok(Some(self.outputter))
    }
}

/// Builds one [`Collector`] pipeline breaker.
pub(crate) struct CollectorFactory<G, O> {
    outputter: O,
    is_collector_worker: bool,
    _group: PhantomData<fn() -> G>,
}

impl<G, O> CollectorFactory<G, O> {
    pub(crate) fn new(outputter: O, is_collector_worker: bool) -> Self {
        Self {
            outputter,
            is_collector_worker,
            _group: PhantomData,
        }
    }
}

impl<G: Send + 'static, O: Send + 'static, Out> UnaryFactory<G, Out> for CollectorFactory<G, O>
where
    O: InitializableOutputter<G, Out>,
    Collector<G, O>: Consumer<G, Out>,
    PipelineBreaker<G, Out, Collector<G, O>>: Unary<G, Out>,
{
    type Unary = PipelineBreaker<G, Out, Collector<G, O>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(Collector {
            groups: Vec::new(),
            outputter: self.outputter,
            is_collector_worker: self.is_collector_worker,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::CollectSender;
    use crate::waker::install_test_worker_waker;
    use crate::worker::WORKER_IDX;
    use arrow_array::{ArrayRef, StringArray};
    use arrow_schema::{DataType, Schema};
    use parquet_variant_compute::{
        ShreddedSchemaBuilder, VariantArray, json_to_variant, shred_variant,
    };

    struct TestGroup(Vec<RecordBatch>);

    impl NormalizationBatches for TestGroup {
        fn visit_batches_mut(&mut self, visit: &mut dyn FnMut(&mut RecordBatch)) {
            for batch in &mut self.0 {
                visit(batch);
            }
        }
    }

    fn docs_batch(rows: Vec<&str>, shred_age: bool) -> RecordBatch {
        let json: ArrayRef = Arc::new(StringArray::from(rows));
        let variant = json_to_variant(&json).unwrap();
        let variant = if shred_age {
            let schema = ShreddedSchemaBuilder::new()
                .with_path("age", &DataType::Int64)
                .unwrap()
                .build();
            shred_variant(&variant, &schema).unwrap()
        } else {
            variant
        };
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![variant.field("d")])),
            vec![Arc::new(variant.into_inner())],
        )
        .unwrap()
    }

    fn normalize(groups: Vec<TestGroup>) -> Vec<TestGroup> {
        install_test_worker_waker();
        let mut normalizers = Normalizer::create_for_workers(groups.len());
        for (worker, (normalizer, group)) in normalizers.iter_mut().zip(groups).enumerate() {
            WORKER_IDX.set(worker);
            normalizer.accept(group).unwrap();
        }
        WORKER_IDX.set(0);

        let mut sender = CollectSender::new();
        loop {
            let mut done = true;
            for normalizer in &mut normalizers {
                done &= normalizer.output(&mut sender).unwrap();
            }
            if done {
                return sender.items;
            }
        }
    }

    #[test]
    fn matching_layouts_pass_through_without_replacing_batches() {
        let first = docs_batch(vec![r#"{"age":30}"#], true);
        let second = docs_batch(vec![r#"{"age":31}"#], true);
        let first_column = first.column(0).clone();

        let groups = normalize(vec![TestGroup(vec![first]), TestGroup(vec![second])]);

        assert!(Arc::ptr_eq(groups[0].0[0].column(0), &first_column));
    }

    #[test]
    fn mismatched_layouts_are_replaced_before_groups_are_emitted() {
        let shredded = docs_batch(vec![r#"{"age":30}"#], true);
        let unshredded = docs_batch(vec![r#"{"age":31}"#], false);

        let groups = normalize(vec![TestGroup(vec![shredded]), TestGroup(vec![unshredded])]);
        let batches: Vec<RecordBatch> = groups.into_iter().flat_map(|group| group.0).collect();
        assert_eq!(batches[0].schema(), batches[1].schema());

        let merged = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
        let variants = VariantArray::try_new(merged.column(0).as_ref()).unwrap();
        let ages: Vec<i64> = (0..variants.len())
            .map(|row| {
                variants
                    .value(row)
                    .get_object_field("age")
                    .unwrap()
                    .as_int64()
                    .unwrap()
            })
            .collect();
        assert_eq!(ages, [30, 31]);
    }
}
