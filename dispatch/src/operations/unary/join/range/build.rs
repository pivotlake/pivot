//! The build side of the range join: receive the ORDER BY's sorted chunks and
//! publish them.
//!
//! The heavy lifting happened upstream — the build input was already sorted by
//! the key (NULLs last) by the ORDER BY operator, and a single-worker funnel
//! delivers every chunk to one consumer in emission order (the contract
//! [`to_single_worker_mpsc`] documents; it holds because the sort emits from
//! one worker). The publishing operator prepares each chunk as it arrives and,
//! once its input closes, publishes the finished table and flips the readiness
//! flag that gates the probe input. Every other worker's operator receives
//! nothing.
//!
//! [`to_single_worker_mpsc`]: crate::operations::channels::to_single_worker_mpsc

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::Array;
use arrow_array::ArrowNativeTypeOp;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;

use crate::operations::channels::Sender;
use crate::operations::unary::join::JoinCell;
use crate::operations::unary::join::range::probe::RangeProbe;
use crate::operations::unary::join::range::{RangeJoinSpec, RangeTable, SortedChunk};
use crate::operations::unary::{self, Unary, UnaryFactory};
use crate::waker::waker_set;

/// Creates one [`RangeBuild`] per worker; `target` marks the publisher, which
/// must be the worker the build funnel delivers to.
pub struct RangeBuildFactory<T: ArrowPrimitiveType> {
    spec: Arc<RangeJoinSpec>,
    publisher: bool,
    table: RangeTable<T>,
    build_ready: Arc<AtomicBool>,
}

impl<T: ArrowPrimitiveType> UnaryFactory<RecordBatch, ()> for RangeBuildFactory<T>
where
    T::Native: ArrowNativeTypeOp + Send,
{
    type Unary = RangeBuild<T>;

    fn build_unary(self) -> Self::Unary {
        RangeBuild {
            spec: self.spec,
            publisher: self.publisher,
            chunks: Vec::new(),
            saw_null_tail: false,
            table: self.table,
            build_ready: self.build_ready,
        }
    }
}

/// Creates one [`RangeProbe`] per worker, all sharing the published table.
pub struct RangeProbeFactory<T: ArrowPrimitiveType> {
    spec: Arc<RangeJoinSpec>,
    table: RangeTable<T>,
}

impl<T: ArrowPrimitiveType> UnaryFactory<RecordBatch, RecordBatch> for RangeProbeFactory<T>
where
    T::Native: ArrowNativeTypeOp + Send,
{
    type Unary = RangeProbe<T>;

    fn build_unary(self) -> Self::Unary {
        RangeProbe::new(self.table, self.spec)
    }
}

/// Create `worker_count` build and probe factories sharing one table and one
/// readiness flag. `target` is the worker the build funnel delivers every
/// chunk to; its build factory is the publisher. The graph builder gates
/// every probe-input root on the returned flag, exactly as for the hash join.
pub(crate) fn create_range_join_factories<T: ArrowPrimitiveType>(
    spec: RangeJoinSpec,
    worker_count: usize,
    target: usize,
) -> (
    impl IntoIterator<Item = RangeBuildFactory<T>>,
    impl IntoIterator<Item = RangeProbeFactory<T>>,
    Arc<AtomicBool>,
)
where
    T::Native: ArrowNativeTypeOp + Send,
{
    let spec = Arc::new(spec);
    let table = RangeTable {
        chunks: Arc::new(JoinCell::new(Vec::new())),
    };
    let build_ready = Arc::new(AtomicBool::new(false));

    let probe_spec = spec.clone();
    let probe_table = table.clone();
    let gate = build_ready.clone();

    let build_factories = (0..worker_count).map(move |worker_id| RangeBuildFactory {
        spec: spec.clone(),
        publisher: worker_id == target,
        table: table.clone(),
        build_ready: build_ready.clone(),
    });
    let probe_factories = (0..worker_count).map(move |_| RangeProbeFactory {
        spec: probe_spec.clone(),
        table: probe_table.clone(),
    });

    (build_factories, probe_factories, gate)
}

/// Per-worker build operator. Only the funnel's target worker ever receives
/// chunks; it prepares them as they arrive and publishes the table in
/// [`Unary::finish`] after the complete sorted input has been delivered.
pub struct RangeBuild<T: ArrowPrimitiveType> {
    spec: Arc<RangeJoinSpec>,
    publisher: bool,
    chunks: Vec<SortedChunk<T>>,
    saw_null_tail: bool,
    table: RangeTable<T>,
    build_ready: Arc<AtomicBool>,
}

impl<T: ArrowPrimitiveType> Unary<RecordBatch, ()> for RangeBuild<T>
where
    T::Native: ArrowNativeTypeOp + Send,
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<()>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        debug_assert!(self.publisher, "the funnel delivers to one worker only");
        if batch.num_rows() == 0 || self.saw_null_tail {
            return Ok(());
        }
        // Matches gather build rows across the stored chunks: a variant
        // column's per-file layouts must agree before rows are spliced, or a
        // shredded row lands under another file's layout and silently loses
        // its typed leaves.
        let batch =
            crate::arrays::variant::unshred_batch_variants(batch).map_err(unary::Error::from)?;

        let keys = batch.column(self.spec.build_key_index).as_primitive::<T>();
        let non_null = keys.len() - keys.null_count();
        if non_null == 0 {
            // Entirely NULL-keyed; this and everything after it is tail.
            self.saw_null_tail = true;
            return Ok(());
        }
        debug_assert!(
            self.chunks
                .last()
                .is_none_or(|last| { last.keys.value(last.keys.len() - 1).is_le(keys.value(0)) }),
            "sorted chunks must arrive in ascending key order"
        );

        let began_null_tail = non_null < batch.num_rows();
        let kept_keys = keys.slice(0, non_null);
        let kept = batch.slice(0, non_null);
        self.chunks.push(SortedChunk {
            keys: kept_keys,
            output: kept.project(&self.spec.build_output_indices)?,
        });
        self.saw_null_tail = began_null_tail;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<()>) -> unary::Result<bool> {
        if self.publisher {
            unsafe { *self.table.chunks.get() = std::mem::take(&mut self.chunks) };
            // The Release store orders the write above before the flag; the
            // probe gate's Acquire load makes it visible to every probe worker.
            self.build_ready.store(true, Ordering::Release);
            waker_set().notify_all();
        }
        Ok(true)
    }
}
