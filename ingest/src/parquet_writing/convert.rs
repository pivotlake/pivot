//! First stage of a flush: flatten each buffered item ([`ToRecordBatch`]) into
//! its Arrow `RecordBatch`, on whatever worker steals it. Conversion is the
//! CPU-heavy half of ingest (id encoding, attribute JSON, column building), so
//! it runs here — inside the dataflow — never on the receive path.
//!
//! Compaction skips this stage entirely: its input is the scan dataflow, which
//! already emits `RecordBatch`es straight into the
//! [`builder`](super::builder).

use std::marker::PhantomData;

use arrow_array::RecordBatch;
use dispatch::{Sender, Unary, UnaryFactory, UnaryResult};

use super::ToRecordBatch;

/// Per-worker factory for [`Convert`].
pub(super) struct ConvertFactory<T>(PhantomData<fn() -> T>);

/// One [`ConvertFactory`] per worker.
pub(super) fn factories<T: ToRecordBatch>(worker_count: usize) -> Vec<ConvertFactory<T>> {
    (0..worker_count)
        .map(|_| ConvertFactory(PhantomData))
        .collect()
}

impl<T: ToRecordBatch> UnaryFactory<T, RecordBatch> for ConvertFactory<T> {
    type Unary = Convert<T>;

    fn build_unary(self) -> Convert<T> {
        Convert(PhantomData)
    }
}

/// Stateless 1→1 conversion; row-less items are dropped.
pub(super) struct Convert<T>(PhantomData<fn() -> T>);

impl<T: ToRecordBatch> Unary<T, RecordBatch> for Convert<T> {
    fn consume<S: Sender<RecordBatch>>(&mut self, item: T, sender: &mut S) -> UnaryResult<()> {
        if let Some(batch) = item.to_record_batch()?
            && batch.num_rows() > 0
        {
            sender.send(batch)?;
        }
        Ok(())
    }
}
