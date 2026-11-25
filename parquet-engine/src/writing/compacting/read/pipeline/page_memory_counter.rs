//! Makes the memory budget count each decompressed page for exactly as long
//! as something holds its bytes.
//!
//! A decompressed page is passed on as shared byte buffers. A fixed-width
//! column's decoder copies the values out and lets the page go. A string
//! column's decoder builds arrays that point into the page, so the page
//! lives as long as the last decoded batch that points into it, wherever
//! that batch was sent. Nobody can tell from the outside which it is.
//!
//! So the [`PageMemoryCounter`] puts the count inside the bytes themselves:
//! it wraps each buffer of a page with a [`MemoryReservation`], which is
//! dropped, and gives the bytes back, when the last reference to the
//! buffer is dropped.

use std::sync::Arc;

use bytes::Bytes;
use dispatch::UnaryResult;
use dispatch::{BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory};

use crate::DecompressedPage;
use crate::types::page::{DataPage, DecompressedPageType};
use crate::writing::compacting::memory_budget::{MemoryBudget, MemoryReservation};

/// One buffer of a decompressed page, with the reservation that counts it.
struct CountedPageBytes {
    bytes: Bytes,
    _memory: MemoryReservation,
}

impl AsRef<[u8]> for CountedPageBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Wraps the bytes of every decompressed page it passes on so that the
/// memory budget counts them until nothing holds them. It neither reads
/// nor changes the bytes.
pub(crate) struct PageMemoryCounter {
    memory_budget: Arc<MemoryBudget>,
}

impl PageMemoryCounter {
    /// One counter per worker.
    pub(crate) fn create_for_workers(
        memory_budget: &Arc<MemoryBudget>,
        worker_count: usize,
    ) -> Vec<Self> {
        (0..worker_count)
            .map(|_| Self {
                memory_budget: memory_budget.clone(),
            })
            .collect()
    }

    /// Wrap each of `buffers`, the bytes of one page, with a reservation
    /// for its size. The `DecompressGate` counted the page's decompressed
    /// size as held when it let the page through, so the reservations take
    /// that count over and add nothing.
    fn count_page_buffers(&self, buffers: &mut [Bytes]) {
        for buffer in buffers {
            let memory = self.memory_budget.reserve_counted_bytes(buffer.len());
            *buffer = Bytes::from_owner(CountedPageBytes {
                bytes: std::mem::take(buffer),
                _memory: memory,
            });
        }
    }
}

impl UnaryFactory<DecompressedPage, DecompressedPage> for PageMemoryCounter {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<DecompressedPage, DecompressedPage> for PageMemoryCounter {
    fn consume(
        &mut self,
        mut page: DecompressedPage,
        sender: &mut dyn Sender<DecompressedPage>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        match &mut page.data {
            DecompressedPageType::Dict { data, .. }
            | DecompressedPageType::Data(DataPage { data, .. }) => self.count_page_buffers(data),
            DecompressedPageType::SkippedData { .. } => {}
        }
        sender.send(page)?;
        Ok(())
    }
}
