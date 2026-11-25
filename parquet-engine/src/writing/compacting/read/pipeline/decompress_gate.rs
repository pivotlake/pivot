//! Holds a fetched page back, compressed, until the compacter releases its
//! rows.
//!
//! A column chunk is fetched whole, but the merge only needs the next rows
//! of each input row group. Decompressed, a page is several times larger
//! than compressed. So the pages of a chunk wait here, and a page goes on
//! to be decompressed only when one of its rows is released (see
//! [`GatedReads`]).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use dispatch::{BuildContext, Built, BuiltUnary, Sender, Unary, UnaryFactory, UnaryResult};
use dispatch::{OperatorIO, WorkStatus};

use super::gated_reads::GatedReads;
use crate::CompressedPage;
use crate::writing::compacting::memory_budget::MemoryBudget;

/// A page that waits at the gate. Pages are ordered so that the one whose
/// first row comes first in its row group is the greatest: it is the next
/// one to be released.
struct HeldPage(CompressedPage);

impl PartialEq for HeldPage {
    fn eq(&self, other: &Self) -> bool {
        self.0.first_row == other.0.first_row
    }
}

impl Eq for HeldPage {}

impl PartialOrd for HeldPage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeldPage {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.first_row.cmp(&self.0.first_row)
    }
}

/// Passes a compressed page on to be decompressed once one of its rows is
/// released, and holds it until then.
///
/// Every worker has one, which holds the pages that reached that worker and
/// passes them on to that worker's queue of pages to decompress.
pub(crate) struct DecompressGate {
    gated_reads: Arc<GatedReads>,
    memory_budget: Arc<MemoryBudget>,
    /// The worker the gate runs on.
    worker: usize,
    /// The pages that wait, by the number of their read.
    held_pages: HashMap<usize, BinaryHeap<HeldPage>>,
}

impl DecompressGate {
    /// One gate per worker.
    pub(crate) fn create_for_workers(
        gated_reads: &Arc<GatedReads>,
        memory_budget: &Arc<MemoryBudget>,
        worker_count: usize,
    ) -> Vec<Self> {
        (0..worker_count)
            .map(|worker| Self {
                gated_reads: gated_reads.clone(),
                memory_budget: memory_budget.clone(),
                worker,
                held_pages: HashMap::new(),
            })
            .collect()
    }

    /// Count `page` as held decompressed rather than compressed, and tell
    /// the compacter that it no longer waits compressed. The
    /// [`PageMemoryCounter`](super::page_memory_counter::PageMemoryCounter)
    /// takes the decompressed count over once the page is decompressed, and
    /// keeps it for as long as the page's bytes are held.
    fn count_page_as_decompressed(&self, page: &CompressedPage) {
        self.memory_budget
            .add(page.header.uncompressed_page_size as usize);
        self.memory_budget.subtract(page.span);
        self.gated_reads.record_page_let_through(page.span);
    }

    /// Send `page` on to be decompressed. It is counted as decompressed
    /// from here on, unless it arrived decompressed and was counted so
    /// then.
    fn pass_page_on(
        &self,
        page: CompressedPage,
        sender: &mut dyn Sender<CompressedPage>,
    ) -> UnaryResult<()> {
        if page.decompressed.is_none() {
            self.count_page_as_decompressed(&page);
        }
        sender.send(page)?;
        Ok(())
    }

    /// If the compacter has released rows since the gate last looked at the
    /// pages it holds, send on those whose first row is now released.
    /// Returns whether any page was sent on.
    ///
    /// The compacter wakes the waiting workers after it releases rows, which
    /// takes their registrations away. So a gate whose worker is still
    /// registered has nothing new to find. One that is not registers again
    /// before it looks, so that a release it does not see yet wakes it.
    fn pass_newly_released_pages_on(
        &mut self,
        sender: &mut dyn Sender<CompressedPage>,
    ) -> UnaryResult<bool> {
        let gated_reads = &self.gated_reads;
        let waiting_gates = &gated_reads.waiting_decompress_gates;
        if self.held_pages.is_empty() || waiting_gates.is_registered(self.worker) {
            return Ok(false);
        }
        waiting_gates.register(self.worker);
        let mut released_pages = Vec::new();
        self.held_pages.retain(|&read, pages| {
            while pages
                .peek()
                .is_some_and(|page| gated_reads.is_row_released(read, page.0.first_row))
            {
                released_pages.extend(pages.pop().map(|page| page.0));
            }
            !pages.is_empty()
        });
        let has_released_pages = !released_pages.is_empty();
        for page in released_pages {
            self.pass_page_on(page, sender)?;
        }
        Ok(has_released_pages)
    }
}

impl UnaryFactory<CompressedPage, CompressedPage> for DecompressGate {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<CompressedPage, CompressedPage> for DecompressGate {
    /// Take a page that was fetched. A page whose first row is released
    /// goes straight on. A dictionary page counts as starting at row 0, so
    /// it goes on with the first rows of its read.
    ///
    /// A page that is not released is held. Before that the gate registers
    /// its worker as waiting and looks once more: the compacter may have
    /// released the row and woken the waiting workers in between.
    fn consume(
        &mut self,
        page: CompressedPage,
        sender: &mut dyn Sender<CompressedPage>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        // A page that the decompressed cache held arrives decompressed.
        if page.decompressed.is_some() {
            self.count_page_as_decompressed(&page);
        }
        self.pass_newly_released_pages_on(sender)?;
        let read = self.gated_reads.find_read_index(&page.row_group);
        let is_released = self.gated_reads.is_row_released(read, page.first_row) || {
            let waiting_gates = &self.gated_reads.waiting_decompress_gates;
            waiting_gates.register(self.worker);
            self.gated_reads.is_row_released(read, page.first_row)
        };
        if is_released {
            self.pass_page_on(page, sender)
        } else {
            self.held_pages
                .entry(read)
                .or_default()
                .push(HeldPage(page));
            Ok(())
        }
    }

    /// Send on the held pages whose rows have been released since.
    fn run(
        &mut self,
        sender: &mut dyn Sender<CompressedPage>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        Ok(if self.pass_newly_released_pages_on(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Held pages are still to go on.
    fn has_pending_work(&self) -> bool {
        !self.held_pages.is_empty()
    }
}
