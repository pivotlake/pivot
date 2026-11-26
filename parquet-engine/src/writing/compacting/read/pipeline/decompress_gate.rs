//! Lets a fetched page through to be decompressed once the compacter
//! releases its rows, and lets go of it until then.
//!
//! A column chunk is fetched whole, but the merge only needs the next rows
//! of each input row group. A page whose rows are not released does not go
//! on: decompressed, it is several times larger than compressed. Nor is it
//! held here: when many files are merged together, the pages that wait for
//! their rows would be most of what the compaction holds. The gate lets go
//! of the page's bytes and keeps where the page lies in its file. The cache
//! the bytes are in keeps them while nothing else needs their memory, and
//! evicts them when something does.
//!
//! When a row of such a page is released (see [`GatedReads`]), the gate
//! asks the fetchers for the page again, by the stretch of its chunk it
//! lies in. The page then arrives here released, from the cache or from the
//! file, and goes on.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use dispatch::{BuildContext, Built, BuiltUnary, Sender, Unary, UnaryFactory, UnaryResult};
use dispatch::{OperatorIO, SharedWorkQueueSender, WorkStatus};

use super::gated_reads::GatedReads;
use crate::CompressedPage;
use crate::types::metadata::{PageStretch, QueryRowGroupMetadata};
use crate::writing::compacting::memory_budget::MemoryBudget;

/// A page whose rows are not released, and whose bytes the gate let go of:
/// what asks for it again. Pages are ordered so that the one whose first
/// row comes first in its row group is the greatest: it is the next one to
/// be released.
struct WaitingPage {
    /// The read the page is of.
    row_group: QueryRowGroupMetadata,
    /// The stretch of its chunk the page is.
    page_stretch: PageStretch,
}

impl PartialEq for WaitingPage {
    fn eq(&self, other: &Self) -> bool {
        self.page_stretch.first_row == other.page_stretch.first_row
    }
}

impl Eq for WaitingPage {}

impl PartialOrd for WaitingPage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WaitingPage {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .page_stretch
            .first_row
            .cmp(&self.page_stretch.first_row)
    }
}

/// Passes a page on to be decompressed if one of its rows is released.
/// Otherwise it lets go of the page, and asks for it again once one is.
///
/// Every worker has one, which takes the pages that reached that worker and
/// passes them on to that worker's queue of pages to decompress.
pub(crate) struct DecompressGate {
    gated_reads: Arc<GatedReads>,
    memory_budget: Arc<MemoryBudget>,
    /// Puts a request for a page to read again into the queue every
    /// worker's fetcher takes from.
    page_request_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
    /// The worker the gate runs on.
    worker: usize,
    /// The pages that wait for their rows to be released, by the number of
    /// their read.
    waiting_pages: HashMap<usize, BinaryHeap<WaitingPage>>,
}

impl DecompressGate {
    /// One gate per worker, each with its sender of `page_request_senders`
    /// into the queue the fetchers take their requests from.
    pub(crate) fn create_for_workers(
        gated_reads: &Arc<GatedReads>,
        memory_budget: &Arc<MemoryBudget>,
        page_request_senders: Vec<SharedWorkQueueSender<QueryRowGroupMetadata>>,
    ) -> Vec<Self> {
        page_request_senders
            .into_iter()
            .enumerate()
            .map(|(worker, page_request_sender)| Self {
                gated_reads: gated_reads.clone(),
                memory_budget: memory_budget.clone(),
                page_request_sender,
                worker,
                waiting_pages: HashMap::new(),
            })
            .collect()
    }

    /// Send `page` on to be decompressed. The memory budget counts it as
    /// decompressed from here on, in place of its compressed size: the
    /// [`PageMemoryCounter`](super::page_memory_counter::PageMemoryCounter)
    /// takes that count over, and keeps it for as long as the page's bytes
    /// are held. The compacter is told that the page no longer waits, which
    /// makes room for another chunk to be fetched ahead, unless it was told
    /// when the page was asked for again.
    fn pass_page_on(
        &self,
        page: CompressedPage,
        sender: &mut dyn Sender<CompressedPage>,
    ) -> UnaryResult<()> {
        self.memory_budget
            .add(page.header.uncompressed_page_size as usize);
        self.memory_budget.subtract(page.span);
        if page.row_group.page_stretch().is_none() {
            self.gated_reads.record_page_let_through(page.span);
        }
        sender.send(page)?;
        Ok(())
    }

    /// Let go of the bytes of `page`, whose rows are not released, and keep
    /// what asks for the page again. Its compressed size is no longer
    /// counted as held.
    fn let_go_of_page(&mut self, read: usize, page: CompressedPage) {
        self.memory_budget.subtract(page.span);
        let page_stretch = PageStretch {
            leaf: page.column_idx,
            file_offset: page.file_offset,
            len: page.span,
            first_page_idx: page.page_idx,
            first_row: page.first_row,
            worker: page.worker_id,
        };
        self.waiting_pages
            .entry(read)
            .or_default()
            .push(WaitingPage {
                row_group: page.row_group,
                page_stretch,
            });
    }

    /// Ask the fetchers for `page` again, now that a row of it is released.
    /// Its compressed size is counted as held again, as a chunk's is when
    /// it is requested, and the compacter is told that the page no longer
    /// waits.
    fn ask_for_page_again(&mut self, page: WaitingPage) -> UnaryResult<()> {
        self.memory_budget.add(page.page_stretch.len);
        self.gated_reads
            .record_page_let_through(page.page_stretch.len);
        let page_request = page.row_group.ask_for_page_stretch(page.page_stretch);
        self.page_request_sender.send(page_request)?;
        Ok(())
    }

    /// Ask again for the waiting pages of `read` whose first row is
    /// released. Returns whether any page was asked for.
    fn ask_for_released_pages_of(&mut self, read: usize) -> UnaryResult<bool> {
        let Some(pages) = self.waiting_pages.get_mut(&read) else {
            return Ok(false);
        };
        let gated_reads = &self.gated_reads;
        let is_released =
            |page: &WaitingPage| gated_reads.is_row_released(read, page.page_stretch.first_row);
        let mut released_pages = Vec::new();
        while pages.peek().is_some_and(is_released) {
            released_pages.extend(pages.pop());
        }
        if pages.is_empty() {
            self.waiting_pages.remove(&read);
        }
        let has_released_pages = !released_pages.is_empty();
        for page in released_pages {
            self.ask_for_page_again(page)?;
        }
        Ok(has_released_pages)
    }

    /// If the compacter has released rows since the gate last looked at the
    /// pages that wait, ask again for those whose first row is now
    /// released. Returns whether any page was asked for.
    ///
    /// The compacter wakes the waiting workers after it releases rows, which
    /// takes their registrations away. So a gate whose worker is still
    /// registered has nothing new to find. One that is not registers again
    /// before it looks, so that a release it does not see yet wakes it.
    fn ask_for_newly_released_pages(&mut self) -> UnaryResult<bool> {
        let waiting_gates = &self.gated_reads.waiting_decompress_gates;
        if self.waiting_pages.is_empty() || waiting_gates.is_registered(self.worker) {
            return Ok(false);
        }
        waiting_gates.register(self.worker);
        let reads: Vec<usize> = self.waiting_pages.keys().copied().collect();
        let mut has_released_pages = false;
        for read in reads {
            has_released_pages |= self.ask_for_released_pages_of(read)?;
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
    /// goes straight on, and one that is not is let go of. A dictionary
    /// page counts as starting at row 0, so it goes on with the first rows
    /// of its read. A page the decompressed cache held arrives
    /// decompressed, and is treated the same.
    ///
    /// The row of a page that is let go of may be released just then. While
    /// the worker is registered as waiting, no row was released since the
    /// gate last looked at the pages that wait, so only this page's read
    /// can have one to ask for. Otherwise the gate registers and looks at
    /// every page that waits: the compacter may have released rows of any
    /// of them.
    fn consume(
        &mut self,
        page: CompressedPage,
        sender: &mut dyn Sender<CompressedPage>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let read = self.gated_reads.find_read_index(&page.row_group);
        if self.gated_reads.is_row_released(read, page.first_row) {
            return self.pass_page_on(page, sender);
        }
        self.let_go_of_page(read, page);
        let waiting_gates = &self.gated_reads.waiting_decompress_gates;
        if waiting_gates.is_registered(self.worker) {
            self.ask_for_released_pages_of(read)?;
        } else {
            self.ask_for_newly_released_pages()?;
        }
        Ok(())
    }

    /// Ask again for the waiting pages whose rows have been released since.
    fn run(
        &mut self,
        _sender: &mut dyn Sender<CompressedPage>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        Ok(if self.ask_for_newly_released_pages()? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Pages that wait are still to be asked for and to go on.
    fn has_pending_work(&self) -> bool {
        !self.waiting_pages.is_empty()
    }
}
