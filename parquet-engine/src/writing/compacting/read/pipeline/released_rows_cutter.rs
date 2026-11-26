//! Cuts the released rows of a read out of its decompressed pages, for the
//! decoders.
//!
//! A scan cuts each row group into ranges of a fixed number of rows as its
//! pages arrive. A compaction knows which rows of each input row group its
//! output takes next, and decoded rows are several times larger than the
//! pages they are decoded from. So a range is cut only of rows the
//! compacter has released (see [`GatedReads`]): from where the last range
//! ended to where the released rows end.
//!
//! Between two releases of a read, the cutter does not hold the pages that
//! still have rows to cut. When many files are merged together the next
//! release of a read is long in coming, and the pages held for all of them
//! would be most of what the compaction holds. The cutter keeps the key
//! under which the decompressed cache holds each page, and takes the page
//! back from the cache when rows of it are released. So the cache decides
//! what happens to the pages in between: it keeps them while nothing else
//! needs their memory, and evicts them when something does. A page the
//! cache evicted is asked for again from the fetchers, by the stretch of
//! its chunk it lies in, and arrives here like any page.
//!
//! What the compaction is finished with does not stay in the caches
//! either: its input files are deleted when it is done, and bytes left at
//! rest would take the place of bytes some query still wants. When the
//! last row of a page is cut, the cutter drops the page from the
//! decompressed cache, and its chunk's bytes up to the page's end from the
//! compressed cache. That cache keeps bytes in stretches that do not end
//! where pages and chunks do, so what such a drop leaves goes when the
//! last column of the row group is cut through.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use bytes::Bytes;

use dispatch::memory::memory_ctx;
use dispatch::{BuildContext, Built, BuiltUnary, Projection, Sender, Unary, UnaryFactory};
use dispatch::{OperatorIO, SharedWorkQueueSender, UnaryResult, WorkStatus};

use super::gated_reads::GatedReads;
use super::page_memory_counter::CountedPageBytes;
use crate::reading::RangeCutter;
use crate::types::metadata::{QueryRowGroupMetadata, ReadKey};
use crate::types::page::DecompressedPageType;
use crate::writing::compacting::memory_budget::MemoryBudget;
use crate::{DecodeRange, DecompressedPage, WorkerAllocator};

/// A read that has rows still to cut.
struct UncutRead {
    read_key: ReadKey,
    /// Every row of the read's row group before this one was cut.
    cut_rows_end: u32,
}

/// Builds the [`ReleasedRowsCutter`] of one worker.
pub(crate) struct ReleasedRowsCutterFactory {
    gated_reads: Arc<GatedReads>,
    memory_budget: Arc<MemoryBudget>,
    /// Puts a request into the queue every worker's fetcher takes from.
    page_request_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
    /// The worker the cutter runs on.
    worker: usize,
    /// The worker's allocator for the dictionaries built of its reads.
    allocator: Arc<WorkerAllocator>,
}

impl ReleasedRowsCutterFactory {
    /// One factory per worker, each with the worker's allocator of
    /// `allocators` and its sender of `page_request_senders` into the
    /// queue the fetchers take their requests from.
    pub(crate) fn create_for_workers(
        gated_reads: &Arc<GatedReads>,
        memory_budget: &Arc<MemoryBudget>,
        allocators: Vec<Arc<WorkerAllocator>>,
        page_request_senders: Vec<SharedWorkQueueSender<QueryRowGroupMetadata>>,
    ) -> Vec<Self> {
        allocators
            .into_iter()
            .zip(page_request_senders)
            .enumerate()
            .map(|(worker, (allocator, page_request_sender))| Self {
                gated_reads: gated_reads.clone(),
                memory_budget: memory_budget.clone(),
                page_request_sender,
                worker,
                allocator,
            })
            .collect()
    }
}

impl UnaryFactory<DecompressedPage, DecodeRange> for ReleasedRowsCutterFactory {
    type Unary = ReleasedRowsCutter;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(ReleasedRowsCutter {
            gated_reads: self.gated_reads,
            memory_budget: self.memory_budget,
            page_request_sender: self.page_request_sender,
            worker: self.worker,
            // Every read names its own column and nothing prunes a row
            // group, so the columns, the predicates and the count of
            // claimed row groups that a scan gives its cutter are never
            // used.
            collected_pages: RangeCutter::new(
                Projection::columns([]),
                Arc::new(Vec::new()),
                Arc::new(AtomicUsize::new(0)),
                self.allocator.clone(),
            ),
            dictionary_allocator: self.allocator,
            uncut_reads: HashMap::new(),
        })
    }
}

/// Cuts a range of rows for the decoders once the rows are released and
/// their pages have arrived.
///
/// Every worker has one, for the reads that worker fetched.
pub(crate) struct ReleasedRowsCutter {
    gated_reads: Arc<GatedReads>,
    memory_budget: Arc<MemoryBudget>,
    /// Puts a request for pages to read again into the queue every
    /// worker's fetcher takes from.
    page_request_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
    /// The worker the cutter runs on.
    worker: usize,
    /// Keeps the pages of each read as they arrive and builds the reads'
    /// dictionaries. The ranges it would cut itself, of a fixed number of
    /// rows, are not used.
    collected_pages: RangeCutter,
    /// The worker's allocator for dictionaries, which the dictionary pages
    /// are copied into.
    dictionary_allocator: Arc<WorkerAllocator>,
    /// The reads that have rows still to cut, by the number of the read.
    uncut_reads: HashMap<usize, UncutRead>,
}

impl ReleasedRowsCutter {
    /// Copy the bytes of a dictionary page, `buffers`, out of the
    /// decompressed cache into the worker's memory for dictionaries, and
    /// count the copy in the memory budget for as long as it is held.
    ///
    /// A string dictionary is built as views into its page, and lives as
    /// long as its read. In the cache, the page lies in a block with the
    /// pages of other reads, and would keep the whole block from being
    /// evicted long after those pages were let go of. The caller drops
    /// the page from the cache: nothing reads it there again.
    fn copy_dictionary_page_out_of_cache(&self, buffers: &mut [Bytes]) {
        for buffer in buffers.iter_mut().filter(|buffer| !buffer.is_empty()) {
            let mut slab = self
                .dictionary_allocator
                .with(|allocator| allocator.get_slab_of_size(buffer.len(), false));
            slab.as_mut_slice().copy_from_slice(buffer);
            let memory = self.memory_budget.reserve(buffer.len());
            *buffer = CountedPageBytes::wrap(Bytes::from_owner(slab.freeze()), memory);
        }
    }

    /// Drop from the compressed cache the bytes of the file of the row
    /// group of `metadata` up to the row group's end. Every column of the
    /// row group is cut through, and so is every row group before it:
    /// a file's rows are read in file order.
    fn drop_row_group_from_compressed_cache(metadata: &QueryRowGroupMetadata) {
        let row_group = metadata.get_metadata();
        let chunks = row_group.columns.iter().filter(|chunk| !chunk.absent);
        let chunk_ends = chunks.map(|chunk| {
            let chunk_start = chunk
                .dictionary_page_offset
                .unwrap_or(chunk.data_page_offset);
            (chunk_start + chunk.total_compressed_size) as usize
        });
        if let Some(row_group_end) = chunk_ends.max() {
            let compressed_cache = &memory_ctx().page_caches().compressed;
            compressed_cache.drop_file_range(&row_group.open_file, 0, row_group_end);
        }
    }

    /// Cut the next ranges of `read`, which has rows still to cut, of the
    /// rows that are released and whose pages have arrived. Returns
    /// whether any range was cut.
    ///
    /// A range is of the released rows that are not cut yet, and of no
    /// more rows than a range of the read may hold (see
    /// [`GatedReads::find_range_rows`]). It is cut once the pages of all
    /// its rows have arrived. So a long release is decoded as its pages
    /// arrive. And the rows of a sort column, which are released a range
    /// at a time, are cut at the same rows in every sort column of a
    /// file, in whatever order their pages and releases arrive: the merge
    /// compares the keys of the same rows.
    ///
    /// The pages whose bytes were let go of after the last cut are taken
    /// back from the decompressed cache first, and counted in the memory
    /// budget while something holds them. A page the cache holds no more
    /// is asked for again, and nothing is cut until it has arrived: its
    /// compressed size is counted as held from here on, as a chunk's is
    /// when it is requested. Once every released row is cut,
    /// the bytes of the pages that still have rows to cut are let go of
    /// again.
    ///
    /// A page whose last row is cut is dropped from the decompressed
    /// cache, and its chunk's bytes up to its end from the compressed
    /// cache: the chunk's pages are cut through in file order. And when
    /// this was the last read of its row group to be cut through, all of
    /// the row group's bytes are dropped from the compressed cache.
    fn cut_released_rows(
        &mut self,
        read: usize,
        sender: &mut dyn Sender<DecodeRange>,
    ) -> UnaryResult<bool> {
        let uncut = self
            .uncut_reads
            .get_mut(&read)
            .expect("rows are cut of a read that has rows still to cut");
        let released_rows_end = self.gated_reads.find_released_rows_end(read);
        if released_rows_end <= uncut.cut_rows_end {
            return Ok(false);
        }
        let pages = self.collected_pages.pages_of(uncut.read_key);
        let ready_rows_end = pages.count_ready_rows();
        let range_rows = self.gated_reads.find_range_rows(read);
        let first_range_end = released_rows_end.min(uncut.cut_rows_end + range_rows);
        if ready_rows_end < first_range_end {
            return Ok(false);
        }
        let memory_budget = &self.memory_budget;
        let is_every_page_back = pages.take_back_page_bytes(|cache_key| {
            let page = memory_ctx().page_caches().decompressed.get(cache_key)?;
            let count_buffer = |buffer: Bytes| {
                let memory = memory_budget.reserve(buffer.len());
                CountedPageBytes::wrap(buffer, memory)
            };
            Some(page.data.into_iter().map(count_buffer).collect())
        });
        if !is_every_page_back {
            for page_stretch in pages.list_stretches_to_ask_for(self.worker) {
                self.memory_budget.add(page_stretch.len);
                let page_request = pages.metadata().ask_for_page_stretch(page_stretch);
                self.page_request_sender.send(page_request)?;
            }
            return Ok(false);
        }
        let mut has_cut = false;
        let mut finished_pages = Vec::new();
        while uncut.cut_rows_end < released_rows_end {
            let rows_end = released_rows_end.min(uncut.cut_rows_end + range_rows);
            if ready_rows_end < rows_end {
                break;
            }
            let rows = uncut.cut_rows_end..rows_end;
            sender.send(pages.cut_rows(rows, Some(&mut finished_pages)))?;
            uncut.cut_rows_end = rows_end;
            has_cut = true;
        }
        let page_caches = memory_ctx().page_caches();
        for (column, cache_key) in &finished_pages {
            page_caches.decompressed.drop_entry(cache_key);
            page_caches.compressed.drop_file_range(
                &cache_key.file,
                pages.find_chunk_start(*column),
                cache_key.offset + cache_key.span,
            );
        }
        if uncut.cut_rows_end == pages.metadata().num_rows() as u32 {
            let read_key = uncut.read_key;
            let metadata = pages.metadata().clone();
            self.collected_pages.close(read_key);
            self.uncut_reads.remove(&read);
            if self.gated_reads.record_read_cut(read) {
                Self::drop_row_group_from_compressed_cache(&metadata);
            }
        } else if uncut.cut_rows_end == released_rows_end {
            pages.let_go_of_page_bytes();
        }
        Ok(has_cut)
    }

    /// If the compacter has released rows since the cutter last looked at
    /// its reads, cut the released rows of each of them. Returns whether
    /// any range was cut.
    ///
    /// The compacter wakes the waiting workers after it releases rows,
    /// which takes their registrations away. So a cutter whose worker is
    /// still registered has nothing new to find. One that is not registers
    /// again before it looks, so that a release it does not see yet wakes
    /// it.
    fn cut_newly_released_rows(
        &mut self,
        sender: &mut dyn Sender<DecodeRange>,
    ) -> UnaryResult<bool> {
        let waiting_cutters = &self.gated_reads.waiting_cutters;
        if self.uncut_reads.is_empty() || waiting_cutters.is_registered(self.worker) {
            return Ok(false);
        }
        waiting_cutters.register(self.worker);
        let reads: Vec<usize> = self.uncut_reads.keys().copied().collect();
        let mut has_cut = false;
        for read in reads {
            has_cut |= self.cut_released_rows(read, sender)?;
        }
        Ok(has_cut)
    }
}

impl Unary<DecompressedPage, DecodeRange> for ReleasedRowsCutter {
    /// Take a page that was decompressed, and cut the rows it makes ready
    /// if they are released.
    ///
    /// While the worker is registered as waiting, no row was released
    /// since the cutter last looked at its reads, so only the read of this
    /// page can have rows to cut. Otherwise the cutter registers and looks
    /// at every read: the compacter may have released rows of any of them.
    fn consume(
        &mut self,
        mut page: DecompressedPage,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let read = self
            .gated_reads
            .find_read_index(&page.query_row_group_metadata);
        if let DecompressedPageType::Dict { data, .. } = &mut page.data {
            self.copy_dictionary_page_out_of_cache(data);
            if let Some(cache_key) = &page.cache_key {
                let decompressed_cache = &memory_ctx().page_caches().decompressed;
                decompressed_cache.drop_entry(cache_key);
            }
        }
        let Some(read_key) = self.collected_pages.collect_page(page)? else {
            return Ok(());
        };
        self.uncut_reads.entry(read).or_insert(UncutRead {
            read_key,
            cut_rows_end: 0,
        });
        if self.gated_reads.waiting_cutters.is_registered(self.worker) {
            self.cut_released_rows(read, sender)?;
        } else {
            self.cut_newly_released_rows(sender)?;
        }
        Ok(())
    }

    /// Cut the rows that have been released since.
    fn run(
        &mut self,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        Ok(if self.cut_newly_released_rows(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// The slots of the dropped pages and row groups that were still read
    /// when they were last looked at are looked at once more.
    fn finish(&mut self, _sender: &mut dyn Sender<DecodeRange>) -> UnaryResult<bool> {
        let page_caches = memory_ctx().page_caches();
        page_caches.decompressed.recycle_emptied_slots();
        page_caches.compressed.recycle_emptied_slots();
        Ok(!self.collected_pages.has_open_reads())
    }

    /// Rows that are not cut are still to go on.
    fn has_pending_work(&self) -> bool {
        !self.uncut_reads.is_empty()
    }
}
