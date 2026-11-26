//! The decode ranges of a row group, and the pages they are cut from.
//!
//! The worker that claimed a row group collects its decompressed pages in a
//! [`RowGroupPages`] as they come back from the decompressor, and cuts a
//! [`DecodeRange`] job as soon as the pages covering the next range have
//! arrived. A job carries the pages it decodes from, shared with the
//! neighbouring range where a page straddles the two, so any worker can
//! decode it without reaching back into the collection. A page lives as
//! long as a job holds it.

use crate::reading::decoding::DecodePlan;
use crate::reading::decoding::leaf_decoders::SharedDictionary;
use crate::thrift::general::Encoding;
use crate::thrift::headers::DataPageHeader;
use crate::types::metadata::{PageStretch, QueryRowGroupMetadata, RowSelection};
use crate::types::page::DataPage;
use bytes::Bytes;
use dispatch::WorkerIdOutput;
use dispatch::memory::CacheKey;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Rows per decode range of a scan. A range is the unit another worker can take over,
/// and a worker taking one over skips into its first page of every column,
/// so a range must be large enough for that skip to stay a small fraction
/// of its decode.
pub const DECODE_RANGE_ROWS: u32 = 16 * 1024;

/// A data page's payload, or only its header when every row of the page was
/// filtered out.
pub enum PageContent {
    Data(DataPage),
    Skipped(DataPageHeader),
}

impl PageContent {
    /// Whether the page's values are indices into a dictionary, so decoding
    /// it needs the dictionary.
    pub fn is_dictionary_encoded(&self) -> bool {
        match self {
            PageContent::Data(page) => matches!(
                page.header.encoding,
                Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
            ),
            PageContent::Skipped(_) => false,
        }
    }
}

/// A decompressed data page.
pub struct StoredPage {
    /// The page's index within its column chunk.
    pub idx: usize,
    /// The row group row of the page's first value.
    pub first_row: u32,
    /// The page's values, which are its row group rows.
    pub rows: u32,
    /// The key under which the decompressed cache holds the page's bytes,
    /// if it does.
    pub cache_key: Option<CacheKey>,
    pub content: PageContent,
}

/// A data page whose bytes the collection let go of: what the page is, and
/// the key that finds its bytes in the decompressed cache again.
struct LetGoPage {
    idx: usize,
    first_row: u32,
    rows: u32,
    header: DataPageHeader,
    cache_key: CacheKey,
    /// Whether the page was asked for again, after the decompressed cache
    /// was found to hold its bytes no more.
    is_asked_for_again: bool,
}

impl StoredPage {
    /// The row group rows the page spans.
    pub fn row_span(&self) -> Range<u32> {
        self.first_row..self.first_row + self.rows
    }

    fn overlaps(&self, rows: &Range<u32>) -> bool {
        self.first_row < rows.end && rows.start < self.first_row + self.rows
    }
}

/// What a range decodes one leaf from.
pub struct RangeColumn {
    /// The leaf's dictionary, built once by the cutter, when its chunk has
    /// one.
    pub dictionary: Option<SharedDictionary>,
    /// The pages holding the range's rows, in page order.
    pub pages: Vec<Arc<StoredPage>>,
}

/// A range of a row group's rows to decode, with everything the decode
/// needs. The job is one pointer wide: it moves through a work-stealing
/// channel and every steal copies it.
pub struct DecodeRange(Box<RangeContents>);

struct RangeContents {
    metadata: QueryRowGroupMetadata,
    plan: Arc<DecodePlan>,
    /// The rows to decode, counted the way the decoder emits them: row group
    /// rows for a whole row group, surviving rows for an index selection.
    rows: Range<u32>,
    /// One entry per decoded leaf, in decode order.
    columns: Vec<RangeColumn>,
    /// The worker that cut the range, which holds the row group's pages.
    cut_on_worker: usize,
    /// The row group's rows still to decode, shared by all its ranges.
    rows_left: Arc<AtomicUsize>,
    /// The scan's count of row groups claimed but not yet decoded, released
    /// with the row group's last rows.
    outstanding_row_groups: Arc<AtomicUsize>,
}

impl DecodeRange {
    pub fn metadata(&self) -> &QueryRowGroupMetadata {
        &self.0.metadata
    }

    pub(crate) fn plan(&self) -> &Arc<DecodePlan> {
        &self.0.plan
    }

    pub fn rows(&self) -> Range<u32> {
        self.0.rows.clone()
    }

    pub fn column(&self, leaf: usize) -> &RangeColumn {
        &self.0.columns[leaf]
    }

    /// The counter of the row group's rows still to decode; zero once
    /// every range is decoded, on whichever worker.
    pub fn rows_left(&self) -> &Arc<AtomicUsize> {
        &self.0.rows_left
    }

    /// Records the range as decoded, and the row group with it when these
    /// were its last rows to decode.
    pub fn decoded(self) {
        let rows = self.0.rows.len();
        if self.0.rows_left.fetch_sub(rows, Ordering::AcqRel) == rows {
            self.0
                .outstanding_row_groups
                .fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// A range routed by worker goes to the worker that cut it: a reader whose
/// ranges are short decodes each row group as one stream there, where a
/// worker taking a range over would skip into its first page of every
/// column.
impl WorkerIdOutput for DecodeRange {
    fn worker_id(&self) -> usize {
        self.0.cut_on_worker
    }
}

struct ColumnPages {
    dictionary: Option<SharedDictionary>,
    /// Whether a data page of the column is dictionary encoded. Such a
    /// column's ranges are not ready until its dictionary has arrived,
    /// whatever the footer says about a dictionary page.
    needs_dictionary: bool,
    /// Indexed by page number; `None` until the page arrives, and again
    /// once the last range over it has been cut, or while its bytes are
    /// let go of.
    pages: Vec<Option<Arc<StoredPage>>>,
    /// The pages that arrived, still have rows to cut, and whose bytes
    /// were let go of.
    let_go_pages: Vec<LetGoPage>,
    /// How many pages, from the first, have arrived without a gap, and the
    /// rows they span.
    contiguous_pages: usize,
    contiguous_rows: u32,
}

/// The pages of one row group as they arrive, and the ranges cut from them.
pub(crate) struct RowGroupPages {
    metadata: QueryRowGroupMetadata,
    plan: Arc<DecodePlan>,
    columns: Vec<ColumnPages>,
    /// The decode ranges, in decoded-row terms (see [`DecodeRange::rows`]).
    ranges: Vec<Range<u32>>,
    /// Per range, how many row group rows every column must have contiguous
    /// pages for before the range can be decoded.
    rows_needed: Vec<u32>,
    rows_left: Arc<AtomicUsize>,
    outstanding_row_groups: Arc<AtomicUsize>,
}

impl RowGroupPages {
    /// An empty collection for `metadata`, decoded by `plan`. The row group
    /// counts in `outstanding_row_groups` until its last range is decoded.
    pub(crate) fn new(
        metadata: QueryRowGroupMetadata,
        plan: Arc<DecodePlan>,
        outstanding_row_groups: Arc<AtomicUsize>,
    ) -> Self {
        let row_group_rows = metadata.num_rows().max(0) as u32;
        let (ranges, rows_needed) = match metadata.selection() {
            RowSelection::All => {
                let ranges: Vec<Range<u32>> = (0..row_group_rows)
                    .step_by(DECODE_RANGE_ROWS as usize)
                    .map(|start| start..(start + DECODE_RANGE_ROWS).min(row_group_rows))
                    .collect();
                let rows_needed = ranges.iter().map(|range| range.end).collect();
                (ranges, rows_needed)
            }
            // The surviving rows of a row group are decoded as one range:
            // their masks address pages by row group position, so the whole
            // row group has to be in place.
            RowSelection::Indices(indices) if indices.is_empty() => (Vec::new(), Vec::new()),
            RowSelection::Indices(indices) => {
                let every_surviving_row = 0..indices.len() as u32;
                (vec![every_surviving_row], vec![row_group_rows])
            }
        };
        let columns = (0..plan.leaf_count())
            .map(|_| ColumnPages {
                dictionary: None,
                needs_dictionary: false,
                pages: Vec::new(),
                let_go_pages: Vec::new(),
                contiguous_pages: 0,
                contiguous_rows: 0,
            })
            .collect();
        Self {
            metadata,
            plan,
            columns,
            rows_left: Arc::new(AtomicUsize::new(ranges.iter().map(Range::len).sum())),
            ranges,
            rows_needed,
            outstanding_row_groups,
        }
    }

    pub(crate) fn metadata(&self) -> &QueryRowGroupMetadata {
        &self.metadata
    }

    /// Where the chunk of `column` starts in its file.
    pub(crate) fn find_chunk_start(&self, column: usize) -> usize {
        let chunk = &self.metadata.columns()[self.plan.file_leaves[column]];
        chunk
            .dictionary_page_offset
            .unwrap_or(chunk.data_page_offset) as usize
    }

    pub(crate) fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Keeps the built dictionary of `column`.
    pub(crate) fn insert_dictionary(&mut self, column: usize, dictionary: SharedDictionary) {
        self.columns[column].dictionary = Some(dictionary);
    }

    /// Keeps a data page of `column`, whose bytes the decompressed cache
    /// holds under `cache_key` if it holds them.
    pub(crate) fn insert_page(
        &mut self,
        column: usize,
        idx: usize,
        first_row: u32,
        rows: u32,
        content: PageContent,
        cache_key: Option<CacheKey>,
    ) {
        let pages = &mut self.columns[column];
        if pages.pages.len() <= idx {
            pages.pages.resize_with(idx + 1, || None);
        }
        assert!(pages.pages[idx].is_none(), "a page arrives once");
        // A page that was let go of and asked for again is held again.
        pages.let_go_pages.retain(|page| page.idx != idx);
        if content.is_dictionary_encoded() {
            pages.needs_dictionary = true;
        }
        pages.pages[idx] = Some(Arc::new(StoredPage {
            idx,
            first_row,
            rows,
            cache_key,
            content,
        }));
        while let Some(Some(next)) = pages.pages.get(pages.contiguous_pages) {
            pages.contiguous_rows += next.rows;
            pages.contiguous_pages += 1;
        }
    }

    /// How many row group rows, from the first, every column has the pages
    /// for.
    pub(crate) fn count_ready_rows(&self) -> u32 {
        self.columns
            .iter()
            .map(|pages| {
                if pages.needs_dictionary && pages.dictionary.is_none() {
                    0
                } else {
                    pages.contiguous_rows
                }
            })
            .min()
            .unwrap_or(0)
    }

    /// How many ranges, from the first, every column has the pages for.
    pub(crate) fn ready_ranges(&self) -> usize {
        let rows_ready = self.count_ready_rows();
        self.rows_needed
            .iter()
            .take_while(|&&needed| needed <= rows_ready)
            .count()
    }

    /// Cuts range `index`, which must be ready (see
    /// [`cut_rows`](Self::cut_rows)).
    pub(crate) fn cut_range(&mut self, index: usize) -> DecodeRange {
        self.cut_rows(self.ranges[index].clone(), None)
    }

    /// Cuts a range of the rows `rows`, which every column must have the
    /// pages for, forgetting the pages no later range needs: from here on
    /// only the ranges cut over them keep them alive. Ranges are cut in
    /// row order, each starting where the one before it ended.
    ///
    /// The pages that are forgotten are added to `finished_pages`, each as
    /// the column it is of and the key under which the decompressed cache
    /// holds it, for a reader that drops them from the caches: no range
    /// will be cut of them again.
    pub(crate) fn cut_rows(
        &mut self,
        rows: Range<u32>,
        mut finished_pages: Option<&mut Vec<(usize, CacheKey)>>,
    ) -> DecodeRange {
        let last_range = self.ranges.last().is_some_and(|last| rows.end == last.end);
        let whole_row_group = matches!(self.metadata.selection(), RowSelection::Indices(_));
        let columns = self
            .columns
            .iter_mut()
            .enumerate()
            .map(|(column_idx, column)| {
                let mut pages = Vec::new();
                for slot in column.pages.iter_mut() {
                    let Some(page) = slot else { continue };
                    if !whole_row_group && !page.overlaps(&rows) {
                        continue;
                    }
                    pages.push(page.clone());
                    if last_range || page.row_span().end <= rows.end {
                        if let (Some(finished_pages), Some(cache_key)) =
                            (finished_pages.as_deref_mut(), &page.cache_key)
                        {
                            finished_pages.push((column_idx, cache_key.clone()));
                        }
                        *slot = None;
                    }
                }
                RangeColumn {
                    dictionary: column.dictionary.clone(),
                    pages,
                }
            })
            .collect();
        DecodeRange(Box::new(RangeContents {
            metadata: self.metadata.clone(),
            plan: self.plan.clone(),
            rows,
            columns,
            cut_on_worker: dispatch::worker::WORKER_IDX.get(),
            rows_left: self.rows_left.clone(),
            outstanding_row_groups: self.outstanding_row_groups.clone(),
        }))
    }

    /// Lets go of the bytes of the data pages held that the decompressed
    /// cache holds too, and keeps the key of each. The ranges already cut
    /// over a page keep its bytes for as long as they need them. Until
    /// [`take_back_page_bytes`](Self::take_back_page_bytes) has brought the
    /// pages back, no range is cut.
    ///
    /// A reader does this when the rows it cuts next are long in being
    /// asked for: the cache may then give the pages' memory to something
    /// else, and keeps them for as long as nothing else needs it.
    pub(crate) fn let_go_of_page_bytes(&mut self) {
        for column in &mut self.columns {
            let ColumnPages {
                pages,
                let_go_pages,
                ..
            } = column;
            for slot in pages {
                let Some(page) = slot.as_deref() else {
                    continue;
                };
                let (PageContent::Data(data_page), Some(cache_key)) =
                    (&page.content, &page.cache_key)
                else {
                    continue;
                };
                // A page's mask belongs to the rows of one query's range,
                // and is not kept apart from its bytes.
                if data_page.filter_mask.is_some() {
                    continue;
                }
                let_go_pages.push(LetGoPage {
                    idx: page.idx,
                    first_row: page.first_row,
                    rows: page.rows,
                    header: data_page.header.clone(),
                    cache_key: cache_key.clone(),
                    is_asked_for_again: false,
                });
                *slot = None;
            }
        }
    }

    /// Takes back the bytes of the pages let go of. `find_bytes` gives the
    /// bytes the decompressed cache holds under a key, or `None` if it
    /// holds them no more. Returns whether every page is back. A page
    /// whose bytes were not found stays let go of, until it is asked for
    /// again (see [`list_stretches_to_ask_for`](Self::list_stretches_to_ask_for))
    /// and arrives like any page. Once it is asked for, it is not looked
    /// for in the cache again: its bytes are on their way, and the cache
    /// may hold them before they have arrived.
    pub(crate) fn take_back_page_bytes(
        &mut self,
        mut find_bytes: impl FnMut(&CacheKey) -> Option<Vec<Bytes>>,
    ) -> bool {
        let mut is_every_page_back = true;
        for column in &mut self.columns {
            let ColumnPages {
                pages,
                let_go_pages,
                ..
            } = column;
            let_go_pages.retain(|page| {
                if page.is_asked_for_again {
                    is_every_page_back = false;
                    return true;
                }
                let Some(data) = find_bytes(&page.cache_key) else {
                    is_every_page_back = false;
                    return true;
                };
                pages[page.idx] = Some(Arc::new(StoredPage {
                    idx: page.idx,
                    first_row: page.first_row,
                    rows: page.rows,
                    cache_key: Some(page.cache_key.clone()),
                    content: PageContent::Data(DataPage {
                        header: page.header.clone(),
                        data,
                        filter_mask: None,
                    }),
                }));
                false
            });
        }
        is_every_page_back
    }

    /// The pages let go of that were not asked for again, each as the
    /// stretch of its chunk to ask for, with `worker` as the worker its
    /// bytes are sent to. They count as asked for from here on. The caller
    /// asks for them when [`take_back_page_bytes`](Self::take_back_page_bytes)
    /// did not find their bytes.
    pub(crate) fn list_stretches_to_ask_for(&mut self, worker: usize) -> Vec<PageStretch> {
        let mut page_stretches = Vec::new();
        for (leaf, column) in self.columns.iter_mut().enumerate() {
            let pages_to_ask_for = column.let_go_pages.iter_mut();
            for page in pages_to_ask_for.filter(|page| !page.is_asked_for_again) {
                page.is_asked_for_again = true;
                page_stretches.push(PageStretch {
                    leaf,
                    file_offset: page.cache_key.offset,
                    len: page.cache_key.span,
                    first_page_idx: page.idx,
                    first_row: page.first_row,
                    worker,
                });
            }
        }
        page_stretches
    }

    /// How many pages of `column` the collection still holds.
    #[cfg(test)]
    pub(crate) fn held_pages(&self, column: usize) -> usize {
        self.columns[column].pages.iter().flatten().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::tests::{i32_schema, make_test_table};
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use crate::types::projection::Projection;
    use crate::types::table::ParquetTable;

    fn pages_of(table: &Arc<ParquetTable>) -> RowGroupPages {
        let metadata = QueryRowGroupMetadata::new(table, 0, RowSelection::All);
        let plan =
            DecodePlan::new(&metadata, &Projection::all_from_schema(table.schema()), &[]).unwrap();
        RowGroupPages::new(metadata, Arc::new(plan), Arc::new(AtomicUsize::new(1)))
    }

    fn data_page(rows: u32) -> PageContent {
        encoded_page(rows, Encoding::PLAIN)
    }

    fn encoded_page(rows: u32, encoding: Encoding) -> PageContent {
        let header = PageHeader::for_data_page(rows as i32, encoding);
        PageContent::Data(DataPage {
            header: header.data_page_header.unwrap(),
            data: Vec::new(),
            filter_mask: None,
        })
    }

    /// A page of 40,000 rows of the one column of a table, which the
    /// decompressed cache holds under a key.
    fn pages_of_one_cached_page() -> RowGroupPages {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a"]), 40_000));
        let file = pages.metadata().get_metadata().open_file.clone();
        let cache_key = CacheKey::new(file, 7, 9);
        pages.insert_page(0, 0, 0, 40_000, data_page(40_000), Some(cache_key));
        pages
    }

    #[test]
    fn a_page_let_go_of_is_cut_again_once_its_bytes_are_back() {
        let mut pages = pages_of_one_cached_page();
        let first = pages.cut_range(0);

        pages.let_go_of_page_bytes();
        let held_while_let_go = pages.held_pages(0);
        let is_every_page_back = pages.take_back_page_bytes(|_| Some(Vec::new()));
        let second = pages.cut_range(1);

        assert_eq!((held_while_let_go, is_every_page_back), (0, true));
        assert_eq!((first.rows(), second.rows()), (0..16_384, 16_384..32_768));
        assert_eq!(second.column(0).pages.len(), 1);
    }

    #[test]
    fn a_page_the_cache_holds_no_more_is_asked_for_again_once() {
        let mut pages = pages_of_one_cached_page();
        pages.let_go_of_page_bytes();

        let is_every_page_back = pages.take_back_page_bytes(|_| None);
        let asked_for = pages.list_stretches_to_ask_for(3);
        let asked_for_again = pages.list_stretches_to_ask_for(3);

        let [page_stretch] = asked_for.as_slice() else {
            panic!("one page is asked for");
        };
        assert!(!is_every_page_back);
        assert_eq!(
            (
                page_stretch.leaf,
                page_stretch.file_offset,
                page_stretch.len
            ),
            (0, 7, 9)
        );
        assert_eq!(
            (
                page_stretch.first_page_idx,
                page_stretch.first_row,
                page_stretch.worker
            ),
            (0, 0, 3)
        );
        assert!(asked_for_again.is_empty());
    }

    #[test]
    fn a_page_asked_for_again_is_not_taken_from_the_cache_before_it_arrives() {
        let mut pages = pages_of_one_cached_page();
        pages.let_go_of_page_bytes();
        pages.take_back_page_bytes(|_| None);
        pages.list_stretches_to_ask_for(3);

        let is_every_page_back = pages.take_back_page_bytes(|_| Some(Vec::new()));

        assert!(!is_every_page_back);
        assert_eq!(pages.held_pages(0), 0);
    }

    #[test]
    fn a_page_asked_for_again_is_cut_once_it_arrives() {
        let mut pages = pages_of_one_cached_page();
        pages.let_go_of_page_bytes();
        pages.take_back_page_bytes(|_| None);

        pages.insert_page(0, 0, 0, 40_000, data_page(40_000), None);
        let is_every_page_back = pages.take_back_page_bytes(|_| None);
        let range = pages.cut_range(0);

        assert!(is_every_page_back);
        assert_eq!(range.column(0).pages.len(), 1);
    }

    #[test]
    fn a_range_is_ready_once_every_column_has_contiguous_pages_over_it() {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a", "b"]), 40_000));
        pages.insert_page(0, 1, 20_000, 20_000, data_page(20_000), None);
        pages.insert_page(1, 0, 0, 40_000, data_page(40_000), None);

        let before = pages.ready_ranges();
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000), None);
        let after = pages.ready_ranges();

        assert_eq!((before, after), (0, 3));
    }

    /// The footer of this table records no dictionary page: the pages
    /// themselves say the column needs one.
    #[test]
    fn a_dictionary_encoded_column_is_not_ready_before_its_dictionary() {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a", "b"]), 1_000));
        pages.insert_page(
            0,
            0,
            0,
            1_000,
            encoded_page(1_000, Encoding::RLE_DICTIONARY),
            None,
        );
        pages.insert_page(1, 0, 0, 1_000, data_page(1_000), None);

        let before = pages.ready_ranges();
        pages.insert_dictionary(0, Arc::new(0u8));
        let after = pages.ready_ranges();

        assert_eq!((before, after), (0, 1));
    }

    /// Two pages over three ranges: the first page spans the first two
    /// ranges, the second page the last two. A range takes the pages over
    /// it, and the collection forgets a page with the last range over it.
    #[test]
    fn a_range_takes_its_pages_and_the_last_range_over_a_page_forgets_it() {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a"]), 40_000));
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000), None);
        pages.insert_page(0, 1, 20_000, 20_000, data_page(20_000), None);

        let first = pages.cut_range(0);
        let held_after_first = pages.held_pages(0);
        let second = pages.cut_range(1);
        let held_after_second = pages.held_pages(0);
        let third = pages.cut_range(2);

        assert_eq!(first.column(0).pages.len(), 1);
        assert_eq!(second.column(0).pages.len(), 2);
        assert_eq!(third.column(0).pages.len(), 1);
        assert_eq!(
            (held_after_first, held_after_second, pages.held_pages(0)),
            (2, 1, 0)
        );
    }

    #[test]
    fn the_row_group_counts_as_decoded_with_its_last_range() {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a"]), 20_000));
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000), None);
        let ranges: Vec<DecodeRange> = (0..pages.range_count())
            .map(|i| pages.cut_range(i))
            .collect();
        let left = ranges[0].rows_left().clone();

        for range in ranges {
            range.decoded();
        }

        assert_eq!(left.load(Ordering::Relaxed), 0);
    }
}
