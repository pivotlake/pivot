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
use crate::thrift::headers::DataPageHeader;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::page::DataPage;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Rows per decode range. A range is the unit another worker can take over,
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

/// A decompressed data page.
pub struct StoredPage {
    /// The page's index within its column chunk.
    pub idx: usize,
    /// The row group row of the page's first value.
    pub first_row: u32,
    /// The page's values, which are its row group rows.
    pub rows: u32,
    pub content: PageContent,
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
    /// The row group's ranges still to decode, shared by all of them.
    ranges_left: Arc<AtomicUsize>,
    /// The scan's count of row groups claimed but not yet decoded, released
    /// with the row group's last range.
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

    /// The counter of the row group's ranges still to decode; zero once
    /// every range is decoded, on whichever worker.
    pub fn ranges_left(&self) -> &Arc<AtomicUsize> {
        &self.0.ranges_left
    }

    /// Records the range as decoded, and the row group with it when this was
    /// its last range.
    pub fn decoded(self) {
        if self.0.ranges_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0
                .outstanding_row_groups
                .fetch_sub(1, Ordering::Relaxed);
        }
    }
}

struct ColumnPages {
    dictionary: Option<SharedDictionary>,
    /// Indexed by page number; `None` until the page arrives, and again
    /// once the last range over it has been cut.
    pages: Vec<Option<Arc<StoredPage>>>,
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
    /// Whether each column's chunk has a dictionary page; a range is not
    /// ready until the dictionaries its decoder needs have arrived.
    expects_dictionary: Vec<bool>,
    /// The decode ranges, in decoded-row terms (see [`DecodeRange::rows`]).
    ranges: Vec<Range<u32>>,
    /// Per range, how many row group rows every column must have contiguous
    /// pages for before the range can be decoded.
    rows_needed: Vec<u32>,
    ranges_left: Arc<AtomicUsize>,
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
        let expects_dictionary = plan.expects_dictionary(&metadata);
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
        let columns = expects_dictionary
            .iter()
            .map(|_| ColumnPages {
                dictionary: None,
                pages: Vec::new(),
                contiguous_pages: 0,
                contiguous_rows: 0,
            })
            .collect();
        Self {
            metadata,
            plan,
            columns,
            expects_dictionary,
            ranges_left: Arc::new(AtomicUsize::new(ranges.len())),
            ranges,
            rows_needed,
            outstanding_row_groups,
        }
    }

    pub(crate) fn metadata(&self) -> &QueryRowGroupMetadata {
        &self.metadata
    }

    pub(crate) fn plan(&self) -> &Arc<DecodePlan> {
        &self.plan
    }

    pub(crate) fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Keeps the built dictionary of `column`.
    pub(crate) fn insert_dictionary(&mut self, column: usize, dictionary: SharedDictionary) {
        self.columns[column].dictionary = Some(dictionary);
    }

    /// Keeps a data page of `column`.
    pub(crate) fn insert_page(
        &mut self,
        column: usize,
        idx: usize,
        first_row: u32,
        rows: u32,
        content: PageContent,
    ) {
        let pages = &mut self.columns[column];
        if pages.pages.len() <= idx {
            pages.pages.resize_with(idx + 1, || None);
        }
        assert!(pages.pages[idx].is_none(), "a page arrives once");
        pages.pages[idx] = Some(Arc::new(StoredPage {
            idx,
            first_row,
            rows,
            content,
        }));
        while let Some(Some(next)) = pages.pages.get(pages.contiguous_pages) {
            pages.contiguous_rows += next.rows;
            pages.contiguous_pages += 1;
        }
    }

    /// How many ranges, from the first, every column has the pages for.
    pub(crate) fn ready_ranges(&self) -> usize {
        let rows_ready = self
            .columns
            .iter()
            .zip(&self.expects_dictionary)
            .map(|(pages, expects_dictionary)| {
                if *expects_dictionary && pages.dictionary.is_none() {
                    0
                } else {
                    pages.contiguous_rows
                }
            })
            .min()
            .unwrap_or(0);
        self.rows_needed
            .iter()
            .take_while(|&&needed| needed <= rows_ready)
            .count()
    }

    /// Cuts range `index`, which must be ready, forgetting the pages no
    /// later range needs: from here on only the ranges cut over them keep
    /// them alive.
    pub(crate) fn cut_range(&mut self, index: usize) -> DecodeRange {
        let rows = self.ranges[index].clone();
        let last_range = index + 1 == self.ranges.len();
        let whole_row_group = matches!(self.metadata.selection(), RowSelection::Indices(_));
        let columns = self
            .columns
            .iter_mut()
            .map(|column| {
                let mut pages = Vec::new();
                for slot in column.pages.iter_mut() {
                    let Some(page) = slot else { continue };
                    if !whole_row_group && !page.overlaps(&rows) {
                        continue;
                    }
                    pages.push(page.clone());
                    if last_range || page.row_span().end <= rows.end {
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
            ranges_left: self.ranges_left.clone(),
            outstanding_row_groups: self.outstanding_row_groups.clone(),
        }))
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
    use crate::reading::decoding::tests::{i32_schema, make_test_table, string_and_i32_table};
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
        let header = PageHeader::for_data_page(rows as i32, Encoding::PLAIN);
        PageContent::Data(DataPage {
            header: header.data_page_header.unwrap(),
            data: Vec::new(),
            filter_mask: None,
        })
    }

    #[test]
    fn a_range_is_ready_once_every_column_has_contiguous_pages_over_it() {
        let mut pages = pages_of(&make_test_table(i32_schema(&["a", "b"]), 40_000));
        pages.insert_page(0, 1, 20_000, 20_000, data_page(20_000));
        pages.insert_page(1, 0, 0, 40_000, data_page(40_000));

        let before = pages.ready_ranges();
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000));
        let after = pages.ready_ranges();

        assert_eq!((before, after), (0, 3));
    }

    #[test]
    fn a_column_with_a_dictionary_is_not_ready_before_it() {
        let mut pages = pages_of(&string_and_i32_table(1_000, false));
        pages.insert_page(0, 0, 0, 1_000, data_page(1_000));
        pages.insert_page(1, 0, 0, 1_000, data_page(1_000));

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
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000));
        pages.insert_page(0, 1, 20_000, 20_000, data_page(20_000));

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
        pages.insert_page(0, 0, 0, 20_000, data_page(20_000));
        let ranges: Vec<DecodeRange> = (0..pages.range_count())
            .map(|i| pages.cut_range(i))
            .collect();
        let left = ranges[0].ranges_left().clone();

        for range in ranges {
            range.decoded();
        }

        assert_eq!(left.load(Ordering::Relaxed), 0);
    }
}
