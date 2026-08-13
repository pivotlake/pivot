//! A map, per column chunk, from data page to the row that page starts at.
//!
//! Decoding a column chunk is sequential: values can only be read from the
//! first page forward, so a row group is decoded end to end by whichever worker
//! claimed it. That caps a scan's decode parallelism at the number of row
//! groups, and leaves the wall time of the whole scan set by the largest row
//! group, however many idle workers are standing by.
//!
//! A page directory lifts that cap. The start of a data page is a clean decode
//! entry point - its values are self-contained, and no encoding carries state
//! across a page boundary - so knowing where every page begins is enough to
//! start decoding at any row: seek to the page holding that row, skip the rows
//! ahead of it inside the page, and decode on from there. A row group can then
//! be cut into row ranges that separate workers decode independently.
//!
//! Recording it is nearly free: splitting a chunk into pages already walks
//! every page header, so the first scan of a row group fills the directory in
//! as a side effect and caches it on the shared
//! [`RowGroupMetadata`](super::metadata::RowGroupMetadata). Every later scan of
//! that file reuses it. Until then the row group reads whole, exactly as it
//! always did, which is what makes the directory a pure optimization: a scan
//! never needs one to be correct.

use crate::parquet::reading::decoding::leaf_decoders::{LeafCheckpoint, PageCheckpoint};
use std::ops::Range;
use std::sync::OnceLock;

/// Where one data page sits inside its column chunk, and which rows it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageEntry {
    /// Offset of the page's header within the file. This is where a read of
    /// the page has to start, since the header precedes the values.
    pub file_offset: usize,
    /// Bytes the page occupies, its header included.
    pub span: usize,
    /// Index of the page's first row within the row group, counting every row
    /// whether or not a filter would keep it.
    pub first_row: usize,
    /// Rows the page holds.
    pub num_rows: usize,
}

impl PageEntry {
    /// One past the page's last byte in the file.
    fn end_offset(&self) -> usize {
        self.file_offset + self.span
    }

    /// One past the page's last row.
    fn end_row(&self) -> usize {
        self.first_row + self.num_rows
    }
}

/// Every data page of one column chunk, in file order.
///
/// Dictionary pages are deliberately absent: they hold no rows, so they cannot
/// be sought to, and a chunk's dictionary is needed by every reader of it
/// regardless of which rows that reader wants (see
/// [`dictionary_span`](Self::dictionary_span)).
#[derive(Clone, Debug)]
pub struct ColumnPageDirectory {
    pages: Vec<PageEntry>,
    /// The chunk's dictionary page, when it has one: everything from the start
    /// of the chunk up to its first data page.
    dictionary: Option<Range<usize>>,
}

impl ColumnPageDirectory {
    /// Builds a directory from one chunk's data pages, in file order, plus the
    /// byte range of its dictionary page if it has one.
    pub fn new(pages: Vec<PageEntry>, dictionary: Option<Range<usize>>) -> Self {
        Self { pages, dictionary }
    }

    /// Total rows the directory accounts for.
    pub fn num_rows(&self) -> usize {
        self.pages.last().map(PageEntry::end_row).unwrap_or(0)
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Rows in the largest data page, which bounds how far into a page a reader
    /// starting at an arbitrary row has to skip before reaching it.
    pub fn max_page_rows(&self) -> usize {
        self.pages.iter().map(|p| p.num_rows).max().unwrap_or(0)
    }

    /// The dictionary page's byte range, when the chunk has one. Every reader
    /// of the chunk needs it, whichever rows it reads, so a split fetches it
    /// alongside its own data pages.
    pub fn dictionary_span(&self) -> Option<Range<usize>> {
        self.dictionary.clone()
    }

    /// The data pages covering rows `rows`, as a page index range.
    ///
    /// The first page is the one holding `rows.start`; the last is the one
    /// holding the final row of the range. An empty range yields an empty page
    /// range.
    pub fn pages_covering(&self, rows: Range<usize>) -> Range<usize> {
        if rows.is_empty() {
            return 0..0;
        }
        let first = self.page_holding(rows.start);
        // `rows.end` is exclusive, so the last covered page is the one holding
        // the row before it. Seeking to `rows.end` itself would step one page
        // too far whenever the range ends exactly on a page boundary.
        let last = self.page_holding(rows.end - 1);
        first..last + 1
    }

    /// Index of the page holding `row`, saturating at the last page for a row
    /// beyond the chunk.
    fn page_holding(&self, row: usize) -> usize {
        // Pages are sorted by `first_row`, so the page holding `row` is the
        // last one starting at or before it.
        match self.pages.binary_search_by_key(&row, |p| p.first_row) {
            Ok(exact) => exact,
            Err(0) => 0,
            Err(after) => after - 1,
        }
    }

    /// How many of its page's rows precede `row` — the rows a decoder must
    /// skip after seeking to that page before `row` is the next one out.
    pub fn skip_into_page(&self, row: usize) -> usize {
        let page = &self.pages[self.page_holding(row)];
        row.saturating_sub(page.first_row)
    }

    /// The start of the page boundary nearest `row`.
    ///
    /// Cutting a split here costs this chunk no skip at all, because the split
    /// begins exactly where one of its pages does.
    pub fn nearest_page_start(&self, row: usize) -> usize {
        let page = self.page_holding(row);
        let at = self.pages[page].first_row;
        match self.pages.get(page + 1) {
            // Round to whichever boundary is closer, so aligning perturbs the
            // requested cut as little as possible and the splits stay even.
            Some(next) if next.first_row - row < row - at => next.first_row,
            _ => at,
        }
    }

    /// The byte range spanned by a run of data pages.
    pub fn span_of(&self, pages: Range<usize>) -> Option<Range<usize>> {
        let first = self.pages.get(pages.start)?;
        let last = self.pages.get(pages.end.checked_sub(1)?)?;
        Some(first.file_offset..last.end_offset())
    }

    /// The row the page at `page` starts on.
    pub fn first_row_of(&self, page: usize) -> usize {
        self.pages[page].first_row
    }
}

/// Decode positions recorded for one column chunk, in row order.
///
/// A page start is a clean entry point, but a split rarely begins on one, and
/// reaching its first row by skipping from the page start costs about what
/// decoding those rows would on a run-length encoded page. A checkpoint records
/// the exact byte the row's value begins at, turning that walk into a seek.
///
/// Every leaf of a row group is read in lockstep - one batch advances them all
/// by the same number of rows - so checkpoints land on the same rows in every
/// column, and a split cut on a recorded row resumes each of its columns with
/// nothing left to skip.
#[derive(Clone, Debug)]
pub struct ColumnCheckpoints {
    points: Vec<LeafCheckpoint>,
}

impl ColumnCheckpoints {
    pub fn new(points: Vec<LeafCheckpoint>) -> Self {
        Self { points }
    }

    /// The last checkpoint at or before `row`, which is where a reader wanting
    /// `row` starts; whatever rows remain between the two are skipped.
    pub fn at_or_before(&self, row: usize) -> Option<LeafCheckpoint> {
        let idx = match self.points.binary_search_by_key(&row, |p| p.row) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(after) => after - 1,
        };
        Some(self.points[idx])
    }

    /// The rows checkpoints were recorded at, in order. A split cut on one of
    /// these resumes every column exactly, with no rows to skip.
    pub fn rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.points.iter().map(|p| p.row)
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Where a split enters one leaf chunk.
///
/// Computed in one place so the fetch and the decode cannot disagree about
/// which page a split starts at — fetching from a later page than the decoder
/// resumes in would leave it waiting for a page nobody read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafEntry {
    /// The chunk's first page this split reads, indexed within the whole chunk.
    pub page_idx: usize,
    /// The recorded position inside that page, when one was available.
    pub checkpoint: Option<PageCheckpoint>,
    /// Rows of that page preceding `checkpoint`.
    pub rows_into_page: usize,
    /// Rows to discard after entering, to land on the split's first row. Zero
    /// when the split was cut exactly on a recorded row.
    pub skip_rows: usize,
}

/// Where a split starting at `first_row` should enter a chunk.
///
/// With a checkpoint it enters at the recorded byte and drops whatever rows
/// remain between there and its first row. Without one it enters the page
/// holding `first_row` and skips from that page's start, which is correct but
/// costs a walk over those rows.
pub fn leaf_entry(
    directory: &ColumnPageDirectory,
    checkpoints: Option<&ColumnCheckpoints>,
    first_row: usize,
) -> LeafEntry {
    match checkpoints.and_then(|points| points.at_or_before(first_row)) {
        Some(point) => LeafEntry {
            page_idx: point.at.page_idx,
            checkpoint: Some(point.at),
            rows_into_page: point.row - directory.first_row_of(point.at.page_idx),
            skip_rows: first_row - point.row,
        },
        None => LeafEntry {
            page_idx: directory.pages_covering(first_row..first_row + 1).start,
            checkpoint: None,
            rows_into_page: 0,
            skip_rows: directory.skip_into_page(first_row),
        },
    }
}

/// One row group's recorded decode positions, one slot per leaf column chunk.
///
/// Filled in by the first scan that decodes the row group whole, and read by
/// every later one, exactly like [`RowGroupPageDirectory`]. Holding only plain
/// numbers, it pins no decoded buffers and is safe to keep for the process's
/// lifetime.
pub struct RowGroupCheckpoints {
    columns: Box<[OnceLock<ColumnCheckpoints>]>,
}

impl RowGroupCheckpoints {
    pub fn new(num_columns: usize) -> Self {
        Self {
            columns: (0..num_columns).map(|_| OnceLock::new()).collect(),
        }
    }

    pub fn get(&self, leaf: usize) -> Option<&ColumnCheckpoints> {
        self.columns.get(leaf).and_then(OnceLock::get)
    }

    /// Records `checkpoints` for leaf chunk `leaf`, keeping whichever entry
    /// wins if two scans record concurrently — they describe the same immutable
    /// file bytes.
    pub fn record(&self, leaf: usize, checkpoints: ColumnCheckpoints) {
        if let Some(slot) = self.columns.get(leaf) {
            let _ = slot.set(checkpoints);
        }
    }
}

impl Clone for RowGroupCheckpoints {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.iter().cloned().collect(),
        }
    }
}

/// One row group's directories, one slot per leaf column chunk.
///
/// Slots fill in independently and at most once: a scan records only the
/// chunks it actually read, so a column no query has projected yet simply has
/// no directory, and a row group becomes splittable for a given projection
/// once every leaf that projection reads has one.
pub struct RowGroupPageDirectory {
    columns: Box<[OnceLock<ColumnPageDirectory>]>,
}

impl RowGroupPageDirectory {
    pub fn new(num_columns: usize) -> Self {
        Self {
            columns: (0..num_columns).map(|_| OnceLock::new()).collect(),
        }
    }

    /// The directory for leaf chunk `leaf`, if a scan has recorded one.
    pub fn get(&self, leaf: usize) -> Option<&ColumnPageDirectory> {
        self.columns.get(leaf).and_then(OnceLock::get)
    }

    /// Records `directory` for leaf chunk `leaf`, keeping whichever entry wins
    /// if two scans record concurrently. They describe the same immutable file
    /// bytes, so the loser is discarded rather than merged.
    pub fn record(&self, leaf: usize, directory: ColumnPageDirectory) {
        if let Some(slot) = self.columns.get(leaf) {
            let _ = slot.set(directory);
        }
    }
}

impl Clone for RowGroupPageDirectory {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.iter().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages of `rows` rows each, laid out back to back from byte 100.
    fn directory(page_rows: &[usize]) -> ColumnPageDirectory {
        let mut pages = Vec::new();
        let mut first_row = 0;
        let mut offset = 100;
        for &num_rows in page_rows {
            let span = num_rows * 4;
            pages.push(PageEntry {
                file_offset: offset,
                span,
                first_row,
                num_rows,
            });
            first_row += num_rows;
            offset += span;
        }
        ColumnPageDirectory::new(pages, None)
    }

    #[test]
    fn num_rows_sums_the_pages() {
        let dir = directory(&[10, 20, 5]);

        assert_eq!(dir.num_rows(), 35);
    }

    #[test]
    fn pages_covering_selects_the_pages_a_row_range_touches() {
        let dir = directory(&[10, 10, 10]);

        assert_eq!(dir.pages_covering(0..10), 0..1);
        assert_eq!(dir.pages_covering(5..15), 0..2);
        assert_eq!(dir.pages_covering(10..30), 1..3);
        assert_eq!(dir.pages_covering(29..30), 2..3);
    }

    /// A range ending exactly on a page boundary must not pull in the page
    /// that starts there, which holds none of its rows.
    #[test]
    fn pages_covering_excludes_the_page_a_range_ends_on() {
        let dir = directory(&[10, 10, 10]);

        assert_eq!(dir.pages_covering(0..20), 0..2);
    }

    #[test]
    fn pages_covering_is_empty_for_an_empty_range() {
        let dir = directory(&[10, 10]);

        assert_eq!(dir.pages_covering(7..7), 0..0);
    }

    #[test]
    fn skip_into_page_counts_rows_before_the_target() {
        let dir = directory(&[10, 10, 10]);

        assert_eq!(dir.skip_into_page(0), 0);
        assert_eq!(dir.skip_into_page(7), 7);
        assert_eq!(dir.skip_into_page(10), 0);
        assert_eq!(dir.skip_into_page(23), 3);
    }

    /// Cutting a split on a page start costs that chunk no skip, so a
    /// requested cut rounds to whichever boundary is closer.
    #[test]
    fn nearest_page_start_rounds_to_the_closer_boundary() {
        let dir = directory(&[10, 10, 10]);

        assert_eq!(dir.nearest_page_start(0), 0);
        assert_eq!(dir.nearest_page_start(4), 0);
        assert_eq!(dir.nearest_page_start(6), 10);
        assert_eq!(dir.nearest_page_start(10), 10);
        assert_eq!(dir.nearest_page_start(24), 20);
    }

    /// Past the last page there is no later boundary to round up to.
    #[test]
    fn nearest_page_start_clamps_to_the_last_page() {
        let dir = directory(&[10, 10]);

        assert_eq!(dir.nearest_page_start(19), 10);
        assert_eq!(dir.nearest_page_start(100), 10);
    }

    #[test]
    fn max_page_rows_reports_the_largest_page() {
        let dir = directory(&[10, 25, 5]);

        assert_eq!(dir.max_page_rows(), 25);
    }

    #[test]
    fn span_of_covers_the_pages_end_to_end() {
        let dir = directory(&[10, 20]);

        assert_eq!(dir.span_of(0..1), Some(100..140));
        assert_eq!(dir.span_of(1..2), Some(140..220));
        assert_eq!(dir.span_of(0..2), Some(100..220));
        assert_eq!(dir.span_of(0..0), None);
    }

    #[test]
    fn first_row_of_reports_each_page_start() {
        let dir = directory(&[10, 20, 5]);

        assert_eq!(dir.first_row_of(0), 0);
        assert_eq!(dir.first_row_of(1), 10);
        assert_eq!(dir.first_row_of(2), 30);
    }

    /// A leaf keeps the first directory recorded for it; a later, racing
    /// recorder does not overwrite it.
    #[test]
    fn record_keeps_the_first_directory_per_leaf() {
        let directories = RowGroupPageDirectory::new(2);

        directories.record(0, directory(&[10]));
        directories.record(0, directory(&[99]));

        assert_eq!(directories.get(0).unwrap().num_rows(), 10);
        assert!(directories.get(1).is_none());
    }
}
