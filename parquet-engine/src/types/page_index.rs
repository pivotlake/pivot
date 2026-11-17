//! Where the data pages of a row group's column chunks are in the file.
//!
//! A Parquet file may carry an offset index per column chunk: the file
//! offset, size and first row of each of its data pages. With it a read of
//! some rows of a row group fetches only the pages holding them instead of
//! every page of every projected chunk.
//!
//! A table holds the index of every row group of every file for as long as
//! it is loaded, so [`RowGroupPageIndex`] keeps it small. The data pages of a
//! chunk lie back to back and end where the chunk ends, so a page is kept as
//! its size and first row alone, eight bytes, and its offset is counted back
//! from the chunk's end. A chunk with one data page keeps nothing: its only
//! page is the chunk past the dictionary page, and reading it is reading the
//! whole chunk. A file without an index keeps nothing either, and its chunks
//! are read whole.

use crate::thrift::footer::OffsetIndex;
use crate::types::metadata::ColumnChunkMeta;
use crate::types::table::{Error, Result};
use std::ops::Range;

/// One data page of a column chunk, as [`RowGroupPageIndex`] keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexedPage {
    /// The page's header and compressed body together, in bytes.
    compressed_size: u32,
    /// The row group row of the page's first value.
    first_row: u32,
}

/// The data pages of the leaves of one row group that span several pages.
#[derive(Debug, PartialEq, Eq)]
pub struct RowGroupPageIndex {
    /// Where each leaf's pages start in `pages`, in leaf order, and where
    /// the last leaf's end.
    leaf_page_starts: Box<[u32]>,
    pages: Box<[IndexedPage]>,
}

/// One data page of a column chunk: where it is in the file and the rows it
/// holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataPageLocation {
    /// The page's index among the chunk's data pages.
    pub page_idx: usize,
    /// The file offset of the page's header.
    pub file_offset: usize,
    /// The page's header and compressed body together, in bytes.
    pub compressed_size: usize,
    /// The row group rows the page holds.
    pub rows: Range<u32>,
}

impl RowGroupPageIndex {
    /// Build the index of a row group of `row_count` rows from the chunk and
    /// offset index of each of its leaves, in leaf order; a leaf without an
    /// offset index gives `None`. Returns `None` when no leaf spans several
    /// indexed pages, which leaves nothing to keep.
    ///
    /// An offset index that does not describe its chunk is an error: reading
    /// by it would fetch the wrong bytes.
    pub(crate) fn from_offset_indexes<'a>(
        leaves: impl Iterator<Item = Option<(&'a ColumnChunkMeta, &'a OffsetIndex)>>,
        row_count: i64,
    ) -> Result<Option<Self>> {
        let mut leaf_page_starts = vec![0u32];
        let mut pages = Vec::new();
        for leaf in leaves {
            if let Some((chunk, offset_index)) = leaf {
                validate_offset_index(chunk, offset_index, row_count)?;
                if offset_index.page_locations.len() > 1 {
                    pages.extend(
                        offset_index
                            .page_locations
                            .iter()
                            .map(|location| IndexedPage {
                                compressed_size: location.compressed_page_size as u32,
                                first_row: location.first_row_index as u32,
                            }),
                    );
                }
            }
            leaf_page_starts.push(pages.len() as u32);
        }
        if pages.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            leaf_page_starts: leaf_page_starts.into_boxed_slice(),
            pages: pages.into_boxed_slice(),
        }))
    }

    /// The data pages of `leaf`, whose chunk is `chunk`, in file order, when
    /// the leaf spans several indexed pages.
    pub(crate) fn locate_leaf_pages(
        &self,
        leaf: usize,
        chunk: &ColumnChunkMeta,
        row_count: u32,
    ) -> Option<impl Iterator<Item = DataPageLocation> + '_> {
        let pages = &self.pages
            [self.leaf_page_starts[leaf] as usize..self.leaf_page_starts[leaf + 1] as usize];
        if pages.is_empty() {
            return None;
        }
        let pages_size: usize = pages.iter().map(|page| page.compressed_size as usize).sum();
        let mut file_offset = chunk.byte_range().end - pages_size;
        Some(pages.iter().enumerate().map(move |(page_idx, page)| {
            let location = DataPageLocation {
                page_idx,
                file_offset,
                compressed_size: page.compressed_size as usize,
                rows: page.first_row
                    ..pages
                        .get(page_idx + 1)
                        .map_or(row_count, |next| next.first_row),
            };
            file_offset += location.compressed_size;
            location
        }))
    }
}

/// Check that `offset_index` describes the data pages of `chunk`, a chunk of
/// a row group of `row_count` rows: the pages lie back to back, the last one
/// ends where the chunk ends, and their first rows start at zero and never
/// go back.
fn validate_offset_index(
    chunk: &ColumnChunkMeta,
    offset_index: &OffsetIndex,
    row_count: i64,
) -> Result<()> {
    let malformed = |problem: String| Error::InvalidFooter(format!("offset index: {problem}"));
    let locations = &offset_index.page_locations;
    let Some(first) = locations.first() else {
        return if row_count == 0 {
            Ok(())
        } else {
            Err(malformed(format!(
                "no pages listed for a chunk of {row_count} rows"
            )))
        };
    };
    let chunk_bytes = chunk.byte_range();
    if first.first_row_index != 0 {
        return Err(malformed(format!(
            "the first page starts at row {}",
            first.first_row_index
        )));
    }
    if first.offset < chunk_bytes.start as i64 {
        return Err(malformed(format!(
            "the first page is at offset {} but its chunk starts at {}",
            first.offset, chunk_bytes.start
        )));
    }
    let mut page_end = first.offset;
    let mut previous_first_row = 0;
    for location in locations {
        if location.offset != page_end {
            return Err(malformed(format!(
                "a page is at offset {} but the page before it ends at {page_end}",
                location.offset
            )));
        }
        if location.compressed_page_size <= 0 {
            return Err(malformed(format!(
                "a page of {} bytes",
                location.compressed_page_size
            )));
        }
        if location.first_row_index < previous_first_row || location.first_row_index > row_count {
            return Err(malformed(format!(
                "a page starts at row {} after one starting at row {previous_first_row}, in a row group of {row_count} rows",
                location.first_row_index
            )));
        }
        page_end += location.compressed_page_size as i64;
        previous_first_row = location.first_row_index;
    }
    if page_end != chunk_bytes.end as i64 {
        return Err(malformed(format!(
            "the last page ends at offset {page_end} but its chunk ends at {}",
            chunk_bytes.end
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thrift::footer::PageLocation;

    /// A chunk of `size` bytes starting at file offset 100, without a
    /// dictionary page.
    fn chunk_of_size(size: i64) -> ColumnChunkMeta {
        ColumnChunkMeta {
            data_page_offset: 100,
            total_compressed_size: size,
            absent: false,
            ..ColumnChunkMeta::absent()
        }
    }

    /// An offset index of pages given as `(offset, size, first row)`.
    fn offset_index(pages: &[(i64, i32, i64)]) -> OffsetIndex {
        OffsetIndex {
            page_locations: pages
                .iter()
                .map(
                    |&(offset, compressed_page_size, first_row_index)| PageLocation {
                        offset,
                        compressed_page_size,
                        first_row_index,
                    },
                )
                .collect(),
        }
    }

    #[test]
    fn pages_are_located_from_their_sizes_and_first_rows() {
        let chunk = chunk_of_size(60);
        let pages = offset_index(&[(100, 10, 0), (110, 30, 4), (140, 20, 9)]);

        let index =
            RowGroupPageIndex::from_offset_indexes([Some((&chunk, &pages))].into_iter(), 12)
                .unwrap()
                .unwrap();

        let located: Vec<_> = index.locate_leaf_pages(0, &chunk, 12).unwrap().collect();
        assert_eq!(
            located,
            vec![
                DataPageLocation {
                    page_idx: 0,
                    file_offset: 100,
                    compressed_size: 10,
                    rows: 0..4
                },
                DataPageLocation {
                    page_idx: 1,
                    file_offset: 110,
                    compressed_size: 30,
                    rows: 4..9
                },
                DataPageLocation {
                    page_idx: 2,
                    file_offset: 140,
                    compressed_size: 20,
                    rows: 9..12
                },
            ]
        );
    }

    #[test]
    fn the_pages_after_a_dictionary_page_are_counted_back_from_the_chunk_end() {
        let chunk = ColumnChunkMeta {
            dictionary_page_offset: Some(100),
            data_page_offset: 125,
            ..chunk_of_size(65)
        };
        let pages = offset_index(&[(125, 30, 0), (155, 10, 7)]);

        let index = RowGroupPageIndex::from_offset_indexes([Some((&chunk, &pages))].into_iter(), 8)
            .unwrap()
            .unwrap();

        let offsets: Vec<usize> = index
            .locate_leaf_pages(0, &chunk, 8)
            .unwrap()
            .map(|page| page.file_offset)
            .collect();
        assert_eq!(offsets, vec![125, 155]);
    }

    #[test]
    fn only_leaves_of_several_indexed_pages_are_kept() {
        let chunk = chunk_of_size(40);
        let one_page = offset_index(&[(100, 40, 0)]);
        let two_pages = offset_index(&[(100, 15, 0), (115, 25, 3)]);

        let index = RowGroupPageIndex::from_offset_indexes(
            [Some((&chunk, &one_page)), None, Some((&chunk, &two_pages))].into_iter(),
            5,
        )
        .unwrap()
        .unwrap();

        assert!(index.locate_leaf_pages(0, &chunk, 5).is_none());
        assert!(index.locate_leaf_pages(1, &chunk, 5).is_none());
        assert_eq!(index.locate_leaf_pages(2, &chunk, 5).unwrap().count(), 2);
    }

    #[test]
    fn a_row_group_of_single_page_leaves_keeps_no_index() {
        let chunk = chunk_of_size(40);
        let one_page = offset_index(&[(100, 40, 0)]);

        let index =
            RowGroupPageIndex::from_offset_indexes([Some((&chunk, &one_page))].into_iter(), 5)
                .unwrap();

        assert_eq!(index, None);
    }

    #[test]
    fn an_offset_index_that_does_not_describe_its_chunk_is_an_error() {
        let chunk = chunk_of_size(40);
        let malformed = [
            // A hole between two pages.
            offset_index(&[(100, 10, 0), (115, 25, 3)]),
            // The last page ends before the chunk does.
            offset_index(&[(100, 10, 0), (110, 25, 3)]),
            // The first page does not start at row zero.
            offset_index(&[(100, 15, 1), (115, 25, 3)]),
            // First rows going back.
            offset_index(&[(100, 15, 0), (115, 5, 3), (120, 20, 2)]),
            // A page starting past the row group's last row.
            offset_index(&[(100, 15, 0), (115, 25, 6)]),
            // No pages at all.
            offset_index(&[]),
        ];

        for pages in &malformed {
            let index =
                RowGroupPageIndex::from_offset_indexes([Some((&chunk, pages))].into_iter(), 5);

            assert!(matches!(index, Err(Error::InvalidFooter(_))), "{pages:?}");
        }
    }
}
