//! One input file of a compaction, and what was read of it.
//!
//! An [`InputFile`] is what the compaction read gets a file's rows from. It
//! is told which columns of which row groups to fetch
//! ([`FileRowGroup::request_column`]) and which rows of a column to decode
//! ([`InputFile::start_decoding_rows`]). It keeps the decoded values that
//! come back ([`FileRowGroup::store_decoded_values`]) and hands them out
//! ([`InputFile::take_input_values`]).
//!
//! It does not know why rows are wanted: the merge order and the output row
//! groups are the compaction read's. It fetches, decompresses and decodes
//! nothing itself: it asks through the [`ReadRequests`] that all files
//! share.
//!
//! Rows are numbered per file here ("file rows"): row 0 is the first row of
//! the file's first row group, and the numbering continues through the
//! file's later row groups. A [`ColumnRead`] numbers the rows of its own
//! row group from 0.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, Field};
use dispatch::Projection;

use super::decoded_values::DecodedValues;
use super::read_requests::ReadRequests;
use crate::ParquetTable;
use crate::reading::DECODE_RANGE_ROWS;
use crate::types::leaves::projected_leaves;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::writing::compacting::read::output::InputValues;
use crate::writing::compacting::read::row_order::key_merge::{RowGroupKeyBounds, SortKeyBounds};
use crate::writing::error::{WriteError, WriteResult};

/// What the decoded values of a column own, as its type says.
///
/// A string or binary value is decoded as a 16-byte view into its page or
/// its dictionary, so its text is not the values' own: the pages are
/// counted apart.
#[derive(Default, Clone, Copy)]
struct DecodedSize {
    /// The bytes the values of one row own, bitmaps apart: the view of each
    /// string or binary leaf, and the width of each fixed-width leaf.
    bytes_per_row: usize,
    /// How many bitmaps of one bit per row the values own at most: the
    /// values of each boolean leaf and the validity of each nullable
    /// field. The decoder leaves the validity out of values that hold no
    /// null, which is only known once they are decoded.
    bitmap_count: usize,
}

impl DecodedSize {
    /// What the decoded values of `field` own. A struct owns what its
    /// children do.
    fn compute_for_field(field: &Field) -> WriteResult<Self> {
        let mut size = match field.data_type() {
            DataType::Struct(children) => {
                let mut size = Self::default();
                for child in children {
                    let child_size = Self::compute_for_field(child)?;
                    size.bytes_per_row += child_size.bytes_per_row;
                    size.bitmap_count += child_size.bitmap_count;
                }
                size
            }
            DataType::Boolean => Self {
                bytes_per_row: 0,
                bitmap_count: 1,
            },
            DataType::Utf8View
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::BinaryView
            | DataType::Binary
            | DataType::LargeBinary => Self {
                bytes_per_row: size_of::<u128>(),
                bitmap_count: 0,
            },
            data_type => Self {
                bytes_per_row: data_type
                    .primitive_width()
                    .ok_or_else(|| WriteError::CompactionOfUndecodedType(data_type.clone()))?,
                bitmap_count: 0,
            },
        };
        size.bitmap_count += field.is_nullable() as usize;
        Ok(size)
    }

    /// The most bytes the decoded values of `row_count` rows own.
    fn count_bytes(&self, row_count: usize) -> usize {
        row_count * self.bytes_per_row + self.bitmap_count * row_count.div_ceil(8)
    }
}

/// The read of one column of one input row group.
pub(crate) struct ColumnRead {
    /// The size of the column's chunk in the file, from the footer. A
    /// request for the column fetches exactly this many bytes.
    pub(crate) compressed_bytes: usize,
    /// The size of the chunk's pages once decompressed, per row, from the
    /// footer.
    decompressed_bytes_per_row: usize,
    /// What the column's decoded values own.
    decoded_size: DecodedSize,
    /// How many rows the row group holds.
    row_count: u32,
    /// Whether the column's chunk was requested from the pipeline.
    pub(crate) is_requested: bool,
    /// Every row before this one was released to be decompressed and
    /// decoded. It is the end of a range of rows the decoder decodes in
    /// one go, or the end of the row group.
    released_rows_end: u32,
    /// How many released rows have not arrived decoded yet.
    awaited_row_count: u32,
    /// How many bytes the memory budget counts for those rows: the most
    /// their decoded values can own.
    awaited_rows_counted_bytes: usize,
    /// Every row before this one has arrived decoded.
    decoded_rows_end: u32,
    /// The decoded values that still hold rows to send on, by their first
    /// row. They arrive in any order.
    unsent_values: BTreeMap<u32, InputValues>,
}

impl ColumnRead {
    /// The read of table column `column` of the row group at `row_group` in
    /// the list of row groups of `table`, with nothing requested yet. The
    /// chunk's sizes are the footer's, summed over the column's leaves.
    /// The decoded size is that of the column's type in this row group.
    fn new(table: &ParquetTable, row_group: usize, column: usize) -> WriteResult<Self> {
        let row_group = QueryRowGroupMetadata::new(table, row_group, RowSelection::All);
        let fields = row_group.get_metadata().schema.fields();
        let mut compressed_bytes = 0;
        let mut decompressed_bytes = 0;
        for leaf in projected_leaves(fields, &row_group, &Projection::columns([column])) {
            let chunk = &row_group.columns()[leaf];
            if !chunk.absent {
                compressed_bytes += chunk.total_compressed_size as usize;
                decompressed_bytes += chunk.total_uncompressed_size as usize;
            }
        }
        let row_count = row_group.num_rows() as usize;
        Ok(Self {
            compressed_bytes,
            decompressed_bytes_per_row: decompressed_bytes.div_ceil(row_count.max(1)),
            decoded_size: DecodedSize::compute_for_field(&fields[column])?,
            row_count: row_count as u32,
            is_requested: false,
            released_rows_end: 0,
            awaited_row_count: 0,
            awaited_rows_counted_bytes: 0,
            decoded_rows_end: 0,
            unsent_values: BTreeMap::new(),
        })
    }

    /// The end of the released rows if every row before `rows_end` were
    /// released: `rows_end` rounded up to the end of the range of rows it
    /// falls in, since a range is decoded whole. Never before the rows
    /// already released.
    fn find_released_rows_end_for(&self, rows_end: u32) -> u32 {
        rows_end
            .next_multiple_of(DECODE_RANGE_ROWS)
            .min(self.row_count)
            .max(self.released_rows_end)
    }

    /// Keep `values`, decoded values of the column that arrived, whose
    /// first row is row `first_row` of the row group.
    fn store_values(&mut self, first_row: u32, values: InputValues) {
        self.unsent_values.insert(first_row, values);
        while let Some(next) = self.unsent_values.get(&self.decoded_rows_end) {
            self.decoded_rows_end += next.values.len() as u32;
        }
    }

    /// Append the decoded rows `rows` of the row group to `sources`, as
    /// slices of the values they arrived in, and let go of the values whose
    /// last row is among them. The memory budget counts values until every
    /// slice of them is dropped too.
    ///
    /// Rows are taken in row order, each once, so values whose last row is
    /// taken hold no row that is still to be taken.
    fn take_rows(&mut self, rows: Range<u32>, sources: &mut Vec<InputValues>) {
        let mut first_row = rows.start;
        while first_row < rows.end {
            let (&values_first_row, unsent) = self
                .unsent_values
                .range(..=first_row)
                .next_back()
                .expect("rows are taken after they have arrived decoded");
            let values_rows_end = values_first_row + unsent.values.len() as u32;
            let taken_rows_end = rows.end.min(values_rows_end);
            sources.push(InputValues {
                values: unsent.values.slice(
                    (first_row - values_first_row) as usize,
                    (taken_rows_end - first_row) as usize,
                ),
                memory: unsent.memory.clone(),
            });
            if taken_rows_end == values_rows_end {
                self.unsent_values.remove(&values_first_row);
            }
            first_row = taken_rows_end;
        }
    }
}

/// One row group of an input file, with the reads of its columns.
pub(crate) struct FileRowGroup {
    /// The row group's index in the table's list of row groups.
    pub(crate) table_index: usize,
    /// The file rows the row group holds.
    pub(crate) rows: Range<u64>,
    /// The read of each column of the table.
    pub(crate) columns: Vec<ColumnRead>,
}

impl FileRowGroup {
    /// The row of the row group that file row `file_rows_end` is, or the
    /// row group's end if that file row is past it.
    fn find_rows_end_within(&self, file_rows_end: u64) -> u32 {
        (file_rows_end.min(self.rows.end) - self.rows.start) as u32
    }

    /// Request the chunk of table column `column` from the pipeline. Does
    /// nothing if it was requested before. Returns whether the request was
    /// made now.
    pub(crate) fn request_column(&mut self, column: usize, requests: &mut ReadRequests) -> bool {
        let read = &mut self.columns[column];
        if read.is_requested {
            return false;
        }
        read.is_requested = true;
        requests.request_column(self.table_index, column, read.compressed_bytes);
        true
    }

    /// Estimate how many more bytes will be held if every row of table
    /// column `column` before `rows_end` is decoded: the chunk if it was
    /// not requested yet, compressed, and the rows not released yet, as
    /// decompressed pages and as decoded values.
    ///
    /// Only the pages are an estimate, from the chunk's average
    /// decompressed size per row: the sizes of its pages are not known
    /// before it is fetched.
    fn estimate_bytes_to_decode(&self, column: usize, rows_end: u32) -> usize {
        let read = &self.columns[column];
        let unrequested_bytes = match read.is_requested {
            true => 0,
            false => read.compressed_bytes,
        };
        let new_rows =
            (read.find_released_rows_end_for(rows_end) - read.released_rows_end) as usize;
        unrequested_bytes
            + new_rows * read.decompressed_bytes_per_row
            + read.decoded_size.count_bytes(new_rows)
    }

    /// Let every row of table column `column` before `rows_end` be
    /// decompressed and decoded, requesting the column's chunk first if it
    /// was not requested.
    ///
    /// Rows are decoded in whole ranges, so the rows up to the end of the
    /// range that `rows_end` falls in are released. The most their decoded
    /// values can own is counted as held from here on.
    fn start_decoding_rows(&mut self, column: usize, rows_end: u32, requests: &mut ReadRequests) {
        self.request_column(column, requests);
        let read = &mut self.columns[column];
        let released_rows_end = read.find_released_rows_end_for(rows_end);
        let new_row_count = released_rows_end - read.released_rows_end;
        let decoded_bytes = read.decoded_size.count_bytes(new_row_count as usize);
        read.released_rows_end = released_rows_end;
        read.awaited_row_count += new_row_count;
        read.awaited_rows_counted_bytes += decoded_bytes;
        requests.release_rows_before(self.table_index, column, released_rows_end, decoded_bytes);
    }

    /// Keep the decoded values of one of the row group's columns that came
    /// back from the pipeline, until their rows are taken. The memory
    /// budget counts the bytes they own from here on, in place of what it
    /// counted for their rows when they were released. The two differ only
    /// by the validity that values without a null do not hold.
    pub(crate) fn store_decoded_values(
        &mut self,
        decoded: DecodedValues,
        requests: &mut ReadRequests,
    ) -> WriteResult<()> {
        let read = &mut self.columns[decoded.column];
        let row_count = decoded.values.len() as u32;
        if row_count > read.awaited_row_count {
            return Err(WriteError::CompactionRowCountMismatch);
        }
        let counted_bytes = match row_count == read.awaited_row_count {
            true => read.awaited_rows_counted_bytes,
            false => {
                read.awaited_rows_counted_bytes * row_count as usize
                    / read.awaited_row_count as usize
            }
        };
        read.awaited_row_count -= row_count;
        read.awaited_rows_counted_bytes -= counted_bytes;
        let memory = Arc::new(requests.memory_budget.reserve(decoded.count_owned_bytes()));
        requests.memory_budget.subtract(counted_bytes);
        let values = InputValues {
            values: decoded.values,
            memory,
        };
        read.store_values(decoded.first_row, values);
        Ok(())
    }
}

/// One input file of a compaction.
pub(crate) struct InputFile {
    /// The file's row groups that hold rows, in file order.
    pub(crate) row_groups: Vec<FileRowGroup>,
}

impl InputFile {
    /// The files of `table` that hold rows, in the table's order, with
    /// nothing requested of them yet.
    ///
    /// A table lists the row groups of all its files in one list, each
    /// file's row groups together and in file order. A row group whose index
    /// within its file is not higher than the previous row group's therefore
    /// starts a new file.
    pub(crate) fn list_files_of(table: &ParquetTable) -> WriteResult<Vec<InputFile>> {
        let column_count = table.schema().fields().len();
        let mut files: Vec<InputFile> = Vec::new();
        let mut previous_index_in_file = None;
        for (table_index, row_group) in table.row_groups().iter().enumerate() {
            let starts_new_file = previous_index_in_file
                .is_none_or(|previous| row_group.file_row_group_idx <= previous);
            previous_index_in_file = Some(row_group.file_row_group_idx);
            if starts_new_file {
                files.push(InputFile {
                    row_groups: Vec::new(),
                });
            }
            let file = files.last_mut().unwrap();
            let row_group_rows = row_group.num_rows as u64;
            if row_group_rows > 0 {
                let first_row = file.count_rows();
                file.row_groups.push(FileRowGroup {
                    table_index,
                    rows: first_row..first_row + row_group_rows,
                    columns: (0..column_count)
                        .map(|column| ColumnRead::new(table, table_index, column))
                        .collect::<WriteResult<_>>()?,
                });
            }
        }
        files.retain(|file| !file.row_groups.is_empty());
        Ok(files)
    }

    /// How many rows the file holds: its last row group ends with the
    /// file's last row.
    fn count_rows(&self) -> u64 {
        self.row_groups
            .last()
            .map_or(0, |row_group| row_group.rows.end)
    }

    /// What the file's statistics say about each of its row groups, in
    /// file order: its row count and the bounds of the first sort column.
    /// `sort_columns` are the table columns the file is sorted by, and
    /// `key_types` their types.
    pub(crate) fn read_key_bounds(
        &self,
        table: &ParquetTable,
        sort_columns: &[usize],
        key_types: &[DataType],
    ) -> WriteResult<Vec<RowGroupKeyBounds>> {
        self.row_groups
            .iter()
            .map(|file_row_group| {
                let row_group = &table.row_groups()[file_row_group.table_index];
                Ok(RowGroupKeyBounds {
                    row_count: file_row_group.rows.end - file_row_group.rows.start,
                    first_key_bounds: sort_columns
                        .first()
                        .zip(key_types.first())
                        .map(|(&column, key_type)| {
                            SortKeyBounds::from_statistics(row_group, column, key_type)
                        })
                        .transpose()?,
                })
            })
            .collect()
    }

    /// The places, among the file's row groups, of those that hold the
    /// file rows `rows`, which must not be empty.
    fn find_row_groups_of_rows(&self, rows: &Range<u64>) -> Range<usize> {
        let first = self
            .row_groups
            .partition_point(|row_group| row_group.rows.end <= rows.start);
        let end = self
            .row_groups
            .partition_point(|row_group| row_group.rows.start < rows.end);
        first..end
    }

    /// Every file row of table column `column` before this one was released
    /// to be decompressed and decoded.
    ///
    /// A file's rows are released in file order, so this is where the
    /// released rows end in the file's first row group that is not released
    /// whole.
    pub(crate) fn find_released_rows_end(&self, column: usize) -> u64 {
        let first_partly_released = self.row_groups.partition_point(|row_group| {
            let read = &row_group.columns[column];
            read.released_rows_end == read.row_count
        });
        match self.row_groups.get(first_partly_released) {
            Some(row_group) => {
                row_group.rows.start + row_group.columns[column].released_rows_end as u64
            }
            None => self.count_rows(),
        }
    }

    /// Estimate how many more bytes will be held if every file row of
    /// table column `column` before `rows_end` is decoded (see
    /// [`FileRowGroup::estimate_bytes_to_decode`]).
    pub(crate) fn estimate_bytes_to_decode(&self, column: usize, rows_end: u64) -> usize {
        let released_rows_end = self.find_released_rows_end(column);
        if rows_end <= released_rows_end {
            return 0;
        }
        self.row_groups[self.find_row_groups_of_rows(&(released_rows_end..rows_end))]
            .iter()
            .map(|row_group| {
                let rows_end = row_group.find_rows_end_within(rows_end);
                row_group.estimate_bytes_to_decode(column, rows_end)
            })
            .sum()
    }

    /// Let every file row of table column `column` before `rows_end` be
    /// decompressed and decoded, requesting the chunks that hold them first
    /// if they were not requested. Does nothing for rows that were released
    /// before. Returns whether any row was released.
    pub(crate) fn start_decoding_rows(
        &mut self,
        column: usize,
        rows_end: u64,
        requests: &mut ReadRequests,
    ) -> bool {
        let released_rows_end = self.find_released_rows_end(column);
        if rows_end <= released_rows_end {
            return false;
        }
        let row_groups = self.find_row_groups_of_rows(&(released_rows_end..rows_end));
        for row_group in &mut self.row_groups[row_groups] {
            let rows_end = row_group.find_rows_end_within(rows_end);
            row_group.start_decoding_rows(column, rows_end, requests);
        }
        true
    }

    /// Whether the file rows `rows` of table column `column`, which must
    /// not be empty, have all arrived decoded.
    pub(crate) fn is_decoded(&self, column: usize, rows: &Range<u64>) -> bool {
        self.row_groups[self.find_row_groups_of_rows(rows)]
            .iter()
            .all(|row_group| {
                row_group.find_rows_end_within(rows.end)
                    <= row_group.columns[column].decoded_rows_end
            })
    }

    /// Append the decoded file rows `rows` of table column `column`, which
    /// must not be empty, to `sources`, in file order, as slices of the
    /// values they arrived in.
    pub(crate) fn take_input_values(
        &mut self,
        column: usize,
        rows: Range<u64>,
        sources: &mut Vec<InputValues>,
    ) {
        let row_groups = self.find_row_groups_of_rows(&rows);
        for row_group in &mut self.row_groups[row_groups] {
            let first_row = rows.start.saturating_sub(row_group.rows.start) as u32;
            let rows_end = row_group.find_rows_end_within(rows.end);
            row_group.columns[column].take_rows(first_row..rows_end, sources);
        }
    }
}
