//! The rows of one column of an input row group as they come out of the
//! decoder.

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_data::ArrayData;
use arrow_schema::DataType;

use crate::ParquetTable;
use crate::reading::record_batch_metadata::{global_row_group, row_index};
use crate::writing::error::{WriteError, WriteResult};

/// The decoded values of one column in consecutive rows of one input row
/// group.
pub(super) struct DecodedValues {
    /// The index of the row group in the table's list of row groups.
    pub(super) row_group: usize,
    /// The table column the values are of.
    pub(super) column: usize,
    /// The row number, in that row group, of the first value.
    pub(super) first_row: u32,
    pub(super) values: ArrayRef,
}

impl DecodedValues {
    /// Read `batch` as the decoder emitted it for a read of one column of a
    /// row group of `table`.
    ///
    /// The batch holds the column the decoder decoded, followed by two
    /// columns that say which row group the rows are in and which rows of
    /// it they are. The decoder does not say which column it decoded: that
    /// is found from the column's name.
    pub(super) fn read_decoder_output(
        batch: RecordBatch,
        table: &ParquetTable,
    ) -> WriteResult<Self> {
        let row_group = global_row_group(&batch)
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or(WriteError::Downcast {
                expected: "UInt32Array",
            })?
            .value(0) as usize;
        let row_numbers = row_index(&batch);
        let first_row = row_numbers.value(0);
        // The compaction reads every row of a row group and refuses a table
        // with deleted rows, so a batch's rows follow one another.
        if row_numbers.value(row_numbers.len() - 1) != first_row + row_numbers.len() as u32 - 1 {
            return Err(WriteError::CompactionRowCountMismatch);
        }
        let column = table.row_groups()[row_group]
            .schema
            .index_of(batch.schema().field(0).name())?;
        Ok(Self {
            row_group,
            column,
            first_row,
            values: batch.column(0).clone(),
        })
    }

    /// How many bytes the values hold in buffers of their own.
    ///
    /// A string or binary column is decoded as views that point into the
    /// decompressed pages and the dictionary. Those bytes are not the
    /// values' own: they are counted with the pages. Only the views and the
    /// validity are counted here.
    pub(super) fn count_owned_bytes(&self) -> usize {
        Self::count_owned_bytes_of(&self.values.to_data())
    }

    /// How many bytes `array` and its children hold in buffers of their
    /// own.
    fn count_owned_bytes_of(array: &ArrayData) -> usize {
        let buffers = array.buffers();
        let owned_buffers = match array.data_type() {
            // The first buffer holds the views. The others are the pages
            // and dictionary blocks the views point into.
            DataType::Utf8View | DataType::BinaryView => &buffers[..1],
            _ => buffers,
        };
        let owned_bytes: usize = owned_buffers.iter().map(|buffer| buffer.capacity()).sum();
        let validity_bytes = array.nulls().map_or(0, |nulls| nulls.buffer().capacity());
        let child_bytes: usize = array
            .child_data()
            .iter()
            .map(Self::count_owned_bytes_of)
            .sum();
        owned_bytes + validity_bytes + child_bytes
    }
}
