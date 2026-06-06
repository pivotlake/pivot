//! Cuts one row group into ~1 MiB pages.
//!
//! A streaming 1→many operator: each column is split into [`PageJob`]s, each
//! tagged with the row group's id and total page count. A page is a list of
//! zero-copy slices into the original batch arrays (it may span a batch
//! boundary), so cutting copies nothing.

use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, StringViewArray};
use arrow_schema::DataType;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use super::types::{PageJob, PipePageJob, RowGroupBatch};

/// Target uncompressed size of one data page. Matches Parquet's usual ~1 MiB
/// data page size: large enough to amortize per-page overhead and compress
/// well, small enough to be a useful unit of parallel work.
const TARGET_PAGE_SIZE: usize = 1024 * 1024;

pub(super) type PagePlannerFactory = DefaultUnaryFactory<PagePlanner>;

pub(super) fn factories(worker_count: usize) -> Vec<PagePlannerFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

#[derive(Default)]
pub(super) struct PagePlanner;

impl Unary<RowGroupBatch, PipePageJob> for PagePlanner {
    fn consume<S: Sender<PipePageJob>>(
        &mut self,
        rg: RowGroupBatch,
        sender: &mut S,
    ) -> UnaryResult<()> {
        let jobs = build_page_jobs(&rg.batches);
        let n_pages = jobs.len();
        for job in jobs {
            sender.send(PipePageJob {
                rg_id: rg.rg_id,
                dest_worker: rg.dest_worker,
                n_pages,
                schema: rg.schema.clone(),
                job,
            })?;
        }
        Ok(())
    }
}

/// Cut one row group's batches into ~1 MiB page jobs. Each column's values are
/// walked in row order and split into pages; pages reference the original
/// arrays by zero-copy slice.
pub(super) fn build_page_jobs(batches: &[RecordBatch]) -> Vec<PageJob> {
    let num_columns = batches[0].num_columns();
    let mut jobs = Vec::new();
    for column in 0..num_columns {
        let arrays: Vec<ArrayRef> = batches.iter().map(|b| b.column(column).clone()).collect();
        let ranges = page_ranges(&arrays, arrays[0].data_type(), TARGET_PAGE_SIZE);
        for (page_index, (start, len)) in ranges.into_iter().enumerate() {
            jobs.push(PageJob {
                column,
                page_index,
                num_rows: len as i64,
                pieces: slice_range(&arrays, start, len),
            });
        }
    }
    jobs
}

/// Split a column (given as its per-batch arrays) into contiguous `(start,
/// len)` row ranges over the logical concatenation, each ~`target` uncompressed
/// bytes when PLAIN-encoded.
fn page_ranges(arrays: &[ArrayRef], data_type: &DataType, target: usize) -> Vec<(usize, usize)> {
    let total: usize = arrays.iter().map(|a| a.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    match plain_fixed_width(data_type) {
        Some(width) => fixed_ranges(total, (target / width).max(1)),
        None => variable_ranges(arrays, target),
    }
}

fn fixed_ranges(len: usize, rows_per_page: usize) -> Vec<(usize, usize)> {
    (0..len)
        .step_by(rows_per_page)
        .map(|start| (start, rows_per_page.min(len - start)))
        .collect()
}

/// Walk a BYTE_ARRAY column's values across its arrays, cutting a page once the
/// accumulated PLAIN size (4-byte length prefix + bytes per value) reaches
/// `target`. Always at least one row per page.
fn variable_ranges(arrays: &[ArrayRef], target: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut page_start = 0;
    let mut row = 0;
    let mut acc = 0usize;
    for array in arrays {
        for_each_value_len(array.as_ref(), |value_len| {
            acc += 4 + value_len;
            row += 1;
            if acc >= target {
                ranges.push((page_start, row - page_start));
                page_start = row;
                acc = 0;
            }
        });
    }
    if page_start < row {
        ranges.push((page_start, row - page_start));
    }
    ranges
}

/// Map a `(start, len)` row range over the logical concatenation of `arrays`
/// to the zero-copy slices that cover it.
fn slice_range(arrays: &[ArrayRef], start: usize, len: usize) -> Vec<ArrayRef> {
    let end = start + len;
    let mut pieces = Vec::new();
    let mut cursor = 0; // global index of the current array's first row
    for array in arrays {
        let (array_start, array_end) = (cursor, cursor + array.len());
        cursor = array_end;
        let lo = start.max(array_start);
        let hi = end.min(array_end);
        if lo < hi {
            pieces.push(array.slice(lo - array_start, hi - lo));
        }
    }
    pieces
}

/// PLAIN byte size of one fixed-width value, or `None` for variable-width types.
fn plain_fixed_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Int32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::Float64 => Some(8),
        _ => None,
    }
}

/// Call `f` with each value's byte length (for BYTE_ARRAY size accounting).
fn for_each_value_len(array: &dyn Array, mut f: impl FnMut(usize)) {
    match array.data_type() {
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            (0..a.len()).for_each(|i| f(a.value(i).len()));
        }
        DataType::Utf8View => {
            let a = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            (0..a.len()).for_each(|i| f(a.value(i).len()));
        }
        // Unsupported here; the encoder rejects it during encoding.
        _ => (0..array.len()).for_each(|_| f(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use std::sync::Arc;

    fn arrays<const N: usize>(arrs: [ArrayRef; N]) -> Vec<ArrayRef> {
        arrs.into_iter().collect()
    }

    #[test]
    fn fixed_width_pages_split_by_row_count() {
        let column = arrays([Arc::new(Int64Array::from((0..250).collect::<Vec<i64>>()))]);

        // 8 bytes/value, target 800 bytes => 100 rows/page => 100, 100, 50.
        let ranges = page_ranges(&column, &DataType::Int64, 800);

        assert_eq!(ranges, vec![(0, 100), (100, 100), (200, 50)]);
    }

    #[test]
    fn variable_width_pages_split_by_byte_size() {
        // Each value encodes as 4 + 6 = 10 bytes ("abcdef").
        let column = arrays([Arc::new(StringArray::from(vec!["abcdef"; 10]))]);

        // Target 25 bytes => cut after 3 values (30 >= 25): 3, 3, 3, 1.
        let ranges = page_ranges(&column, &DataType::Utf8, 25);

        assert_eq!(ranges, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
    }

    #[test]
    fn pages_and_slices_span_batch_boundaries() {
        // Three batches of 3 rows; 8 bytes/value, target 16 bytes => 2 rows/page,
        // so page boundaries fall mid-batch.
        let column = arrays([
            Arc::new(Int64Array::from(vec![0, 1, 2])),
            Arc::new(Int64Array::from(vec![3, 4, 5])),
            Arc::new(Int64Array::from(vec![6, 7, 8])),
        ]);

        let ranges = page_ranges(&column, &DataType::Int64, 16);
        let second_page = slice_range(&column, ranges[1].0, ranges[1].1);

        // The second page (rows 2..4) straddles batches 0 and 1: tail of batch 0
        // (row 2) + head of batch 1 (row 3).
        assert_eq!(ranges, vec![(0, 2), (2, 2), (4, 2), (6, 2), (8, 1)]);
        assert_eq!(
            second_page.iter().map(|a| a.len()).collect::<Vec<_>>(),
            vec![1, 1]
        );
    }
}
