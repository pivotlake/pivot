//! The column index of one encoded leaf: the value range of each of its data
//! pages, which lets a reader skip pages by a predicate on the column.
//!
//! A leaf of one data page takes its page's range from the statistics already
//! computed for the chunk. A leaf of several pages takes each page's range
//! from the page's own values, one more pass over values that are about to be
//! encoded and compressed.
//!
//! The format requires a range for every page that holds a value, so a leaf
//! gets no column index when a page has none to give: a type without bounds,
//! or a float, whose NaNs have no place in an order. It gets none either when
//! a bound is longer than [`MAX_PAGE_BOUND_BYTES`], which keeps the index of
//! a column of long strings from growing with them.

use arrow_array::{Array, ArrayRef, Scalar};
use arrow_schema::DataType;

use crate::thrift::footer::{BoundaryOrder, ColumnIndex, Statistics};
use crate::types::table::scalar_lt;

use super::super::leaves::Leaf;
use super::super::stats;
use super::super::types::EncodedPage;

/// The longest bound a column index records for a page.
const MAX_PAGE_BOUND_BYTES: usize = 64;

/// One page's bounds as the column index stores them, and as single-value
/// arrays for ordering the pages by.
struct PageBounds {
    min: ArrayRef,
    max: ArrayRef,
    min_bytes: Vec<u8>,
    max_bytes: Vec<u8>,
}

/// Build the column index of `leaf`, encoded as `data_pages` with the chunk
/// statistics `statistics`, when every page that holds a value has bounds
/// short enough to record.
pub(super) fn build_column_index(
    leaf: &Leaf,
    statistics: &Statistics,
    data_pages: &[EncodedPage],
) -> Option<ColumnIndex> {
    if matches!(
        leaf.values.data_type(),
        DataType::Float32 | DataType::Float64
    ) {
        return None;
    }
    if let [page] = data_pages {
        return build_lone_page_column_index(statistics, page);
    }
    let mut bounds_per_page: Vec<Option<PageBounds>> = Vec::with_capacity(data_pages.len());
    let mut first_value = 0;
    for page in data_pages {
        let values = leaf.values.slice(first_value, page.value_count);
        first_value += page.value_count;
        if values.is_empty() {
            bounds_per_page.push(None);
            continue;
        }
        let (min, max) = stats::column_min_max(&values)?;
        let (min_bytes, max_bytes) = (stats::stat_bytes(&min)?, stats::stat_bytes(&max)?);
        if min_bytes.len() > MAX_PAGE_BOUND_BYTES || max_bytes.len() > MAX_PAGE_BOUND_BYTES {
            return None;
        }
        bounds_per_page.push(Some(PageBounds {
            min,
            max,
            min_bytes,
            max_bytes,
        }));
    }
    let boundary_order = find_boundary_order(&bounds_per_page);
    let null_counts = data_pages
        .iter()
        .map(|page| page.num_rows - page.value_count as i64)
        .collect();
    let null_pages = bounds_per_page.iter().map(Option::is_none).collect();
    let (min_values, max_values) = bounds_per_page
        .into_iter()
        .map(|bounds| match bounds {
            Some(bounds) => (bounds.min_bytes, bounds.max_bytes),
            None => (Vec::new(), Vec::new()),
        })
        .unzip();
    Some(ColumnIndex {
        null_pages,
        min_values,
        max_values,
        boundary_order,
        null_counts,
    })
}

/// The column index of a leaf whose only data page is `page`: the page holds
/// every value of the chunk, so its bounds are the chunk's `statistics`.
fn build_lone_page_column_index(
    statistics: &Statistics,
    page: &EncodedPage,
) -> Option<ColumnIndex> {
    let (min_bytes, max_bytes) = if page.value_count == 0 {
        (Vec::new(), Vec::new())
    } else {
        (statistics.min_value.clone()?, statistics.max_value.clone()?)
    };
    if min_bytes.len() > MAX_PAGE_BOUND_BYTES || max_bytes.len() > MAX_PAGE_BOUND_BYTES {
        return None;
    }
    Some(ColumnIndex {
        null_pages: vec![page.value_count == 0],
        min_values: vec![min_bytes],
        max_values: vec![max_bytes],
        boundary_order: BoundaryOrder::ASCENDING,
        null_counts: vec![page.num_rows - page.value_count as i64],
    })
}

/// How the bounds of the pages that hold values are ordered from one page to
/// the next: both bounds never going down, both never going up, or neither.
fn find_boundary_order(bounds_per_page: &[Option<PageBounds>]) -> BoundaryOrder {
    let bounds: Vec<&PageBounds> = bounds_per_page.iter().flatten().collect();
    let is_less =
        |a: &ArrayRef, b: &ArrayRef| scalar_lt(&Scalar::new(a.clone()), &Scalar::new(b.clone()));
    let ascending = bounds
        .windows(2)
        .all(|pair| !is_less(&pair[1].min, &pair[0].min) && !is_less(&pair[1].max, &pair[0].max));
    let descending = bounds
        .windows(2)
        .all(|pair| !is_less(&pair[0].min, &pair[1].min) && !is_less(&pair[0].max, &pair[1].max));
    if ascending {
        BoundaryOrder::ASCENDING
    } else if descending {
        BoundaryOrder::DESCENDING
    } else {
        BoundaryOrder::UNORDERED
    }
}
