//! What a table scan reports to a query's live stats about its row groups:
//! which ones it read, which it found in memory, and which it skipped and why.
//!
//! Each is a custom stat whose name is one of the constants below and whose
//! value names the row group (see
//! [`RowGroupMetadata::describe_for_live_stats`](crate::RowGroupMetadata::describe_for_live_stats)).
//! All but the dictionary one are reported as the fetcher's: by the fetcher
//! itself, by the row-group injector that feeds it, and by the code that
//! builds the scan. The range cutter reports the dictionary one as its own.

/// The fetcher asked for the row group's projected columns, because the
/// decompressed cache did not hold them all. The value also holds `bytes`,
/// how many it asked for. The compressed cache may serve some of those from
/// memory; the operator's IO counters say what storage was really read.
pub const ROW_GROUP_READ: &str = "row_group_read";

/// Every projected column of the row group was already in memory, so the
/// fetcher read nothing for it.
pub const ROW_GROUP_FROM_MEMORY: &str = "row_group_from_memory";

/// The statistics in the file's footer proved no row of the row group can
/// match a pushed-down predicate, so the scan was planned without it.
pub const ROW_GROUP_PRUNED_BY_STATISTICS: &str = "row_group_pruned_by_statistics";

/// The scan's dynamic filter (a Top-N's boundary or a join's key bounds)
/// proved no row of the row group can match, when a worker came to claim it.
pub const ROW_GROUP_PRUNED_BY_DYNAMIC_FILTER: &str = "row_group_pruned_by_dynamic_filter";

/// A column's dictionary does not hold a pushed-down equality constant, so
/// the row group's pages were dropped after its read.
pub const ROW_GROUP_PRUNED_BY_DICTIONARY: &str = "row_group_pruned_by_dictionary";
