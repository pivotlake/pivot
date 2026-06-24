//! An inert operator used to retire a node in place.

use crate::data_flow::WorkStatus;
use crate::io::{FsRequest, HttpRequest};
use crate::operations::{FinishStatus, Operator, Result};

/// A do-nothing operator that a node's real operator is replaced with when it is
/// abandoned (e.g. the scan upstream of a satisfied `LIMIT`).
///
/// Swapping the real operator out drops it, and with it any batches buffered in
/// its channels, freeing that memory at once. Leaving an inert node in its
/// place, rather than restructuring the graph, keeps every edge index stable:
/// traversals just run these trivial methods and move on, and any late IO
/// completion routed to the node lands here harmlessly. It reports `Done` from
/// `try_finish` so the surrounding finish protocol treats it as already
/// complete.
pub struct AbandonedOperator;

impl Operator for AbandonedOperator {
    fn run_cpu_work(&mut self) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    fn next_fs_requests(&mut self) -> Result<Vec<FsRequest>> {
        Ok(vec![])
    }

    fn process_fs_response(&mut self, _request: FsRequest) -> Result<()> {
        Ok(())
    }

    fn process_http_response(&mut self, _request: HttpRequest) -> Result<()> {
        Ok(())
    }

    fn try_finish(&mut self) -> Result<FinishStatus> {
        Ok(FinishStatus::Done)
    }
}
