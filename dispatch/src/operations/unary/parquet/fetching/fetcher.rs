//! The [`RowGroupFetcher`] unary operator that drives disk IO for a single
//! row group at a time.
//!
//! It accepts a [`RowGroupRequest`], yields the aligned [`IORequest`]s for each
//! projected column, and as completions arrive it fills in the corresponding
//! buffer slots. Once every column chunk has been read, it emits a
//! [`RowGroupBuffer`] downstream for decompression/decoding.

use crate::io::IORequest;
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary::parquet::types::requests::RowGroupBuffer;
use crate::operations::unary::parquet::types::requests::RowGroupRequest;

/// Reads column chunks for one row group at a time via async disk IO.
///
/// Processes at most one [`RowGroupRequest`] concurrently:
/// 1. `consume` — stores the request (which already contains pending IO
///    requests for cache-missed blocks).
/// 2. `next_io_requests` — hands the pending reads to the IO scheduler.
/// 3. `process_disk_response` — slots completed buffers into the right column.
/// 4. Once all buffers arrive (`complete()`), the finished [`RowGroupBuffer`]
///    is sent downstream.
#[derive(Default)]
pub struct RowGroupFetcher {
    /// The in-progress row group, or `None` when idle.
    row_group_request: Option<RowGroupRequest>,
}

impl RowGroupFetcher {
    /// If the current row group's IO is fully satisfied, convert it to a
    /// [`RowGroupBuffer`] and send it downstream.
    fn send_out_buffer_if_complete<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
        if let Some(r) = self.row_group_request.as_ref()
            && r.complete()
        {
            let row_group = self.row_group_request.take().unwrap();
            let buffer = row_group.into_row_group_buffer();
            sender.send(buffer)?;
        }
        Ok(())
    }
}

impl Unary<RowGroupRequest, RowGroupBuffer> for RowGroupFetcher {
    fn consume<S: Sender<RowGroupBuffer>>(
        &mut self,
        request: RowGroupRequest,
        sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
        assert!(self.row_group_request.is_none());
        self.row_group_request = Some(request);
        self.send_out_buffer_if_complete(sender)?;
        Ok(())
    }

    fn next_io_requests(&mut self) -> crate::operations::unary::Result<Vec<IORequest>> {
        if let Some(rg) = self.row_group_request.as_mut()
            && !rg.pending_io().is_empty()
        {
            return Ok(std::mem::take(rg.pending_io()));
        }
        Ok(vec![])
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.row_group_request.is_none()
    }

    fn process_disk_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        _request: IORequest,
    ) -> crate::operations::unary::Result<()> {
        // The requester already committed this block's bytes into the cache
        // slot; we just count it off and emit once every block has landed.
        self.row_group_request.as_mut().unwrap().complete_one();
        self.send_out_buffer_if_complete(sender)?;
        Ok(())
    }

    fn finish<S: Sender<RowGroupBuffer>>(
        &mut self,
        _sender: &mut S,
    ) -> crate::operations::unary::Result<bool> {
        Ok(self.row_group_request.is_none())
    }
}
