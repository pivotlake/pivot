use crate::data_flow::WorkStatus;
use crate::io::cache::CACHE;
use crate::io::{IOLocation, IORequest};
use crate::operations::{Operator, Sender};
use crate::table::source::TableSource;
use crate::table::{Projection, RowGroupMetadataHandle, Table};
use bytes::Bytes;
use parquet::file::reader::{ChunkReader, Length};
use std::io::Cursor;
use std::os::fd::AsRawFd;
use std::sync::Arc;

pub struct RowGroupBuffer {
    pub metadata: RowGroupMetadataHandle,
    pub bytes: Bytes,
    pub base_offset: u64,
    pub filtered_indices: Option<Vec<u32>>,
}

impl Length for RowGroupBuffer {
    fn len(&self) -> u64 {
        // Return the virtual file length so offset validation passes
        self.base_offset + self.bytes.len() as u64
    }
}

impl ChunkReader for RowGroupBuffer {
    type T = Cursor<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        // Translate absolute file offset to relative buffer offset
        let relative_start = (start - self.base_offset) as usize;
        let bytes = self.bytes.slice(relative_start..);
        Ok(Cursor::new(bytes))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        // Translate absolute file offset to relative buffer offset
        let relative_start = (start - self.base_offset) as usize;
        Ok(self.bytes.slice(relative_start..relative_start + length))
    }
}

#[derive(Debug)]
pub struct RowGroupRequest {
    metadata: RowGroupMetadataHandle,
    location: IOLocation,
}

pub struct TableInput<O: Sender<RowGroupBuffer>> {
    source: Arc<TableSource>,
    table: Arc<Table>,
    projection: Option<Projection>,
    tx: O,
    current_row_group: Option<RowGroupRequest>,
    waiting: usize,
}

impl<O: Sender<RowGroupBuffer>> TableInput<O> {
    pub fn new(
        tx: O,
        source: Arc<TableSource>,
        table: Arc<Table>,
        projection: Option<Projection>,
    ) -> Self {
        Self {
            tx,
            source,
            table,
            projection,
            current_row_group: None,
            waiting: 0,
        }
    }

    fn set_current_row_group_if_not_set(&mut self) {
        if self.current_row_group.is_some() {
            return;
        }

        if let Some(metadata_handle) = self.source.pop_row_group() {
            let metadata = &self.table.row_groups[metadata_handle.row_group_index];
            let parquet_row_group_metadata = metadata
                .arrow_metadata
                .metadata()
                .row_group(metadata.row_group);
            let (offset, length) = calculate_projected_row_group_range(
                parquet_row_group_metadata,
                self.projection.as_ref(),
            );
            self.current_row_group = Some(RowGroupRequest {
                metadata: metadata_handle,
                location: IOLocation {
                    raw_fd: metadata.file.as_raw_fd(),
                    offset,
                    size: length,
                },
            });
        }
    }
}

impl<O: Sender<RowGroupBuffer>> Operator for TableInput<O> {
    fn run_cpu_work(&mut self) -> crate::operations::Result<WorkStatus> {
        self.set_current_row_group_if_not_set();
        if let Some(rg) = self.current_row_group.as_ref()
            && let Some(bytes) = CACHE.get(&rg.location)
        {
            let row_group_request = self.current_row_group.take().unwrap();
            self.tx.send(RowGroupBuffer {
                metadata: row_group_request.metadata,
                bytes,
                base_offset: row_group_request.location.offset as u64,
                filtered_indices: None,
            })?;
            return Ok(WorkStatus::Ran);
        }
        Ok(WorkStatus::Pending)
    }

    fn next_io_request(&mut self) -> crate::operations::Result<Option<IORequest>> {
        self.set_current_row_group_if_not_set();
        // We don't want to fetch it if already in cache; let's wait till we're asked for CPU work
        if let Some(rg) = self.current_row_group.as_ref()
            && !CACHE.contains(&rg.location)
        {
            let row_group_request = self.current_row_group.take().unwrap();
            self.waiting += 1;
            return Ok(Some(IORequest {
                location: row_group_request.location,
                ctx: Box::new(row_group_request.metadata),
            }));
        }
        Ok(None)
    }

    fn process_disk_response(
        &mut self,
        bytes: Bytes,
        request: IORequest,
    ) -> crate::operations::Result<()> {
        self.waiting -= 1;
        self.tx.send(RowGroupBuffer {
            metadata: *request.ctx.downcast::<RowGroupMetadataHandle>().unwrap(),
            bytes,
            base_offset: request.location.offset as u64,
            filtered_indices: None,
        })?;
        Ok(())
    }

    fn try_finish(&mut self) -> crate::operations::Result<bool> {
        Ok(self.waiting == 0 && self.current_row_group.is_none() && self.source.is_empty())
    }
}

/// Calculate the byte range for a row group with projection
pub(crate) fn calculate_projected_row_group_range(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    projection: Option<&Projection>,
) -> (usize, usize) {
    let mut min_offset = u64::MAX;
    let mut max_end = 0u64;

    for (col_idx, col) in row_group.columns().iter().enumerate() {
        // Skip columns not in the projection
        if let Some(p) = projection
            && !p.includes(col_idx)
        {
            continue;
        }

        // Use data_page_offset, falling back to dictionary_page_offset if present
        let col_start = if let Some(dict_offset) = col.dictionary_page_offset() {
            dict_offset as u64
        } else {
            col.data_page_offset() as u64
        };

        let col_end = col_start + col.compressed_size() as u64;

        min_offset = min_offset.min(col_start);
        max_end = max_end.max(col_end);
    }

    // Handle empty projection edge case
    if min_offset == u64::MAX {
        return (0, 0);
    }

    (min_offset as usize, (max_end - min_offset) as usize)
}
