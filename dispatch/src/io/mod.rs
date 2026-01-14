use crate::identified::Identifier;
use std::any::Any;
use std::fmt::Debug;
use std::os::fd::RawFd;

mod requester;
pub use requester::{Error as IORequesterError, IORequester};
pub mod backend;
pub mod cache;
mod disk_buffer;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct IOLocation {
    pub raw_fd: RawFd,
    pub offset: usize,
    pub size: usize,
}

pub struct IORequest {
    pub location: IOLocation,
    pub ctx: Box<dyn Any>,
}

pub struct PipelineRequest {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
    pub request: IORequest,
}

impl PipelineRequest {
    pub fn new(pipeline_id: Identifier, operator_idx: Identifier, request: IORequest) -> Self {
        Self {
            data_flow_id: pipeline_id,
            operator_idx,
            request,
        }
    }
}
