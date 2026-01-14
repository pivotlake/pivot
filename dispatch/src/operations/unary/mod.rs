pub mod group;
pub mod parquet;

mod factory;
pub use factory::*;

pub use group::GroupFactory;

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use arrow_schema::ArrowError;
use bytes::Bytes;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

use super::Operator;
use super::channels::{Receiver, Sender};

mod pipeline_breaker;
pub use pipeline_breaker::Consumer;

mod count;
pub use count::CountFactory;

mod filter;
pub use filter::FilterFactory;

mod project;
pub use project::ProjectFactory;

mod order_by_limit;
pub use order_by_limit::{OrderBy, OrderByLimitFactory};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] super::channels::Error),
    #[error("{0}")]
    OrderByLimit(#[from] order_by_limit::Error),
    #[error("{0}")]
    Group(#[from] group::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub trait Unary<I, O> {
    fn consume<S: Sender<O>>(&mut self, object: I, sender: &mut S) -> Result<()>;
    fn run<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    fn finish<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<bool> {
        Ok(true)
    }
}

pub struct UnaryOperator<I, O, U: Unary<I, O>, R: Receiver<I>, S: Sender<O>> {
    unary: U,
    receiver: R,
    sender: S,
    notified_finished: bool,
    siblings_left: Arc<AtomicUsize>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, U: Unary<I, O>, R: Receiver<I>, S: Sender<O>> UnaryOperator<I, O, U, R, S> {
    pub fn new(unary: U, receiver: R, sender: S, siblings_left: Arc<AtomicUsize>) -> Self {
        Self {
            unary,
            receiver,
            sender,
            notified_finished: false,
            siblings_left,
            _phantom: Default::default(),
        }
    }
}

impl<I, O, U: Unary<I, O>, IN: Receiver<I>, OUT: Sender<O>> Operator
    for UnaryOperator<I, O, U, IN, OUT>
{
    fn run_cpu_work(&mut self) -> super::Result<WorkStatus> {
        let item = match self.receiver.try_recv() {
            None => return Ok(self.unary.run(&mut self.sender)?),
            Some(t) => t,
        };

        self.unary.consume(item, &mut self.sender)?;

        Ok(WorkStatus::Ran)
    }

    fn next_io_request(&mut self) -> super::Result<Option<IORequest>> {
        Ok(None)
    }

    fn process_disk_response(&mut self, _buffer: Bytes, _context: IORequest) -> super::Result<()> {
        unreachable!()
    }

    fn try_finish(&mut self) -> super::Result<bool> {
        if !self.receiver.is_empty() {
            return Ok(false);
        }

        let ready = if self.notified_finished {
            self.siblings_left.load(Ordering::Relaxed) == 0
        } else {
            self.notified_finished = true;
            self.siblings_left.fetch_sub(1, Ordering::Relaxed) == 1
        };

        if ready {
            // We need to now re-check - there may have been a race where from the time we checked we
            // were empty to now.
            if !self.receiver.is_empty() {
                // TODO: is this really what we want to do? - ALSO- DO WE NEED TO FENCE ATOMICS??
                self.notified_finished = false;
                self.siblings_left.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }

            let res = self.unary.finish(&mut self.sender)?;
            return Ok(res);
        }
        Ok(false)
    }

    fn try_steal_cpu_work(&mut self) -> super::Result<WorkStatus> {
        match self.receiver.steal() {
            Some(s) => {
                self.unary.consume(s, &mut self.sender)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }
}
