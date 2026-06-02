//! Nullary operators: no input channel, one output sender.
//!
//! A nullary operator is a source-like stage that can generate output or perform
//! side effects without receiving upstream input. It is driven directly by the
//! worker event loop through [`Operator`].

use super::Operator;
use crate::api::{Chain, OperatorFactory};
use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use crate::operations::channels::Sender;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Channel(#[from] super::channels::Error),
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Factory for constructing a [`Nullary`] during the build step on a worker thread.
pub trait NullaryFactory<O>: Send + 'static {
    type Nullary: Nullary<O>;
    fn build_nullary(self) -> Self::Nullary;
}

/// A source-like transform that receives no input and may emit items of type `O`.
pub trait Nullary<O> {
    /// Run one unit of work, possibly sending output.
    fn run<S: Sender<O>>(&mut self, sender: &mut S) -> Result<WorkStatus>;

    /// Return any pending IO requests.
    fn next_io_requests(&mut self) -> Result<Vec<IORequest>> {
        Ok(vec![])
    }

    /// Handle a completed disk read.
    fn process_disk_response<S: Sender<O>>(
        &mut self,
        _sender: &mut S,
        _request: IORequest,
    ) -> Result<()> {
        unreachable!()
    }

    /// Try to finish. Called until it returns `true`.
    fn finish<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<bool> {
        Ok(true)
    }
}

/// Wraps a [`Nullary`] and its output sender as a concrete [`Operator`].
pub struct NullaryOperator<O, N: Nullary<O>, S: Sender<O>> {
    nullary: N,
    sender: S,
    _phantom: PhantomData<O>,
}

impl<O, N: Nullary<O>, S: Sender<O>> NullaryOperator<O, N, S> {
    pub fn new(nullary: N, sender: S) -> Self {
        Self {
            nullary,
            sender,
            _phantom: PhantomData,
        }
    }
}

impl<O, N: Nullary<O>, S: Sender<O>> Operator for NullaryOperator<O, N, S> {
    fn run_cpu_work(&mut self) -> super::Result<WorkStatus> {
        Ok(self.nullary.run(&mut self.sender)?)
    }

    fn next_io_requests(&mut self) -> super::Result<Vec<IORequest>> {
        Ok(self.nullary.next_io_requests()?)
    }

    fn process_disk_response(&mut self, request: IORequest) -> super::Result<()> {
        Ok(self
            .nullary
            .process_disk_response(&mut self.sender, request)?)
    }

    fn try_finish(&mut self) -> super::Result<bool> {
        Ok(self.nullary.finish(&mut self.sender)?)
    }
}

/// An [`OperatorFactory`] for root/source operators with no input channel.
pub struct NullaryOperatorFactory<O, NF: NullaryFactory<O>> {
    nullary_factory: NF,
    _phantom: PhantomData<fn() -> O>,
}

impl<O, NF: NullaryFactory<O>> NullaryOperatorFactory<O, NF> {
    pub fn new(nullary_factory: NF) -> Self {
        Self {
            nullary_factory,
            _phantom: PhantomData,
        }
    }
}

impl<O: 'static, NF: NullaryFactory<O>> OperatorFactory<O> for NullaryOperatorFactory<O, NF> {
    fn build<S: Sender<O> + 'static>(self: Box<Self>, sender: S) -> Chain {
        Chain::root(Box::new(NullaryOperator::new(
            self.nullary_factory.build_nullary(),
            sender,
        )))
    }
}

/// Runs a `FnOnce() -> O` exactly once on the worker it lands on, sends the
/// result downstream, and finishes.
///
/// Used to plumb one-shot setup work (e.g. building a `ParquetTable`) into a
/// worker thread that has a `MemoryContext`, so the caller doesn't have to
/// have one. See [`Dispatch::run_on_worker`](crate::Dispatch::run_on_worker).
pub struct OneShotNullaryFactory<O, F>
where
    F: FnOnce() -> O + Send + 'static,
{
    func: F,
    _phantom: PhantomData<fn() -> O>,
}

impl<O, F> OneShotNullaryFactory<O, F>
where
    F: FnOnce() -> O + Send + 'static,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

impl<O, F> NullaryFactory<O> for OneShotNullaryFactory<O, F>
where
    O: Send + 'static,
    F: FnOnce() -> O + Send + 'static,
{
    type Nullary = OneShotNullary<O, F>;

    fn build_nullary(self) -> Self::Nullary {
        OneShotNullary {
            func: Some(self.func),
            _phantom: PhantomData,
        }
    }
}

pub struct OneShotNullary<O, F: FnOnce() -> O> {
    func: Option<F>,
    _phantom: PhantomData<fn() -> O>,
}

impl<O, F: FnOnce() -> O + Send> Nullary<O> for OneShotNullary<O, F> {
    fn run<S: Sender<O>>(&mut self, sender: &mut S) -> Result<WorkStatus> {
        match self.func.take() {
            Some(f) => {
                sender.send(f())?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }

    fn finish<S: Sender<O>>(&mut self, _sender: &mut S) -> Result<bool> {
        // We're done once `run` has consumed the closure.
        Ok(self.func.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::Operator;
    use crate::operations::unary::test_utils::CollectSender;

    #[derive(Default)]
    struct EmitOne {
        emitted: bool,
    }

    impl Nullary<i32> for EmitOne {
        fn run<S: Sender<i32>>(&mut self, sender: &mut S) -> Result<WorkStatus> {
            if self.emitted {
                return Ok(WorkStatus::Pending);
            }
            self.emitted = true;
            sender.send(7)?;
            Ok(WorkStatus::Ran)
        }

        fn finish<S: Sender<i32>>(&mut self, _sender: &mut S) -> Result<bool> {
            Ok(self.emitted)
        }
    }

    #[test]
    fn nullary_operator_emits_once() {
        let mut operator = NullaryOperator::new(EmitOne::default(), CollectSender::<i32>::new());

        assert!(matches!(operator.run_cpu_work().unwrap(), WorkStatus::Ran));
        assert_eq!(operator.sender.items, vec![7]);
        assert!(matches!(
            operator.run_cpu_work().unwrap(),
            WorkStatus::Pending
        ));
        assert!(operator.try_finish().unwrap());
    }
}
