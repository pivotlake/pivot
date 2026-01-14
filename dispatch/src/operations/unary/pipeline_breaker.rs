use crate::data_flow::WorkStatus;
use crate::operations::{Sender, Unary, unary};
use std::marker::PhantomData;
use std::mem;
use tracing::debug;

pub trait Consumer<I, O> {
    type Outputter: Outputter<O>;
    fn consume<S: Sender<O>>(&mut self, object: I, sender: &mut S) -> unary::Result<()>;

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>>;
}

pub trait Outputter<O> {
    fn output<S: Sender<O>>(&mut self, sender: &mut S) -> unary::Result<bool>;
}

pub enum PipelineBreaker<I, O, C: Consumer<I, O>> {
    Consuming(C),
    Outputting(C::Outputter, PhantomData<(I, O)>),
    Complete,
}

impl<I, O, C: Consumer<I, O>> Unary<I, O> for PipelineBreaker<I, O, C> {
    fn consume<S: Sender<O>>(&mut self, object: I, sender: &mut S) -> unary::Result<()> {
        match self {
            PipelineBreaker::Consuming(c) => c.consume(object, sender),
            _ => panic!("Consume called after outting began"),
        }
    }

    fn run<S: Sender<O>>(&mut self, sender: &mut S) -> unary::Result<WorkStatus> {
        match self {
            PipelineBreaker::Outputting(o, ..) => {
                if o.output(sender)? {
                    *self = PipelineBreaker::Complete
                }
                Ok(WorkStatus::Ran)
            }
            _ => Ok(WorkStatus::Pending),
        }
    }

    fn finish<S: Sender<O>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if matches!(self, PipelineBreaker::Consuming(..)) {
            let consume = mem::replace(self, PipelineBreaker::Complete);
            match consume {
                PipelineBreaker::Consuming(c) => {
                    *self = match c.into_outputter()? {
                        Some(o) => PipelineBreaker::Outputting(o, PhantomData::default()),
                        None => PipelineBreaker::Complete,
                    }
                }
                _ => unreachable!(),
            }
        }

        match self {
            PipelineBreaker::Outputting(o, ..) => {
                debug!("Trying to output");
                if o.output(sender)? {
                    *self = PipelineBreaker::Complete;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            PipelineBreaker::Complete => Ok(true),
            _ => unreachable!(),
        }
    }
}
