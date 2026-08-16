//! Pipeline breakers: operators that must see all input before producing output.
//!
//! A normal unary operator (like filter or project) can emit output for each input
//! item immediately. A pipeline breaker (like group-by) must
//! accumulate all input first, then emit results in a separate output phase.
//!
//! [`PipelineBreaker`] implements [`Unary`] as a state machine with three phases:
//!
//! 1. **Consuming** — Input items arrive via [`consume`](Unary::consume) and are
//!    accumulated by the [`Consumer`] (e.g. inserting into a hash table or heap).
//!
//! 2. **Outputting** — After all siblings have finished and [`finish`](Unary::finish)
//!    is called, the consumer is converted into an [`Outputter`] which emits results
//!    one batch at a time. The worker's event loop calls [`run`](Unary::run) repeatedly
//!    to drain the outputter.
//!
//! 3. **Complete** — The outputter has emitted everything.

use crate::data_flow::WorkStatus;
use crate::operations::channels::Sender;
use crate::operations::{Unary, unary};
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Accumulates input items during the consuming phase.
///
/// Once all input has been consumed (all siblings drained), [`into_outputter`](Consumer::into_outputter)
/// converts the accumulated state into an [`Outputter`] that emits final results.
/// Returns `None` if there is nothing to output (e.g. empty input).
pub trait Consumer<I, O> {
    type Outputter: Outputter<O>;

    /// Process one input item, optionally sending intermediate results to `sender`.
    fn consume(&mut self, object: I, sender: &mut dyn Sender<O>) -> unary::Result<()>;

    /// Transition from consuming to outputting. Called once after all input is drained.
    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>>;

    /// Whether this consumer can still use more input. Default `true`; a
    /// consumer that fills up early (e.g. a satisfied `LIMIT`) returns `false`
    /// to stop being fed. See [`Unary::ready_for_more_work`].
    fn ready_for_more_work(&mut self) -> bool {
        true
    }

    /// Whether this consumer is done consuming before its input drains.
    /// See [`Unary::finished_consuming`].
    fn finished_consuming(&self) -> bool {
        false
    }

    /// See [`Operator::upstream_cancel_flag`](crate::operations::Operator::upstream_cancel_flag).
    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        None
    }
}

/// Emits final results after the consuming phase is complete.
///
/// [`output`](Outputter::output) is called repeatedly until it returns `true`,
/// indicating all results have been emitted.
pub trait Outputter<O> {
    /// Emit the next batch of results. Returns `true` when done.
    fn output(&mut self, sender: &mut dyn Sender<O>) -> unary::Result<bool>;
}

/// A [`Unary`] implementation that breaks the pipeline into consume and output phases.
///
/// Used by operators that need all input before they can produce output (aggregations,
/// sorts). See the [module docs](self) for the state machine description.
pub enum PipelineBreaker<I, O, C: Consumer<I, O>> {
    /// Accumulating input.
    Consuming(C),
    /// Emitting final results.
    Outputting(C::Outputter, PhantomData<(I, O)>),
    /// All results emitted.
    Complete,
}

impl<I, O, C: Consumer<I, O>> Unary<I, O> for PipelineBreaker<I, O, C> {
    fn consume(
        &mut self,
        object: I,
        sender: &mut dyn Sender<O>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        match self {
            PipelineBreaker::Consuming(c) => c.consume(object, sender),
            // A breaker that finished *early* (e.g. a satisfied `LIMIT`) can still
            // have input sitting in its channel when it transitions to output:
            // it stopped consuming before draining, and its upstream is being
            // abandoned. That input is surplus to the result already decided, so
            // drop it. (A breaker that finishes the normal way drains first, so
            // it never reaches here.)
            _ => Ok(()),
        }
    }

    fn run(&mut self, sender: &mut dyn Sender<O>) -> unary::Result<WorkStatus> {
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

    fn ready_for_more_work(&mut self) -> bool {
        match self {
            PipelineBreaker::Consuming(c) => c.ready_for_more_work(),
            // Past the consuming phase, `ready_for_more_work` instead gates
            // whether the worker calls `run` to drain the outputter, so it
            // must stay `true` here, or the output would never be emitted.
            _ => true,
        }
    }

    fn finished_consuming(&self) -> bool {
        match self {
            PipelineBreaker::Consuming(c) => c.finished_consuming(),
            _ => true,
        }
    }

    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        match self {
            PipelineBreaker::Consuming(c) => c.upstream_cancel_flag(),
            _ => None,
        }
    }

    fn finish(&mut self, sender: &mut dyn Sender<O>) -> unary::Result<bool> {
        if matches!(self, PipelineBreaker::Consuming(..)) {
            let consume = mem::replace(self, PipelineBreaker::Complete);
            match consume {
                PipelineBreaker::Consuming(c) => {
                    *self = match c.into_outputter()? {
                        Some(o) => PipelineBreaker::Outputting(o, PhantomData),
                        None => PipelineBreaker::Complete,
                    }
                }
                _ => unreachable!(),
            }
        }

        match self {
            PipelineBreaker::Outputting(o, ..) => {
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
