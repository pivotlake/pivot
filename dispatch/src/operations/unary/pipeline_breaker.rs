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
use crate::io::{OperatorIO, ReadResponse};
use crate::memory::Spiller;
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

    /// See [`Unary::spill`]. The default spills nothing.
    fn spill(&mut self, _spiller: &mut Spiller, _count: usize) -> unary::Result<()> {
        Ok(())
    }
}

/// What one call to [`Outputter::output`] came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStatus {
    /// Every result has been emitted.
    Done,
    /// Made progress, or expects to on the next call.
    Working,
    /// Cannot make progress until reads it issued have landed, so the worker
    /// may wait for IO rather than call again at once.
    WaitingForIo,
}

/// Emits final results after the consuming phase is complete.
///
/// [`output`](Outputter::output) is called repeatedly until it returns
/// [`OutputStatus::Done`].
pub trait Outputter<O> {
    /// Emit the next batch of results, issuing through `io` whatever reads
    /// the coming results need.
    fn output(
        &mut self,
        sender: &mut dyn Sender<O>,
        io: &mut OperatorIO,
    ) -> unary::Result<OutputStatus>;

    /// Handle a completed read this outputter issued through
    /// [`output`](Self::output).
    fn process_read_response(&mut self, _response: ReadResponse) -> unary::Result<()> {
        unreachable!("an outputter that issues no reads receives no read responses")
    }

    /// See [`Unary::spill`]. An outputter still holds whatever it has not
    /// emitted yet, so it may be asked too. The default spills nothing.
    fn spill(&mut self, _spiller: &mut Spiller, _count: usize) -> unary::Result<()> {
        Ok(())
    }
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

    fn run(
        &mut self,
        sender: &mut dyn Sender<O>,
        io: &mut OperatorIO,
    ) -> unary::Result<WorkStatus> {
        match self {
            PipelineBreaker::Outputting(outputter, ..) => match outputter.output(sender, io)? {
                OutputStatus::Done => {
                    *self = PipelineBreaker::Complete;
                    Ok(WorkStatus::Ran)
                }
                OutputStatus::Working => Ok(WorkStatus::Ran),
                OutputStatus::WaitingForIo => Ok(WorkStatus::Pending),
            },
            _ => Ok(WorkStatus::Pending),
        }
    }

    fn process_read_response(
        &mut self,
        _sender: &mut dyn Sender<O>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> unary::Result<()> {
        match self {
            PipelineBreaker::Outputting(outputter, ..) => outputter.process_read_response(response),
            _ => unreachable!("only the outputting phase issues reads"),
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

    fn spill(&mut self, spiller: &mut Spiller, count: usize) -> unary::Result<()> {
        match self {
            PipelineBreaker::Consuming(consumer) => consumer.spill(spiller, count),
            PipelineBreaker::Outputting(outputter, ..) => outputter.spill(spiller, count),
            PipelineBreaker::Complete => Ok(()),
        }
    }

    fn finish(&mut self, _sender: &mut dyn Sender<O>) -> unary::Result<bool> {
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

        // Outputting needs the operator's IO handle, which finishing does not
        // have, so the outputter is driven through `run` from here on.
        match self {
            PipelineBreaker::Outputting(..) => Ok(false),
            PipelineBreaker::Complete => Ok(true),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TestOperatorIO;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Records which phase was asked to spill, and how much.
    struct Recording {
        asked: Rc<RefCell<Vec<(&'static str, usize)>>>,
    }

    impl Consumer<(), ()> for Recording {
        type Outputter = Recording;

        fn consume(&mut self, _: (), _: &mut dyn Sender<()>) -> unary::Result<()> {
            Ok(())
        }

        fn into_outputter(self) -> unary::Result<Option<Recording>> {
            Ok(Some(self))
        }

        fn spill(&mut self, _spiller: &mut Spiller, count: usize) -> unary::Result<()> {
            self.asked.borrow_mut().push(("consuming", count));
            Ok(())
        }
    }

    impl Outputter<()> for Recording {
        /// Never done, so the breaker stays in its outputting phase.
        fn output(
            &mut self,
            _: &mut dyn Sender<()>,
            _: &mut OperatorIO,
        ) -> unary::Result<OutputStatus> {
            Ok(OutputStatus::Working)
        }

        fn spill(&mut self, _spiller: &mut Spiller, count: usize) -> unary::Result<()> {
            self.asked.borrow_mut().push(("outputting", count));
            Ok(())
        }
    }

    #[test]
    fn a_spill_request_reaches_whichever_phase_holds_the_rows() {
        let asked = Rc::new(RefCell::new(Vec::new()));
        let mut breaker = PipelineBreaker::Consuming(Recording {
            asked: asked.clone(),
        });
        let mut test_io = TestOperatorIO::default();
        let mut spiller = Spiller::new(test_io.io());
        let mut sender = crate::operations::unary::test_utils::CollectSender::<()>::new();

        breaker.spill(&mut spiller, 1).unwrap();
        breaker.finish(&mut sender).unwrap();
        breaker.spill(&mut spiller, 2).unwrap();

        assert_eq!(*asked.borrow(), [("consuming", 1), ("outputting", 2)]);
    }
}
