//! A closure-backed fold stage that gathers every worker's output on one worker.

use super::{Result, Unary, UnaryFactory};
use crate::operations::channels::Sender;

struct Fold<State, Consume, Finish> {
    state: State,
    consume: Consume,
    finish: Finish,
}

pub struct FanInFactory<State, Consume, Finish> {
    fold: Option<Fold<State, Consume, Finish>>,
}

impl<State, Consume, Finish> FanInFactory<State, Consume, Finish> {
    /// Build the sole receiving worker's stateful fold.
    pub(crate) fn new(state: State, consume: Consume, finish: Finish) -> Self {
        Self {
            fold: Some(Fold {
                state,
                consume,
                finish,
            }),
        }
    }

    /// Build a non-receiving worker placeholder. Its channel endpoint forwards
    /// upstream items to the worker that owns the fold.
    pub(crate) fn empty() -> Self {
        Self { fold: None }
    }
}

impl<Input, Output, State, Consume, Finish> UnaryFactory<Input, Output>
    for FanInFactory<State, Consume, Finish>
where
    State: Send + 'static,
    Consume: FnMut(&mut State, Input, &mut dyn Sender<Output>) -> Result<()> + Send + 'static,
    Finish: FnOnce(State, &mut dyn Sender<Output>) -> Result<()> + Send + 'static,
{
    type Unary = FanIn<State, Consume, Finish>;

    fn build_unary(self) -> Self::Unary {
        FanIn { fold: self.fold }
    }
}

pub struct FanIn<State, Consume, Finish> {
    fold: Option<Fold<State, Consume, Finish>>,
}

impl<Input, Output, State, Consume, Finish> Unary<Input, Output> for FanIn<State, Consume, Finish>
where
    State: Send + 'static,
    Consume: FnMut(&mut State, Input, &mut dyn Sender<Output>) -> Result<()> + Send + 'static,
    Finish: FnOnce(State, &mut dyn Sender<Output>) -> Result<()> + Send + 'static,
{
    fn consume<SenderType: Sender<Output>>(
        &mut self,
        item: Input,
        sender: &mut SenderType,
    ) -> Result<()> {
        let fold = self
            .fold
            .as_mut()
            .expect("the fan-in receiver must own the fold state");
        (fold.consume)(&mut fold.state, item, sender)
    }

    fn finish<SenderType: Sender<Output>>(&mut self, sender: &mut SenderType) -> Result<bool> {
        if let Some(fold) = self.fold.take() {
            (fold.finish)(fold.state, sender)?;
        }
        Ok(true)
    }
}
