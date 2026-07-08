//! The closure-backed fold stage behind
//! [`OperatorSpec::fan_in`](crate::OperatorSpec::fan_in): paired with a fan-in
//! channel, one worker receives every item and folds it into a caller-supplied
//! state with a `consume` closure, then a `finish` closure consumes the state
//! once every sibling has drained. The other workers' operators carry no fold,
//! receive nothing, and emit nothing.

use super::{Result, Unary, UnaryFactory};
use crate::operations::channels::Sender;

/// One worker's fold: the running state plus the two closures driving it.
struct Fold<St, C, F> {
    state: St,
    consume: C,
    finish: F,
}

/// Per-worker factory for [`FanIn`]. Only the receiving worker's carries the
/// fold.
pub struct FanInFactory<St, C, F> {
    fold: Option<Fold<St, C, F>>,
}

impl<St, C, F> FanInFactory<St, C, F> {
    /// The receiving worker's factory (`state` + closures); every other worker
    /// gets [`empty`](Self::empty).
    pub(crate) fn new(state: St, consume: C, finish: F) -> Self {
        Self {
            fold: Some(Fold {
                state,
                consume,
                finish,
            }),
        }
    }

    /// A no-op sibling: its receiver is forever empty and its `finish` emits
    /// nothing.
    pub(crate) fn empty() -> Self {
        Self { fold: None }
    }
}

impl<I, O, St, C, F> UnaryFactory<I, O> for FanInFactory<St, C, F>
where
    St: Send + 'static,
    C: FnMut(&mut St, I, &mut dyn Sender<O>) -> Result<()> + Send + 'static,
    F: FnOnce(St, &mut dyn Sender<O>) -> Result<()> + Send + 'static,
{
    type Unary = FanIn<St, C, F>;

    fn build_unary(self) -> FanIn<St, C, F> {
        FanIn { fold: self.fold }
    }
}

/// The fold operator itself; see the module doc.
pub struct FanIn<St, C, F> {
    fold: Option<Fold<St, C, F>>,
}

impl<I, O, St, C, F> Unary<I, O> for FanIn<St, C, F>
where
    St: Send + 'static,
    C: FnMut(&mut St, I, &mut dyn Sender<O>) -> Result<()> + Send + 'static,
    F: FnOnce(St, &mut dyn Sender<O>) -> Result<()> + Send + 'static,
{
    fn consume<S: Sender<O>>(&mut self, item: I, sender: &mut S) -> Result<()> {
        let fold = self
            .fold
            .as_mut()
            .expect("the fan-in channel delivers items only to the worker holding the fold");
        (fold.consume)(&mut fold.state, item, sender)
    }

    fn finish<S: Sender<O>>(&mut self, sender: &mut S) -> Result<bool> {
        // `take` so the fold finalizes once even if `finish` is called again.
        if let Some(fold) = self.fold.take() {
            (fold.finish)(fold.state, sender)?;
        }
        Ok(true)
    }
}
