//! Closure-backed sink stage that consumes items without forwarding them.

use super::{Result, Unary, UnaryFactory};
use crate::operations::channels::Sender;

/// Factory for one worker-local sink.
pub struct SinkFactory<Consume, Finish> {
    consume: Consume,
    finish: Finish,
}

impl<Consume, Finish> SinkFactory<Consume, Finish> {
    pub(crate) fn new(consume: Consume, finish: Finish) -> Self {
        Self { consume, finish }
    }
}

impl<Input, Output, Consume, Finish> UnaryFactory<Input, Output> for SinkFactory<Consume, Finish>
where
    Input: 'static,
    Output: 'static,
    Consume: FnMut(Input) -> Result<()> + Send + 'static,
    Finish: FnMut(&mut dyn Sender<Output>) -> Result<()> + Send + 'static,
{
    type Unary = Sink<Consume, Finish>;

    fn build_unary(self) -> Self::Unary {
        Sink {
            consume: self.consume,
            finish: self.finish,
        }
    }
}

pub struct Sink<Consume, Finish> {
    consume: Consume,
    finish: Finish,
}

impl<Input, Output, Consume, Finish> Unary<Input, Output> for Sink<Consume, Finish>
where
    Consume: FnMut(Input) -> Result<()> + Send,
    Finish: FnMut(&mut dyn Sender<Output>) -> Result<()> + Send,
{
    fn consume<SenderType: Sender<Output>>(
        &mut self,
        item: Input,
        _sender: &mut SenderType,
    ) -> Result<()> {
        (self.consume)(item)
    }

    fn finish<SenderType: Sender<Output>>(&mut self, sender: &mut SenderType) -> Result<bool> {
        (self.finish)(sender)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::operations::unary::test_utils::run_unary_to_completion;

    #[test]
    fn consumes_inputs_and_emits_only_from_finish() {
        // Setup
        let sum = Arc::new(AtomicUsize::new(0));
        let consumed = sum.clone();
        let finished = sum.clone();
        let sink = SinkFactory::new(
            move |value| {
                consumed.fetch_add(value, Ordering::Relaxed);
                Ok(())
            },
            move |sender: &mut dyn Sender<usize>| {
                sender.send(finished.load(Ordering::Relaxed))?;
                Ok(())
            },
        )
        .build_unary();

        // Execute
        let output = run_unary_to_completion(sink, vec![1, 2, 3]);

        // Assert
        assert_eq!(output, vec![6]);
    }
}
