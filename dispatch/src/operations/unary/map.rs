//! Map operator: apply a 1→1 transform `F: FnMut(I) -> O` and forward the
//! output downstream.
//!
//! Generic over both the input and the output type, so it covers
//! `RecordBatch → RecordBatch` transforms *and* the cross-type case (one item
//! per call instead of a callback-driven N).
//!
//! The output type lives in the closure signature, which means every `send`
//! downstream is monomorphized — no dyn dispatch.

use crate::operations::channels::Sender;
use crate::operations::unary::{self, Unary, UnaryFactory};
use std::sync::Arc;

/// Factory wrapping the builder of the per-worker closure, which it calls in
/// [`build_unary`](UnaryFactory::build_unary), on the worker, so every
/// worker's closure is created in parallel rather than one after another on
/// the thread compiling the query.
pub struct MapFactory<F>(pub Arc<dyn Fn() -> F + Send + Sync>);

impl<I, O, F> UnaryFactory<I, O> for MapFactory<F>
where
    F: FnMut(I) -> O + Send + 'static,
    I: 'static,
    O: 'static,
{
    type Unary = Map<F>;

    fn build_unary(self) -> Self::Unary {
        Map { func: (self.0)() }
    }
}

pub struct Map<F> {
    func: F,
}

impl<I, O, F> Unary<I, O> for Map<F>
where
    F: FnMut(I) -> O + Send,
{
    fn consume(
        &mut self,
        item: I,
        sender: &mut dyn Sender<O>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        sender.send((self.func)(item))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::{CollectSender, run_unary};

    #[test]
    fn maps_each_item_one_to_one() {
        // Setup
        let map = MapFactory(|x: i32| x * 2).build_unary();

        // Execute
        let out = run_unary(map, vec![1, 2, 3]);

        // Assert
        assert_eq!(out, vec![2, 4, 6]);
    }

    #[test]
    fn output_type_can_differ_from_input() {
        // Setup: i32 → String.
        let mut map = MapFactory(|x: i32| format!("v={x}")).build_unary();

        // Execute
        let mut collector: CollectSender<String> = CollectSender::new();
        let mut test_io = crate::io::TestOperatorIO::default();
        let mut io = test_io.io();
        map.consume(1, &mut collector, &mut io).unwrap();
        map.consume(2, &mut collector, &mut io).unwrap();

        // Assert
        assert_eq!(collector.items, vec!["v=1".to_string(), "v=2".to_string()]);
    }

    #[test]
    fn empty_input_produces_empty_output() {
        // Setup
        let map = MapFactory(|x: i32| x + 1).build_unary();

        // Execute
        let out = run_unary(map, Vec::<i32>::new());

        // Assert
        assert!(out.is_empty());
    }
}
