//! A [`UnaryFactory`] that creates a [`Unary`] via its [`Default`] impl.
//!
//! Useful for unary transforms that need no per-worker configuration (e.g. the
//! parquet indexer and decompressor, which are stateless or self-initializing).

use crate::operations::{Unary, UnaryFactory};
use std::marker::PhantomData;

/// Creates a [`Unary`] by calling `U::default()`. The factory carries no state.
pub struct DefaultUnaryFactory<U> {
    _phantom: PhantomData<U>,
}

impl<U> DefaultUnaryFactory<U> {
    pub fn new() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }

    /// Creates one default factory for each dispatch worker.
    pub fn create_for_workers(worker_count: usize) -> Vec<Self> {
        (0..worker_count).map(|_| Self::new()).collect()
    }
}

impl<U> Default for DefaultUnaryFactory<U> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<U> Send for DefaultUnaryFactory<U> {}
unsafe impl<U> Sync for DefaultUnaryFactory<U> {}

impl<I, O, U: Unary<I, O> + 'static + Default> UnaryFactory<I, O> for DefaultUnaryFactory<U> {
    type Unary = U;

    fn build_unary(self) -> Self::Unary {
        U::default()
    }
}
