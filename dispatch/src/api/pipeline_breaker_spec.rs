use crate::operations::{Count, Group, OrderByLimit, PipelineBreaker};
use crate::{OrderBy, OutputSpec, dispatcher};
use ahash::RandomState;
use crossbeam_deque::Injector;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc::channel;
use std::sync::{Arc, Barrier};

/// Specification for terminal pipeline operations.
///
/// Pipeline breakers synchronize across workers and produce final output.
/// Created internally by `Node::count()`, `order_by_limit()`, and `group_by_count()`.
pub enum PipelineBreakerSpec {
    /// Count rows across all workers.
    Count(Box<dyn OutputSpec>),
    /// Sort by columns and limit results (distributed top-K).
    OrderByLimit {
        order_by: Vec<OrderBy>,
        output: Box<dyn OutputSpec>,
        limit: usize,
    },
    /// Group by a column and count occurrences.
    GroupByCount {
        group_by_column: usize,
        output: Box<dyn OutputSpec>,
    },
}

impl PipelineBreakerSpec {
    /// Create a breaker factory. Sets up coordination primitives (barriers, channels).
    pub fn build_breaker<'a>(&'a self) -> Box<dyn Fn() -> Box<dyn PipelineBreaker> + 'a> {
        match self {
            PipelineBreakerSpec::Count(o) => {
                let count: Arc<_> = Arc::new(AtomicUsize::new(0));
                let barrier = Arc::new(Barrier::new(dispatcher().workers()));
                Box::new(move || {
                    Box::new(Count::new(count.clone(), barrier.clone(), o.build_output()))
                })
            }
            PipelineBreakerSpec::OrderByLimit {
                order_by,
                output,
                limit,
            } => {
                let (tx, rx) = channel();
                let rx_opt = Rc::new(RefCell::new(Some(rx)));

                Box::new(move || {
                    Box::new(OrderByLimit::new(
                        *limit,
                        output.build_output(),
                        tx.clone(),
                        rx_opt.clone().take(),
                        order_by.clone(),
                    ))
                })
            }
            PipelineBreakerSpec::GroupByCount {
                group_by_column,
                output,
            } => {
                let partitions_injected = Arc::new(AtomicBool::new(false));
                let hash_state = RandomState::new();
                let partition_injector: Arc<Injector<_>> = Default::default();
                let (tx, rx) = channel();
                let rx_opt = Rc::new(RefCell::new(Some(rx)));

                Box::new(move || {
                    Box::new(Group::new(
                        hash_state.clone(),
                        partition_injector.clone(),
                        output.build_output(),
                        *group_by_column,
                        tx.clone(),
                        rx_opt.take(),
                        partitions_injected.clone(),
                    ))
                })
            }
        }
    }
}
