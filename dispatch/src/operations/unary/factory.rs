use crate::api::{Chain, OperatorFactory};
use crate::operations::ChannelFactory;
use crate::operations::channels::Sender;
use crate::operations::unary::{Unary, UnaryOperator};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

pub trait UnaryFactory<I, O>: Send + 'static {
    type Unary: Unary<I, O>;
    fn build_unary(self) -> Self::Unary;
}

pub struct UnaryOperatorFactory<
    I,
    O,
    UF: UnaryFactory<I, O>,
    C: ChannelFactory<I>,
    OP: OperatorFactory<I>,
> {
    head: OP,
    siblings_left: Arc<AtomicUsize>,
    unary_factory: UF,
    channel_factory: C,
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O, UF: UnaryFactory<I, O>, C: ChannelFactory<I>, OP: OperatorFactory<I>>
    UnaryOperatorFactory<I, O, UF, C, OP>
{
    pub fn new(
        head: OP,
        unary_factory: UF,
        channel_factory: C,
        siblings_left: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            head,
            siblings_left,
            unary_factory,
            channel_factory,
            _phantom: Default::default(),
        }
    }
}

unsafe impl<
    I,
    O,
    UF: UnaryFactory<I, O> + Send,
    C: ChannelFactory<I> + Send,
    OP: OperatorFactory<I> + Send,
> Send for UnaryOperatorFactory<I, O, UF, C, OP>
{
}

unsafe impl<
    I,
    O,
    UF: UnaryFactory<I, O> + Sync,
    C: ChannelFactory<I> + Sync,
    OP: OperatorFactory<I> + Sync,
> Sync for UnaryOperatorFactory<I, O, UF, C, OP>
{
}

impl<I: 'static, O: 'static, UF: UnaryFactory<I, O>, C: ChannelFactory<I>, OP: OperatorFactory<I>>
    OperatorFactory<O> for UnaryOperatorFactory<I, O, UF, C, OP>
{
    fn build<OS: Sender<O> + 'static>(self: Box<Self>, sender: OS) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = Box::new(self.head).build(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }
}
