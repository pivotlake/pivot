use crate::dispatcher;
use crate::operations::channels::{
    MpscReceiver, MpscSender, Receiver, Sender, StealableReceiver, WorkerAwareSender,
    WorkerIdOutput, mpsc_channel,
};
use crossbeam_deque::{Stealer, Worker};
use std::rc::Rc;

pub trait ChannelFactory<T>: Send {
    type Sender: Sender<T> + 'static;
    type Receiver: Receiver<T> + 'static;
    fn build(self) -> (Self::Sender, Self::Receiver);
}

pub fn stealable<T: Send>() -> impl IntoIterator<Item = StealableChannelFactory<T>> {
    let workers: Vec<_> = (0..dispatcher().workers())
        .map(|_| Worker::new_lifo())
        .collect();

    let workers_with_stealers = workers
        .iter()
        .enumerate()
        .map(|(i, _)| {
            workers
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, w)| w.stealer())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    workers
        .into_iter()
        .zip(workers_with_stealers)
        .map(|(w, s)| StealableChannelFactory {
            worker: w,
            stealers: s,
        })
}

pub struct StealableChannelFactory<T: Send> {
    worker: Worker<T>,
    stealers: Vec<Stealer<T>>,
}

impl<T: Send> StealableChannelFactory<T> {
    pub fn new(worker: Worker<T>, stealers: Vec<Stealer<T>>) -> StealableChannelFactory<T> {
        StealableChannelFactory { worker, stealers }
    }
}

impl<T: Send + 'static> ChannelFactory<T> for StealableChannelFactory<T> {
    type Sender = Rc<Worker<T>>;
    type Receiver = StealableReceiver<T>;

    fn build(self) -> (Rc<Worker<T>>, StealableReceiver<T>) {
        let worker = Rc::new(self.worker);
        (
            worker.clone(),
            StealableReceiver::new(worker, self.stealers),
        )
    }
}

pub struct ReturnToWorkerMpscFactory<T> {
    senders: Vec<MpscSender<T>>,
    receiver: MpscReceiver<T>,
}

impl<T: 'static + Send + WorkerIdOutput> ChannelFactory<T> for ReturnToWorkerMpscFactory<T> {
    type Sender = WorkerAwareSender<T>;
    type Receiver = MpscReceiver<T>;

    fn build(self) -> (Self::Sender, Self::Receiver) {
        (WorkerAwareSender::new(self.senders), self.receiver)
    }
}

pub fn return_to_worker_mpsc<T: 'static + Send + WorkerIdOutput>()
-> impl IntoIterator<Item = ReturnToWorkerMpscFactory<T>> {
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..dispatcher().workers())
        .map(|_| mpsc_channel::<T>())
        .unzip();

    receivers.into_iter().enumerate().map(move |(_i, rx)| {
        let txs: Vec<_> = senders.iter().cloned().collect();
        ReturnToWorkerMpscFactory {
            senders: txs,
            receiver: rx,
        }
    })
}
