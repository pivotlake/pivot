use crate::identified::Identifier;
use arrow_schema::ArrowError;
use crossbeam_deque::{Stealer, Worker};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, mpsc};
use thiserror::Error;

mod factory;
pub use factory::{
    ChannelFactory, ReturnToWorkerMpscFactory, StealableChannelFactory, return_to_worker_mpsc,
    stealable,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub trait Sender<O> {
    fn send(&mut self, item: O) -> Result<()>;
}

pub trait Receiver<I> {
    fn is_empty(&self) -> bool;
    fn try_recv(&self) -> Option<I>;
    fn steal(&self) -> Option<I>;
}

impl<O> Sender<O> for Rc<Worker<O>> {
    fn send(&mut self, item: O) -> Result<()> {
        self.push(item);
        Ok(())
    }
}

pub struct MpscSender<T> {
    inner: mpsc::Sender<T>,
    count: Arc<AtomicUsize>,
}

impl<T> Clone for MpscSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            count: self.count.clone(),
        }
    }
}

impl<T> Sender<T> for MpscSender<T> {
    fn send(&mut self, item: T) -> Result<()> {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.inner.send(item).unwrap();
        Ok(())
    }
}

pub struct StealableReceiver<I> {
    worker: Rc<Worker<I>>,
    stealers: Vec<Stealer<I>>,
}

impl<I> StealableReceiver<I> {
    pub fn new(worker: Rc<Worker<I>>, stealers: Vec<Stealer<I>>) -> Self {
        Self { worker, stealers }
    }
}

impl<I> Receiver<I> for StealableReceiver<I> {
    fn is_empty(&self) -> bool {
        self.worker.is_empty()
    }

    fn try_recv(&self) -> Option<I> {
        // if self.worker.len() > 1 {
        // debug!("Channel @ {:?}", self.worker.len())
        // }
        self.worker.pop()
    }

    fn steal(&self) -> Option<I> {
        use crossbeam_deque::Steal;
        for stealer in &self.stealers {
            loop {
                match stealer.steal() {
                    Steal::Success(item) => return Some(item),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }
}

pub trait WorkerIdOutput: 'static {
    fn worker_id(&self) -> Identifier;
}

pub struct WorkerAwareSender<O: WorkerIdOutput> {
    senders: Vec<MpscSender<O>>,
}

impl<O: WorkerIdOutput> WorkerAwareSender<O> {
    pub fn new(senders: Vec<MpscSender<O>>) -> Self {
        Self { senders }
    }
}

impl<O: WorkerIdOutput> Sender<O> for WorkerAwareSender<O> {
    fn send(&mut self, item: O) -> Result<()> {
        let worker_idx = item.worker_id();
        self.senders[worker_idx].send(item)?;
        Ok(())
    }
}

pub struct MpscReceiver<O> {
    inner: mpsc::Receiver<O>,
    count: Arc<AtomicUsize>,
}

impl<T> MpscReceiver<T> {
    pub fn new(inner: mpsc::Receiver<T>, count: Arc<AtomicUsize>) -> Self {
        Self { inner, count }
    }
    pub fn into_parts(self) -> (mpsc::Receiver<T>, Arc<AtomicUsize>) {
        (self.inner, self.count)
    }
}

impl<T> From<MpscReceiver<T>> for mpsc::Receiver<T> {
    fn from(value: MpscReceiver<T>) -> Self {
        value.inner
    }
}

impl<O> Receiver<O> for MpscReceiver<O> {
    fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    fn try_recv(&self) -> Option<O> {
        match self.inner.try_recv().ok() {
            Some(s) => {
                self.count.fetch_sub(1, Ordering::Relaxed);
                Some(s)
            }
            None => None,
        }
    }

    fn steal(&self) -> Option<O> {
        None
    }
}

pub fn mpsc_channel<T>() -> (MpscSender<T>, MpscReceiver<T>) {
    let (tx, rx) = channel();
    let count = Arc::new(AtomicUsize::default());
    (
        MpscSender {
            inner: tx,
            count: count.clone(),
        },
        MpscReceiver { inner: rx, count },
    )
}
