//! Holds fetched row groups back from decoding until their rows are near.
//!
//! A consumer that lays a table's rows out in an order of its own (a sorted
//! compaction) has every row group fetched up front, but decoding them all at
//! once would hold the whole decoded input. The gate sits between the fetchers
//! and the decode stages and releases a row group once the consumer has laid
//! out enough rows, publishing its progress in `consumed`. The caller decides
//! each row group's threshold in `release_after`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dispatch::{Sender, Unary, UnaryFactory, UnaryResult, WorkStatus};

use crate::RowGroupBuffer;

/// When each row group may be decoded: once `consumed` reaches its entry in
/// `release_after`, indexed by the row group's global index.
#[derive(Clone)]
pub struct DecodeGate {
    pub consumed: Arc<AtomicUsize>,
    pub release_after: Arc<Vec<usize>>,
}

pub(crate) struct DecodeGateFactory {
    gate: DecodeGate,
}

impl DecodeGateFactory {
    pub(crate) fn new(gate: DecodeGate) -> Self {
        Self { gate }
    }
}

impl UnaryFactory<RowGroupBuffer, RowGroupBuffer> for DecodeGateFactory {
    type Unary = DecodeGateUnary;

    fn build_unary(self) -> DecodeGateUnary {
        DecodeGateUnary {
            gate: self.gate,
            held: BTreeMap::new(),
        }
    }
}

pub(crate) struct DecodeGateUnary {
    gate: DecodeGate,
    /// Fetched row groups waiting for their threshold, keyed by it and then
    /// by row group so the next to go out is always first.
    held: BTreeMap<(usize, usize), RowGroupBuffer>,
}

impl DecodeGateUnary {
    /// Release every held row group whose threshold the consumer has reached;
    /// returns whether any went out.
    fn release(&mut self, sender: &mut dyn Sender<RowGroupBuffer>) -> UnaryResult<bool> {
        let consumed = self.gate.consumed.load(Ordering::Acquire);
        let mut released = false;
        while let Some(entry) = self.held.first_entry() {
            if entry.key().0 > consumed {
                break;
            }
            sender.send(entry.remove())?;
            released = true;
        }
        Ok(released)
    }
}

impl Unary<RowGroupBuffer, RowGroupBuffer> for DecodeGateUnary {
    fn consume(
        &mut self,
        buffer: RowGroupBuffer,
        sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let group = buffer.metadata.index();
        self.held
            .insert((self.gate.release_after[group], group), buffer);
        self.release(sender)?;
        Ok(())
    }

    fn run(&mut self, sender: &mut dyn Sender<RowGroupBuffer>) -> UnaryResult<WorkStatus> {
        Ok(if self.release(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Called again until every held row group is out.
    fn finish(&mut self, sender: &mut dyn Sender<RowGroupBuffer>) -> UnaryResult<bool> {
        self.release(sender)?;
        Ok(self.held.is_empty())
    }
}
