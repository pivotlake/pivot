//! The index as dataflow stages.
//!
//! Building:
//!
//! ```text
//!   rows of a file, in whatever ranges        every worker        one worker per file
//!   the scan hands out                        ────────────        ───────────────────
//!   ─────────────────────────────────▶  TokenRunBuilder  ─▶  IndexAssembler  ─▶  AssembledIndex
//!                                        (the caller's          gathers a file's      the file's index,
//!                                         stage: it knows       runs, merges them     ready to write
//!                                         the batch layout)
//! ```
//!
//! Tokenizing is the caller's stage because only it knows how its batches
//! carry text and row positions; it emits a [`FileTokenRun`] per range. The
//! runs of one file all name the same assembly worker, and travel to it over a
//! `return_to_worker_mpsc` channel, so [`IndexAssembler`] needs no shared
//! state.
//!
//! Looking up:
//!
//! ```text
//!   LookupRequest { index file, rows, payload }  ─▶  IndexLookups  ─▶  LookupResult { payload, matching rows }
//! ```
//!
//! [`IndexLookups`] runs one [`TextIndexLookup`] per request, many at a time,
//! each parked on its outstanding read. The payload is whatever the caller
//! needs back with the answer (the row group the rows belong to, say).

use crate::api::BuildContext;
use crate::io::{OperatorIO, ReadRequestId, ReadResponse};
use crate::memory::FileBytes;
use crate::operations::channels::{Sender, WorkerIdOutput};
use crate::operations::unary::{Error as UnaryError, Result as UnaryResult, Unary, UnaryFactory};
use crate::operations::{Built, BuiltUnary};
use crate::text_index::builder::{TokenRun, write_index_file};
use crate::text_index::lookup::{IndexFile, LookupProgress, TextIndexLookup, TextQuery};
use crate::text_index::row_mask::RowMask;
use crate::{Identifier, text_index};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

impl From<text_index::Error> for UnaryError {
    fn from(error: text_index::Error) -> Self {
        match error {
            text_index::Error::Io(error) => UnaryError::Io(error),
            other => UnaryError::Operator(Box::new(other)),
        }
    }
}

/// One range of a file's rows, tokenized, on its way to the worker that
/// assembles the file's index.
pub struct FileTokenRun<F> {
    /// Which file the run belongs to.
    pub file: F,
    /// Rows in the whole file. The index is assembled once runs covering all
    /// of them have arrived.
    pub file_row_count: u32,
    /// The worker assembling this file's index: the same for every run of the
    /// file.
    pub assembly_worker: Identifier,
    pub run: TokenRun,
}

impl<F: 'static> WorkerIdOutput for FileTokenRun<F> {
    fn worker_id(&self) -> Identifier {
        self.assembly_worker
    }
}

/// A file's finished index, ready to be written.
pub struct AssembledIndex<F> {
    pub file: F,
    pub bytes: FileBytes,
}

/// Gathers each file's token runs and merges them into its index once every
/// row of the file is covered.
pub struct IndexAssembler<F> {
    gathering: HashMap<F, GatheredRuns>,
}

struct GatheredRuns {
    runs: Vec<TokenRun>,
    rows_covered: u64,
}

impl<F> Default for IndexAssembler<F> {
    fn default() -> Self {
        Self {
            gathering: HashMap::new(),
        }
    }
}

impl<F: Hash + Eq + Clone> Unary<FileTokenRun<F>, AssembledIndex<F>> for IndexAssembler<F> {
    fn consume(
        &mut self,
        run: FileTokenRun<F>,
        sender: &mut dyn Sender<AssembledIndex<F>>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let gathered = self
            .gathering
            .entry(run.file.clone())
            .or_insert_with(|| GatheredRuns {
                runs: Vec::new(),
                rows_covered: 0,
            });
        gathered.rows_covered += run.run.rows().len() as u64;
        gathered.runs.push(run.run);
        if gathered.rows_covered < run.file_row_count as u64 {
            return Ok(());
        }

        let gathered = self.gathering.remove(&run.file).unwrap();
        sender.send(AssembledIndex {
            file: run.file,
            bytes: write_index_file(gathered.runs)?,
        })?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<AssembledIndex<F>>) -> UnaryResult<bool> {
        if !self.gathering.is_empty() {
            return Err(UnaryError::Operator(
                format!(
                    "{} text indexes were left without all of their file's rows",
                    self.gathering.len()
                )
                .into(),
            ));
        }
        Ok(true)
    }
}

/// A question for an index file: which rows of `rows` match the query?
pub struct LookupRequest<P> {
    pub index: IndexFile,
    /// Rows of the indexed file to answer for.
    pub rows: Range<u32>,
    /// Handed back untouched with the answer.
    pub payload: P,
}

pub struct LookupResult<P> {
    pub payload: P,
    /// One bit per row of the request's range.
    pub matching: RowMask,
}

/// Lookups parked on a read before more requests are taken. Each holds a few
/// pinned cache entries, and index reads are small, so a modest number in
/// flight hides per-read latency without holding much.
const MAX_LOOKUPS_IN_FLIGHT: usize = 64;

/// Answers [`LookupRequest`]s for one query.
pub struct IndexLookups<P> {
    query: Arc<TextQuery>,
    in_flight: HashMap<ReadRequestId, (TextIndexLookup, P)>,
}

impl<P> IndexLookups<P> {
    pub fn new(query: Arc<TextQuery>) -> Self {
        Self {
            query,
            in_flight: HashMap::new(),
        }
    }

    /// Send the lookup's answer on, or park it on the read it now waits for.
    fn settle(
        &mut self,
        lookup: TextIndexLookup,
        payload: P,
        progress: LookupProgress,
        sender: &mut dyn Sender<LookupResult<P>>,
    ) -> UnaryResult<()> {
        match progress {
            LookupProgress::Done(matching) => sender.send(LookupResult { payload, matching })?,
            LookupProgress::Reading(read) => {
                self.in_flight.insert(read, (lookup, payload));
            }
        }
        Ok(())
    }
}

impl<P> Unary<LookupRequest<P>, LookupResult<P>> for IndexLookups<P> {
    fn consume(
        &mut self,
        request: LookupRequest<P>,
        sender: &mut dyn Sender<LookupResult<P>>,
        io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let mut lookup = TextIndexLookup::new(request.index, self.query.clone(), request.rows);
        let progress = lookup.advance(io)?;
        self.settle(lookup, request.payload, progress, sender)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.in_flight.len() < MAX_LOOKUPS_IN_FLIGHT
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<LookupResult<P>>,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> UnaryResult<()> {
        let (mut lookup, payload) = self
            .in_flight
            .remove(&response.id())
            .expect("a response for a read no lookup is waiting on");
        let progress = lookup.process_read_response(response, io)?;
        self.settle(lookup, payload, progress, sender)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<LookupResult<P>>) -> UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

/// Builds one worker's [`IndexLookups`].
pub struct IndexLookupsFactory<P> {
    query: Arc<TextQuery>,
    _payload: PhantomData<fn(P)>,
}

impl<P> IndexLookupsFactory<P> {
    /// One factory per worker, all answering `query`.
    pub fn create_for_workers(worker_count: usize, query: Arc<TextQuery>) -> Vec<Self> {
        (0..worker_count)
            .map(|_| Self {
                query: query.clone(),
                _payload: PhantomData,
            })
            .collect()
    }
}

impl<P: 'static> UnaryFactory<LookupRequest<P>, LookupResult<P>> for IndexLookupsFactory<P> {
    type Unary = IndexLookups<P>;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(IndexLookups::new(self.query))
    }
}
