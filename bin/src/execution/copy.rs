use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use arrow::ipc::reader::StreamDecoder;
use arrow_array::RecordBatch;
use bytes::Bytes;
use dispatch::{CancelToken, ChannelInputFull, ChannelInputSender, DataFlowHandle};
use planner::{CopyFeed, CopyFormat, TextRowFramer};
use tokio::sync::Notify;

use super::dataflow::{affected_rows, affected_rows_from_record_batch};
use super::{Error, Executor, Result};

/// A running `COPY ... FROM STDIN` ingest, the executor's whole copy surface:
/// the frontend feeds it protocol bytes and finishes or drops it, and every
/// dispatch-facing concern (decoding, backpressure, cancellation, the
/// statement's transaction) stays inside.
pub struct CopyIngest {
    feed: CopyFeeder,
    format: CopyFormat,
    cancel: CancelToken,
    /// Taken by [`finish`](Self::finish); still present on drop marks an
    /// unfinished ingest, which the drop aborts.
    handle: Option<DataFlowHandle<usize>>,
    /// The transaction the statement was planned in, which its bound table
    /// stages into. Taken and committed by [`finish`](Self::finish); rolled
    /// back on drop otherwise.
    transaction: Option<Arc<dyn planner::catalog::CatalogTransaction>>,
    column_count: usize,
}

impl fmt::Debug for CopyIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyIngest")
            .field("format", &self.format)
            .field("column_count", &self.column_count)
            .finish_non_exhaustive()
    }
}

/// The sequential half of one copy format: reassembles the protocol's byte
/// frames into the items the dataflow's workers take. A partial item is
/// carried across frames, so an item may span any number of frames and one
/// frame may complete several.
trait FrameDecoder {
    type Item: Send;

    /// Take the next protocol frame. The previous frame's items have all
    /// been drained through [`next_item`](Self::next_item) by then.
    fn push_frame(&mut self, frame: Bytes);

    /// The next item the frames so far complete. An error means the stream
    /// is malformed.
    fn next_item(&mut self) -> Result<Option<Self::Item>>;

    /// The stream is over: whatever is still held, or an error when it was
    /// cut mid-item.
    fn finish(&mut self) -> Result<Option<Self::Item>>;
}

/// Text rows are only cut here, into chunks of whole rows; the workers parse
/// them.
impl FrameDecoder for TextRowFramer {
    type Item = Bytes;

    fn push_frame(&mut self, frame: Bytes) {
        self.push(&frame);
    }

    fn next_item(&mut self) -> Result<Option<Bytes>> {
        Ok(self.next_chunk())
    }

    /// Whole rows went out as they completed; a last row without a trailing
    /// newline is still data.
    fn finish(&mut self) -> Result<Option<Bytes>> {
        Ok(std::mem::take(self).finish())
    }
}

/// An Arrow IPC stream reads sequentially, so its batches are decoded here.
/// The frame is handed to the decoder as is: a message contained in one
/// frame is sliced in place, and the decoded batches keep the frame's
/// allocation alive.
struct ArrowFrameDecoder {
    decoder: StreamDecoder,
    /// The part of the current frame not yet decoded.
    frame: arrow::buffer::Buffer,
}

impl ArrowFrameDecoder {
    fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            frame: arrow::buffer::Buffer::from(Bytes::new()),
        }
    }
}

impl FrameDecoder for ArrowFrameDecoder {
    type Item = RecordBatch;

    fn push_frame(&mut self, frame: Bytes) {
        self.frame = arrow::buffer::Buffer::from(frame);
    }

    fn next_item(&mut self) -> Result<Option<RecordBatch>> {
        self.decoder
            .decode(&mut self.frame)
            .map_err(|e| Error::Copy(format!("decoding COPY arrow stream: {e}")))
    }

    /// Complete batches went out as they decoded, so there is nothing to
    /// flush; a stream cut mid-message is an error.
    fn finish(&mut self) -> Result<Option<RecordBatch>> {
        self.decoder
            .finish()
            .map_err(|e| Error::Copy(format!("COPY arrow stream ended mid-message: {e}")))?;
        Ok(None)
    }
}

/// A decoder and the channel its items feed, the same for every format.
struct DataflowFeed<D: FrameDecoder> {
    decoder: D,
    sender: ChannelInputSender<D::Item>,
    /// Signalled by workers claiming items; [`send`](Self::send) sleeps on
    /// it while the queue is full.
    space_freed: Arc<Notify>,
    cancel: CancelToken,
}

impl<D: FrameDecoder> DataflowFeed<D> {
    fn new(
        decoder: D,
        sender: ChannelInputSender<D::Item>,
        space_freed: Arc<Notify>,
        cancel: CancelToken,
    ) -> Self {
        Self {
            decoder,
            sender,
            space_freed,
            cancel,
        }
    }

    /// Decode one frame and queue every item it completes.
    async fn push(&mut self, frame: Bytes) -> Result<()> {
        self.decoder.push_frame(frame);
        while let Some(item) = self.decoder.next_item()? {
            if !self.send(item).await {
                break;
            }
        }
        Ok(())
    }

    /// End the stream: queue what the decoder still holds and close the
    /// channel.
    async fn finish(&mut self) -> Result<()> {
        if let Some(item) = self.decoder.finish()? {
            self.send(item).await;
        }
        self.sender.close();
        Ok(())
    }

    /// Queue one item for the workers, sleeping while the queue is full. A
    /// dead dataflow claims nothing again, so the item is dropped and
    /// `false` reported; its own error surfaces when the flow is collected
    /// at finish.
    async fn send(&self, mut item: D::Item) -> bool {
        loop {
            if self.cancel.is_cancelled() {
                return false;
            }
            // Register before trying so a claim racing the failed send
            // cannot lose the notification. The timeout lets a cancelled
            // dataflow that will never claim again be observed promptly.
            let notified = self.space_freed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.sender.try_send(item) {
                Ok(()) => return true,
                Err(ChannelInputFull(returned)) => {
                    item = returned;
                    let _ = tokio::time::timeout(Duration::from_millis(50), notified).await;
                }
            }
        }
    }
}

/// The feed of the statement's format.
enum CopyFeeder {
    Text(DataflowFeed<TextRowFramer>),
    Arrow(DataflowFeed<ArrowFrameDecoder>),
}

impl CopyFeeder {
    async fn push(&mut self, frame: Bytes) -> Result<()> {
        match self {
            CopyFeeder::Text(feed) => feed.push(frame).await,
            CopyFeeder::Arrow(feed) => feed.push(frame).await,
        }
    }

    async fn finish(&mut self) -> Result<()> {
        match self {
            CopyFeeder::Text(feed) => feed.finish().await,
            CopyFeeder::Arrow(feed) => feed.finish().await,
        }
    }

    fn close(&self) {
        match self {
            CopyFeeder::Text(feed) => feed.sender.close(),
            CopyFeeder::Arrow(feed) => feed.sender.close(),
        }
    }
}

impl CopyIngest {
    /// Columns each incoming row must carry (for the frontend's copy-in
    /// response).
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// The format the client's bytes are in.
    pub fn format(&self) -> &CopyFormat {
        &self.format
    }

    /// Decode one protocol frame and feed what it completes to the running
    /// dataflow, sleeping while the queue is full. An error means the stream
    /// is malformed; the copy cannot proceed and should be dropped.
    pub async fn push(&mut self, bytes: Bytes) -> Result<()> {
        self.feed.push(bytes).await
    }

    /// Verify the stream ended cleanly, drain the dataflow, and commit,
    /// returning the ingested-row count. Consuming `self` makes completion
    /// single-use; every failure path drops the remains, which aborts.
    pub async fn finish(mut self) -> Result<usize> {
        self.feed.finish().await?;
        let handle = self.handle.take().expect("a copy finishes only once");
        let outputs = tokio::task::spawn_blocking(move || handle.collect())
            .await
            .map_err(Error::WorkerPanic)??;
        let count = affected_rows(outputs)?;
        let transaction = self
            .transaction
            .take()
            .expect("the transaction resolves only here");
        transaction.commit().await?;
        Ok(count)
    }
}

/// Dropping an unfinished ingest aborts it: cancel the dataflow, stop the
/// queue, and roll back the statement's transaction, discarding whatever the
/// copy staged. A finished ingest already took both fields, so its drop does
/// nothing.
impl Drop for CopyIngest {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.cancel.cancel();
            self.feed.close();
        }
        if let Some(transaction) = self.transaction.take() {
            transaction.rollback();
        }
    }
}

impl Executor {
    /// Launch the ingest dataflow for a planned COPY FROM STDIN: a chunk
    /// channel fanning out to per-worker decode-and-conform stages, feeding
    /// the target table's insert sink, staging into the statement's
    /// transaction. The returned [`CopyIngest`] owns the whole exchange.
    pub(super) async fn launch_copy_ingest(
        &self,
        statement: planner::CopyFromStdin,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    ) -> Result<CopyIngest> {
        let column_count = if statement.columns.is_empty() {
            statement.table.columns().len()
        } else {
            statement.columns.len()
        };
        let format = statement.format.clone();
        let space_freed = Arc::new(Notify::new());
        let on_claim = {
            let space_freed = space_freed.clone();
            Box::new(move || space_freed.notify_one()) as Box<dyn Fn() + Send + Sync>
        };
        let dispatcher = self.dispatcher.clone();
        // Compile and launch on the blocking pool: the insert sink's write
        // preparation may touch the store.
        let (feed, handle) = tokio::task::spawn_blocking(move || -> Result<(CopyFeed, _)> {
            let (feed, spec) = statement.compile_ingest(&dispatcher, on_claim)?;
            Ok((feed, spec.map(|| affected_rows_from_record_batch).execute()))
        })
        .await
        .map_err(Error::PlannerPanic)??;
        let cancel = handle.cancel_token();
        let feed = match feed {
            CopyFeed::TextRows(sender) => CopyFeeder::Text(DataflowFeed::new(
                TextRowFramer::default(),
                sender,
                space_freed,
                cancel.clone(),
            )),
            CopyFeed::ArrowBatches(sender) => CopyFeeder::Arrow(DataflowFeed::new(
                ArrowFrameDecoder::new(),
                sender,
                space_freed,
                cancel.clone(),
            )),
        };
        Ok(CopyIngest {
            feed,
            format,
            cancel,
            handle: Some(handle),
            transaction: Some(transaction),
            column_count,
        })
    }
}
