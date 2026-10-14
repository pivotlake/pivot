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
    /// The sequential half: reassembles the protocol's byte frames into the
    /// units the dataflow's workers take.
    decoder: CopyDecoder,
    format: CopyFormat,
    /// Signalled by workers claiming chunks; [`push`](Self::push) sleeps on
    /// it while the queue is full.
    space_freed: Arc<Notify>,
    cancel: CancelToken,
    /// Taken by [`finish`](Self::finish); still present on drop marks an
    /// unfinished ingest, which the drop aborts.
    handle: Option<DataFlowHandle<std::result::Result<usize, String>>>,
    /// The transaction the statement was planned in, which its bound table
    /// stages into. Taken and committed by [`finish`](Self::finish); rolled
    /// back on drop otherwise.
    transaction: Option<Arc<dyn planner::catalog::CatalogTransaction>>,
    column_count: usize,
}

/// The per-format decoder and the channel it feeds. A text copy is cut into
/// chunks of whole rows and parsed on the workers; an Arrow copy is decoded
/// here, since an IPC stream reads sequentially.
enum CopyDecoder {
    Text {
        framer: TextRowFramer,
        sender: ChannelInputSender<Bytes>,
    },
    Arrow {
        decoder: StreamDecoder,
        sender: ChannelInputSender<RecordBatch>,
    },
}

impl CopyDecoder {
    fn close(&self) {
        match self {
            CopyDecoder::Text { sender, .. } => sender.close(),
            CopyDecoder::Arrow { sender, .. } => sender.close(),
        }
    }
}

impl fmt::Debug for CopyIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyIngest")
            .field("format", &self.format)
            .field("column_count", &self.column_count)
            .finish_non_exhaustive()
    }
}

/// Queue one chunk for the workers, sleeping while the queue is full. A dead
/// dataflow claims nothing again, so the chunk is dropped and `false`
/// reported; its own error surfaces when the flow is collected at finish.
async fn feed_dataflow<T: Send>(
    sender: &ChannelInputSender<T>,
    space_freed: &Notify,
    cancel: &CancelToken,
    mut chunk: T,
) -> bool {
    loop {
        if cancel.is_cancelled() {
            return false;
        }
        // Register before trying so a claim racing the failed send cannot
        // lose the notification. The timeout lets a cancelled dataflow that
        // will never claim again be observed promptly.
        let notified = space_freed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        match sender.try_send(chunk) {
            Ok(()) => return true,
            Err(ChannelInputFull(returned)) => {
                chunk = returned;
                let _ = tokio::time::timeout(Duration::from_millis(50), notified).await;
            }
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
    /// dataflow, sleeping while the queue is full. The decoder carries a
    /// partial row or message across frames, so a unit may span any number
    /// of frames (and one frame may complete several). An error means the
    /// stream is malformed; the copy cannot proceed and should be dropped.
    ///
    /// The frame arrives as refcounted [`Bytes`]. An Arrow frame is handed
    /// to the decoder as is: a message contained in one frame is sliced in
    /// place, and the decoded batches keep the frame's allocation alive. A
    /// text frame is appended to the row in progress.
    pub async fn push(&mut self, bytes: Bytes) -> Result<()> {
        match &mut self.decoder {
            CopyDecoder::Text { framer, sender } => {
                framer.push(&bytes);
                while let Some(rows) = framer.next_chunk() {
                    if !feed_dataflow(sender, &self.space_freed, &self.cancel, rows).await {
                        return Ok(());
                    }
                }
                Ok(())
            }
            CopyDecoder::Arrow { decoder, sender } => {
                let mut buffer = arrow::buffer::Buffer::from(bytes);
                loop {
                    let batch = decoder
                        .decode(&mut buffer)
                        .map_err(|e| Error::Copy(format!("decoding COPY arrow stream: {e}")))?;
                    let Some(batch) = batch else {
                        return Ok(());
                    };
                    if !feed_dataflow(sender, &self.space_freed, &self.cancel, batch).await {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Verify the stream ended cleanly, drain the dataflow, and commit,
    /// returning the ingested-row count. Consuming `self` makes completion
    /// single-use; every failure path drops the remains, which aborts.
    pub async fn finish(mut self) -> Result<usize> {
        match &mut self.decoder {
            // Whole rows went out as they completed; a last row without a
            // trailing newline is still data.
            CopyDecoder::Text { framer, sender } => {
                if let Some(rows) = std::mem::take(framer).finish() {
                    feed_dataflow(sender, &self.space_freed, &self.cancel, rows).await;
                }
            }
            // Complete batches were sent as they decoded, so there is nothing
            // to flush; a stream cut mid-message is an error.
            CopyDecoder::Arrow { decoder, .. } => {
                decoder.finish().map_err(|e| {
                    Error::Copy(format!("COPY arrow stream ended mid-message: {e}"))
                })?;
            }
        }
        self.decoder.close();
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
            self.decoder.close();
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
        let decoder = match feed {
            CopyFeed::TextRows(sender) => CopyDecoder::Text {
                framer: TextRowFramer::default(),
                sender,
            },
            CopyFeed::ArrowBatches(sender) => CopyDecoder::Arrow {
                decoder: StreamDecoder::new(),
                sender,
            },
        };
        Ok(CopyIngest {
            decoder,
            format,
            cancel: handle.cancel_token(),
            space_freed,
            handle: Some(handle),
            transaction: Some(transaction),
            column_count,
        })
    }
}
