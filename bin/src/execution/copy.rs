use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use arrow::ipc::reader::StreamDecoder;
use arrow_array::RecordBatch;
use bytes::Bytes;
use dispatch::{CancelToken, ChannelInputFull, ChannelInputSender, DataFlowHandle};
use tokio::sync::Notify;

use super::copy_csv::CsvRowDecoder;
use super::dataflow::{affected_rows, affected_rows_from_record_batch};
use super::{Error, Executor, Result};

/// Reassemble one COPY format's arbitrarily split protocol frames into
/// client-schema batches.
enum CopyFrameDecoder {
    Arrow(StreamDecoder),
    Csv(CsvRowDecoder),
}

impl CopyFrameDecoder {
    fn new(format: &planner::CopyFormat, column_count: usize) -> Self {
        match format {
            planner::CopyFormat::ArrowIpc => Self::Arrow(StreamDecoder::new()),
            planner::CopyFormat::Csv(options) => {
                Self::Csv(CsvRowDecoder::new(column_count, options.clone()))
            }
        }
    }

    fn decode_frame(&mut self, bytes: Bytes) -> Result<Vec<RecordBatch>> {
        match self {
            Self::Arrow(decoder) => {
                let mut buffer = arrow::buffer::Buffer::from(bytes);
                let mut batches = Vec::new();
                loop {
                    match decoder.decode(&mut buffer) {
                        Ok(Some(batch)) => batches.push(batch),
                        Ok(None) => return Ok(batches),
                        Err(e) => {
                            return Err(Error::Copy(format!("decoding COPY arrow stream: {e}")));
                        }
                    }
                }
            }
            Self::Csv(decoder) => decoder
                .decode(&bytes)
                .map_err(|e| Error::Copy(format!("decoding COPY csv data: {e}"))),
        }
    }

    fn finish(&mut self) -> Result<Option<RecordBatch>> {
        match self {
            Self::Arrow(decoder) => {
                decoder.finish().map_err(|e| {
                    Error::Copy(format!("COPY arrow stream ended mid-message: {e}"))
                })?;
                Ok(None)
            }
            Self::Csv(decoder) => decoder
                .finish()
                .map_err(|e| Error::Copy(format!("decoding COPY csv data: {e}"))),
        }
    }
}

/// A running `COPY ... FROM STDIN` ingest, the executor's whole copy surface:
/// the frontend feeds it protocol bytes and finishes or drops it, and every
/// dispatch-facing concern (decoding, backpressure, cancellation, the
/// statement's transaction) stays inside.
pub struct CopyIngest {
    /// The sequential half: reassembles protocol byte frames into
    /// client-schema batches.
    decoder: CopyFrameDecoder,
    sender: ChannelInputSender<RecordBatch>,
    /// Signalled by workers claiming batches; [`push`](Self::push) sleeps on
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
    format: planner::CopyFormat,
}

impl fmt::Debug for CopyIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyIngest")
            .field("column_count", &self.column_count)
            .finish_non_exhaustive()
    }
}

impl CopyIngest {
    /// Columns each incoming batch must carry (for the frontend's
    /// copy-in response).
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// The validated format, used by pgwire to advertise text or binary COPY.
    pub fn format(&self) -> &planner::CopyFormat {
        &self.format
    }

    /// Decode one protocol frame and feed every batch it completes to the
    /// running dataflow. The decoder carries partial CSV rows or Arrow
    /// messages across frames.
    pub async fn push(&mut self, bytes: Bytes) -> Result<()> {
        for batch in self.decoder.decode_frame(bytes)? {
            self.send_to_dataflow(batch).await;
        }
        Ok(())
    }

    /// Send one decoded batch, sleeping while the bounded input queue is full.
    async fn send_to_dataflow(&mut self, mut batch: RecordBatch) {
        loop {
            // A dead dataflow claims nothing again; stop feeding it. Its own
            // error surfaces when the flow is collected at finish.
            if self.cancel.is_cancelled() {
                return;
            }
            // Register before trying so a claim racing the failed send cannot
            // lose the notification. The timeout lets a cancelled dataflow
            // that will never claim again be observed promptly.
            let notified = self.space_freed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.sender.try_send(batch) {
                Ok(()) => return,
                Err(ChannelInputFull(returned)) => {
                    batch = returned;
                    let _ = tokio::time::timeout(Duration::from_millis(50), notified).await;
                }
            }
        }
    }

    /// Verify the stream ended cleanly, drain the dataflow, and commit,
    /// returning the ingested-row count. Consuming `self` makes completion
    /// single-use; every failure path drops the remains, which aborts.
    pub async fn finish(mut self) -> Result<usize> {
        if let Some(batch) = self.decoder.finish()? {
            self.send_to_dataflow(batch).await;
        }
        self.sender.close();
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
            self.sender.close();
        }
        if let Some(transaction) = self.transaction.take() {
            transaction.rollback();
        }
    }
}

impl Executor {
    /// Launch the ingest dataflow for a planned COPY FROM STDIN: a batch
    /// channel fanning out to per-worker schema-conformance stages, feeding
    /// the target table's insert sink, staging into the statement's
    /// transaction. The returned [`CopyIngest`] owns the whole exchange.
    pub(super) async fn launch_copy_ingest(
        &self,
        statement: planner::CopyFromStdin,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    ) -> Result<CopyIngest> {
        let column_count = statement.incoming_column_count();
        let format = statement.format.clone();
        let decoder = CopyFrameDecoder::new(&format, column_count);
        let space_freed = Arc::new(Notify::new());
        let on_claim = {
            let space_freed = space_freed.clone();
            Box::new(move || space_freed.notify_one()) as Box<dyn Fn() + Send + Sync>
        };
        let dispatcher = self.dispatcher.clone();
        // Compile and launch on the blocking pool: the insert sink's write
        // preparation may touch the store.
        let (sender, handle) =
            tokio::task::spawn_blocking(move || -> Result<(ChannelInputSender<RecordBatch>, _)> {
                let (sender, spec) = statement.compile_ingest(&dispatcher, on_claim)?;
                Ok((
                    sender,
                    spec.map(|| affected_rows_from_record_batch).execute(),
                ))
            })
            .await
            .map_err(Error::PlannerPanic)??;
        Ok(CopyIngest {
            decoder,
            cancel: handle.cancel_token(),
            sender,
            space_freed,
            handle: Some(handle),
            transaction: Some(transaction),
            column_count,
            format,
        })
    }
}
