//! The pgwire side of `COPY FROM STDIN`: per-session lifecycle around the
//! executor's running ingest (see [`crate::execution::CopyIngest`]).
//!
//! The running copy lives in the pgwire session's typed extensions, so its
//! lifetime is the connection's: dropping an unfinished session (client
//! disconnect, error teardown) drops its [`CopyIngest`], which cancels the
//! dataflow and rolls back the statement's transaction. The session layer owns
//! only protocol policy: one copy at a time, and errors deferred to
//! `CopyDone` (the protocol offers no clean earlier exit from copy-in mode).

use std::sync::Arc;

use crate::execution::CopyIngest;
use pgwire::api::ClientInfo;
use pgwire::error::{ErrorInfo, PgWireError};
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("{0}")]
    InvalidData(String),
    #[error(transparent)]
    Executor(#[from] crate::execution::Error),
    #[error("no COPY FROM STDIN in progress")]
    NoActiveCopy,
    #[error("COPY FROM STDIN is already in progress")]
    AlreadyActive,
    #[error("COPY FROM STDIN aborted by the client: {0}")]
    ClientAbort(String),
}

impl Error {
    pub(crate) fn into_pgwire(self) -> PgWireError {
        let info = ErrorInfo::new("ERROR".to_string(), "XX000".to_string(), self.to_string());
        PgWireError::UserError(Box::new(info))
    }
}

/// One connection's running copy: the executor's ingest plus the first decode
/// failure. After a failure, later frames are discarded and the error is
/// reported at `CopyDone`.
struct ActiveCopy {
    ingest: CopyIngest,
    error: Option<String>,
}

#[derive(Default)]
struct CopySessionSlot {
    active: Mutex<Option<ActiveCopy>>,
}

fn get_slot<C: ClientInfo>(client: &C) -> Result<Arc<CopySessionSlot>, Error> {
    client
        .session_extensions()
        .get::<CopySessionSlot>()
        .ok_or(Error::NoActiveCopy)
}

/// Install a running copy on this pgwire session, returning its column count
/// for the copy-in response. Refusing a second copy drops the new ingest,
/// which aborts it.
pub(crate) async fn begin_copy<C: ClientInfo>(
    client: &C,
    ingest: CopyIngest,
) -> Result<usize, Error> {
    let slot = client
        .session_extensions()
        .get_or_insert_with(CopySessionSlot::default);
    let mut active = slot.active.lock().await;
    if active.is_some() {
        return Err(Error::AlreadyActive);
    }
    let column_count = ingest.column_count();
    *active = Some(ActiveCopy {
        ingest,
        error: None,
    });
    Ok(column_count)
}

/// Feed one pgwire `CopyData` payload to this session's active copy. The
/// payload stays refcounted end to end; nothing copies it.
pub(crate) async fn push_copy<C: ClientInfo>(client: &C, data: bytes::Bytes) -> Result<(), Error> {
    let slot = get_slot(client)?;
    let mut guard = slot.active.lock().await;
    let active = guard.as_mut().ok_or(Error::NoActiveCopy)?;
    if active.error.is_some() {
        return Ok(());
    }
    if let Err(error) = active.ingest.push(data).await {
        active.error = Some(error.to_string());
    }
    Ok(())
}

/// Remove and finish this session's active copy: commit and return the row
/// count, unless a decode error was recorded, in which case the ingest drops
/// (aborting the dataflow and rolling back) and the error is reported.
pub(crate) async fn finish_copy<C: ClientInfo>(client: &C) -> Result<usize, Error> {
    let slot = get_slot(client)?;
    let active = slot.active.lock().await.take().ok_or(Error::NoActiveCopy)?;
    if let Some(message) = active.error {
        return Err(Error::InvalidData(message));
    }
    Ok(active.ingest.finish().await?)
}

/// Remove this session's active copy; dropping it cancels the dataflow and
/// rolls back. A missing session is harmless when handling a client abort.
pub(crate) async fn abort_copy<C: ClientInfo>(client: &C) {
    let Some(slot) = client.session_extensions().get::<CopySessionSlot>() else {
        return;
    };
    let active = slot.active.lock().await.take();
    drop(active);
}
