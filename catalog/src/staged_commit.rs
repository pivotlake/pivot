//! The writes a transaction's statements stage, and their commit.
//!
//! A statement that writes (an INSERT, a CREATE, a DROP) does not touch the
//! datastore's durable state as it runs. It stages an action on its
//! transaction as its dataflow starts, never while it is compiled, so a
//! statement that failed to compile has staged nothing; what the dataflow
//! puts outside the catalog meanwhile (the files an INSERT uploads) belongs
//! to that action, which commits or removes it. The transaction's commit
//! applies the actions in the order the statements ran, so a statement sees
//! what the ones before it did; a failure stops the commit, the effects
//! applied before it stay, and the actions after it are discarded.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_deque::{Injector, Steal};
use planner::catalog::{Error as CatalogError, Result as CatalogResult};

/// One durable effect a statement staged on its transaction against a
/// datastore of type `D`.
pub trait CommitAction<D>: Send {
    /// Make the effect durable and visible. Blocking I/O: the commit runs
    /// every action on the blocking pool.
    fn apply(self: Box<Self>, datastore: &D) -> CatalogResult<()>;

    /// Undo what the action put outside the catalog before its commit: a
    /// rolled-back or never-committed INSERT removes the files it uploaded.
    /// Best effort: what cannot be removed is logged and left behind,
    /// unreferenced.
    fn discard(self: Box<Self>);
}

/// The queue a transaction's statements stage their actions on, shared
/// between the transaction and the bindings and dataflows it hands out.
pub type CommitQueue<D> = Arc<Injector<Box<dyn CommitAction<D>>>>;

/// The actions one transaction staged against a datastore, awaiting its
/// commit in the order they were staged.
pub struct StagedCommit<D> {
    datastore: Arc<D>,
    actions: CommitQueue<D>,
    /// Whether commit has run. A second commit is refused, and the drop
    /// safety net discards the staged writes of a transaction its owner
    /// never resolved, so no code path can leak them into limbo; deliberate
    /// paths still resolve explicitly, which keeps the timing (and the thread
    /// doing the work) chosen rather than wherever the last reference happens
    /// to drop.
    committed: AtomicBool,
}

impl<D: Send + Sync + 'static> StagedCommit<D> {
    /// An empty staging queue for one transaction against `datastore`.
    pub fn new(datastore: Arc<D>) -> Self {
        Self {
            datastore,
            actions: Arc::new(Injector::new()),
            committed: AtomicBool::new(false),
        }
    }

    /// The queue a statement's dataflow stages its action on.
    pub fn queue(&self) -> CommitQueue<D> {
        Arc::clone(&self.actions)
    }

    /// Apply every staged action in order on the blocking pool. A read-only
    /// transaction staged nothing and finishes inline. A failure stops the
    /// commit, keeps the effects applied before it, and discards the actions
    /// after it.
    pub async fn commit(&self) -> CatalogResult<()> {
        // Marked at entry, not on success: the swap refuses a second commit,
        // and a failed commit has already drained the staged state, so the
        // drop net must not "roll back" what is no longer staged.
        if self.committed.swap(true, Ordering::AcqRel) {
            return Err(CatalogError::Other("transaction already committed".into()));
        }
        if self.actions.is_empty() {
            return Ok(());
        }
        let datastore = Arc::clone(&self.datastore);
        let actions = drain_injector(&self.actions);
        tokio::task::spawn_blocking(move || -> CatalogResult<()> {
            let mut actions = actions.into_iter();
            while let Some(action) = actions.next() {
                if let Err(error) = action.apply(&datastore) {
                    for unapplied in actions {
                        unapplied.discard();
                    }
                    return Err(error);
                }
            }
            Ok(())
        })
        .await
        .map_err(|error| CatalogError::Other(format!("commit thread panicked: {error}").into()))?
    }

    /// Discard every staged action, removing what it already wrote. Blocking
    /// store I/O when an INSERT uploaded files, so a caller on an async
    /// runtime hops to the blocking pool first.
    pub fn rollback(&self) {
        for action in drain_injector(&self.actions) {
            action.discard();
        }
    }
}

/// The safety net behind the explicit resolution points: a transaction
/// dropped with staged writes and neither commit nor rollback run discards
/// them here, so no owner mistake can leave them in limbo. Reaching this is
/// a bug in the owner, hence the warning; read-only transactions drop
/// silently.
impl<D> Drop for StagedCommit<D> {
    fn drop(&mut self) {
        if self.committed.load(Ordering::Acquire) {
            return;
        }
        // An explicitly rolled-back transaction already drained everything,
        // so the staged-writes check keeps it out of the net too.
        if !self.actions.is_empty() {
            tracing::warn!(
                "transaction dropped with staged writes and no resolution; rolling back"
            );
            for action in drain_injector(&self.actions) {
                action.discard();
            }
        }
    }
}

/// Everything `injector` holds, in the order it was pushed.
pub fn drain_injector<T>(injector: &Injector<T>) -> Vec<T> {
    let mut items = Vec::new();
    loop {
        match injector.steal() {
            Steal::Success(item) => items.push(item),
            Steal::Retry => continue,
            Steal::Empty => return items,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// What the actions of a test did, in order.
    #[derive(Default)]
    struct Journal(Mutex<Vec<String>>);

    impl Journal {
        fn entries(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    /// An action that records itself as applied or discarded, failing to
    /// apply when told to. A discard has no datastore in hand, so the action
    /// carries the journal itself.
    struct Step {
        name: &'static str,
        fails: bool,
        journal: Arc<Journal>,
    }

    impl CommitAction<Journal> for Step {
        fn apply(self: Box<Self>, journal: &Journal) -> CatalogResult<()> {
            if self.fails {
                return Err(CatalogError::Other(format!("{} failed", self.name).into()));
            }
            journal
                .0
                .lock()
                .unwrap()
                .push(format!("applied {}", self.name));
            Ok(())
        }

        fn discard(self: Box<Self>) {
            self.journal
                .0
                .lock()
                .unwrap()
                .push(format!("discarded {}", self.name));
        }
    }

    fn stage(
        staged: &StagedCommit<Journal>,
        journal: &Arc<Journal>,
        name: &'static str,
        fails: bool,
    ) {
        staged.queue().push(Box::new(Step {
            name,
            fails,
            journal: journal.clone(),
        }));
    }

    fn commit(staged: &StagedCommit<Journal>) -> CatalogResult<()> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(staged.commit())
    }

    #[test]
    fn a_failing_action_keeps_the_ones_before_it_and_discards_the_ones_after() {
        let journal = Arc::new(Journal::default());
        let staged = StagedCommit::new(journal.clone());
        stage(&staged, &journal, "first", false);
        stage(&staged, &journal, "second", true);
        stage(&staged, &journal, "third", false);

        let outcome = commit(&staged);

        assert!(outcome.unwrap_err().to_string().contains("second failed"));
        assert_eq!(journal.entries(), ["applied first", "discarded third"]);
    }

    #[test]
    fn a_transaction_dropped_with_staged_writes_discards_them() {
        let journal = Arc::new(Journal::default());
        let staged = StagedCommit::new(journal.clone());
        stage(&staged, &journal, "unresolved", false);

        drop(staged);

        assert_eq!(journal.entries(), ["discarded unresolved"]);
    }

    #[test]
    fn a_second_commit_is_refused() {
        let staged = StagedCommit::new(Arc::new(Journal::default()));
        commit(&staged).unwrap();

        let outcome = commit(&staged);

        assert!(
            outcome
                .unwrap_err()
                .to_string()
                .contains("already committed")
        );
    }
}
