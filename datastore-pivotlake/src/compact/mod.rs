//! Background compaction of a table's Parquet files, whichever table format
//! holds the table, and the datastore's other maintenance settings.
//!
//! Frequent writes litter a table with small files, and scans pay per file.
//! The [`CompacterHandle`] runs two strategies, split by file size: files
//! below half the configured target are merged as *small files*, files at
//! least half full are candidates for *layout optimization*. Both rewrite
//! their inputs into target-sized outputs and swap them into the table in a
//! single commit. A row group larger than the target stays one oversized
//! file, since row groups are the smallest independently encoded unit. Each
//! pass works on one partition, trying its small files first, then its
//! layout.
//!
//! # The files the rounds select from
//!
//! The compacter keeps, per table, the files the table's format hands it
//! as its own lists
//! ([`index::TableFileLists`]): a partition at a time, its small files and
//! its large ones each in the order a partition's files stand in. A table's
//! format hands its files out in manifests it never changes, a manifest of
//! an Iceberg table or the whole of one version of a Delta table, and the
//! lists are brought to the version the datastore has published by the
//! manifests that version has and the lists do not hold, and the manifests
//! the lists hold and it no longer has. That happens before every
//! selection, a round's first included, so a selection sees what was
//! committed since the one before, and reads nothing of a table whose
//! manifests it already holds. A merge of the round's own is no exception:
//! its commit publishes a version, and the selection that follows takes
//! its inputs out of the lists and its outputs in, as it does the files of
//! any other commit.
//! Nothing is kept when the compacter starts, so its first round over a
//! table takes every manifest the table has.
//!
//! # Which partitions a round examines
//!
//! A partition is settled once a round finds nothing left to merge in it,
//! and stays so until a file joins or leaves it. A routine round examines
//! the partitions that are not settled as it begins, and no other: a
//! settled partition is not looked at. A round that fails leaves its
//! partitions unsettled for the next. `COMPACT ... FINAL` examines every
//! partition, one after another.
//!
//! # What the rounds remember of a partition's layout
//!
//! Per partition, the rounds remember the pairs of its large files that
//! clear the contested bar, each with its share, and which large files
//! joined since they last measured ([`overlap::OverlapCache`]), so a
//! selection measures only the files that joined the partition since the
//! last, and a selection in a partition none joined measures nothing.
//! Nothing is remembered of a file of no pair. A pair below
//! the bar is not remembered: a final sweep, which merges those too,
//! measures every pair of a partition again when it reaches it, and lets go
//! of the ones below the bar when it is done with the partition. A file is
//! forgotten as it leaves its partition's list, a partition left without a
//! file with everything remembered of it, and a table the datastore no
//! longer holds by the next round over any table.
//!
//! # Small files
//!
//! A merge of small files takes a run of neighbours: files that stand next
//! to one another among the partition's small files. Every run of at least
//! two is weighed whose bytes stay within the most the inputs of one merge
//! may hold, five times the file target by default: a merge writes as many
//! files as its rows fill. A
//! run has at most [`MAX_SMALL_FILES_PER_MERGE`] files, since a merge holds
//! the rows of all its inputs in memory, or up to a hundred while they hold
//! no more than 64 MiB between them.
//!
//! A run may merge when it has the configured number of files and is
//! balanced: its bytes are enough times those of its largest file, each
//! file weighed with a fixed cost so that many tiny files qualify, and a
//! large file is not rewritten to absorb a few small ones. The balance
//! asked is five times in a partition with few small files, which five
//! files of one size meet, and falls to two as the partition gets crowded,
//! from ten small files to fifty.
//!
//! A routine round therefore leaves small files unmerged: a run that is
//! not balanced enough merges only once its partition is crowded enough to
//! ask for less, or under `COMPACT ... FINAL`. Fewer than five files stay
//! as they are in a partition with few small files, and so do files of
//! which the most one merge's inputs may hold takes fewer than five. Even
//! the least balance
//! leaves out two files of different sizes, and a file of more than a few
//! mebibytes beside one or two much smaller ones.
//!
//! Of the runs that may merge, a round takes the one that rewrites the
//! fewest bytes for each file it removes, the last of equals in the
//! partition's order. A run that would merge to about the size of the file
//! before it counts as cheaper than it is: merging it leaves two neighbours
//! of one size for a later merge.
//!
//! A routine round swaps up to [`MAX_SMALL_FILE_MERGES_PER_COMMIT`] merges
//! of one partition into the table in one commit. Their runs are picked
//! together when the merges are selected, from the partition's small files
//! as the table holds them: each run is the cheapest of those that hold no
//! file of a run picked before it, so no file is an input of two merges,
//! and no merge takes the output of another of its commit. A run after the
//! first is picked only while the inputs of the commit's merges stay
//! within [`MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER`] between them. The
//! merges are written at the same time, each on a thread of its own. Until
//! the commit is made, no other merge takes an input of any of them.
//!
//! `COMPACT ... FINAL` drops those guards: it merges the partition's first
//! small files, [`MAX_SMALL_FILES_PER_MERGE`] at a time and each merge in a
//! commit of its own, until fewer than two are left.
//!
//! # Layout optimization
//!
//! Layout optimization reduces the work left after min/max pruning. A scan
//! can discard a file when its sort-key range misses the predicate, then
//! discard row groups within a retained file by the same test. If many files
//! cover the same key range, file pruning retains all of them and each can
//! contribute row groups whose bounds meet the predicate. Rewriting their
//! rows in sort order cuts the shared range into files with narrower,
//! non-overlapping ranges, so a query usually retains fewer files and fewer
//! row groups.
//!
//! Candidates are files at least half the target size in one partition of a
//! table with a sort key. A pair is measured on the first sort column on
//! which the files are not both fixed at the same value. Their ranges must
//! share more than a single boundary value, and a routine pair qualifies
//! only when more than
//! [`overlap::CONTESTED_ROW_GROUP_PERCENT_BAR`] percent of each file's row
//! groups lie inside the other file's range. Missing row-group bounds count
//! as contested because they cannot prove that the group is outside it.
//!
//! A routine round starts with the file having the most qualifying partners
//! and merges a full clique, meaning that every pair in the group qualifies.
//! The default clique has [`DEFAULT_LAYOUT_CLIQUE_SIZE`] files, six at the
//! default 20 percent bar. Smaller cliques wait for more files instead of
//! causing repeated partial rewrites. The default size also makes each
//! output's range narrow enough that it normally falls below the bar against
//! an unmerged broad file, keeping successive layout scales apart. Files in
//! different partitions are never paired. Neither are files with unusable
//! bounds on the sort key or no rows, pairs whose ranges are disjoint or merely
//! touch, or pairs fixed at one identical value across the entire sort key.
//! These exclusions apply even to a final sweep.
//!
//! For example, the pinned distribution scenario begins with 1,000 identical
//! target-full files, each spanning 1,000 keys in ten equal row groups. With
//! six-file routine cliques it performs 472 merges and converges to:
//!
//! ```text
//! key-range span (keys)   files
//!                     5     352
//!                     6     512
//!                    28      36
//!                    29      72
//!                   167      16
//!                   168       8
//!                 1,000       4
//! ```
//!
//! Thus 864 files cover only five or six adjacent keys, with progressively
//! fewer files at the wider leftover scales. A narrow query can prune almost
//! all of those files before using row-group bounds inside the few it keeps,
//! instead of visiting all 1,000 original full-range files.
//!
//! `COMPACT ... FINAL` still requires a measurable pair that shares a real
//! key interval, but drops the contested-row-group bar and the full-clique
//! requirement. It takes a seed and its strongest directly overlapping
//! partners, up to the same group-size cap, and keeps merging until no such
//! group remains. This lets a broad file be rewritten with narrower files
//! nested inside its range even when those narrow files do not all overlap
//! one another.
//!
//! # Merging and committing
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! metadata (paths, sizes, partition values, and stats; no directory
//! scanning), and the merge (read, upload, swap) runs through the table's
//! own store. A table in a bucket compacts through the exact same code path
//! as a local one.
//!
//! The swapped-out inputs are not deleted with the swap: a query that loaded
//! the prior version is still reading them, and the swap's commit is their
//! durable removal from the table. The objects stay in place until whatever
//! vacuums the table's storage ages them out.
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups, the encode stages repack them into full-size
//! row groups, and the upload stage writes each merged file over the shared
//! io_uring ring, the same operators an INSERT uses. The compacter's own task
//! only drives that dataflow (from a blocking thread) and, once the uploads
//! land, commits the swap.
//!
//! Crash safety comes from the table's metadata, not from ordering tricks: a
//! file is part of the table iff the current version lists it. The merged
//! files are uploaded, then swapped in for the inputs in one commit (labelled
//! a rearrangement, not a data change); a crash before the commit leaves only
//! *orphan* objects (unlisted merged files), never a double read.
//!
//! One commit swaps in one layout merge, or the small-file merges of one
//! partition a round selected together: every input of every merge that
//! was written leaves the table and every output joins it, or none does.
//! A merge that fails leaves no output and keeps its inputs in the table;
//! the merges written beside it are committed all the same, and the round
//! ends there with the failure, which is what its `COMPACT` answers, its
//! partition left for the next round to examine. When the commit fails,
//! the outputs of all the merges are deleted and the table stays as it
//! was; only a commit whose outcome is unknown leaves its outputs, since
//! it may have been made. A compacter told to stop while it writes such a
//! group finishes every merge of the group, all of them having started
//! together, and commits them.
//!
//! Merging always writes into the table's own directory, so compacting a
//! file the table adopted from elsewhere copies those rows under the table.
//! The adopted original is dropped from the metadata but never deleted: it
//! is not the table's to delete. A table that adopts a directory of small
//! files and then compacts them therefore ends up storing that data twice,
//! once where the user put it and once in its own directory. That is the
//! price of never writing into a directory the user owns.
//!
//! # Hosting
//!
//! The datastore runs one [`CompacterHandle`] actor over itself: what names
//! its tables, holds the published copy of each, refreshes one from storage
//! and commits a swap. A round names its table by identity and keeps no
//! copy of it: it reads the published copy whenever it lists partitions or
//! selects a merge. Nor does the sweep over all the tables keep a copy of
//! any across its rounds, only each one's name and identity: a copy kept
//! that long would be a version the rounds' commits leave behind, held in
//! memory with whatever it holds of the files they remove. A merge looks
//! its inputs up in that same copy and its commit publishes the next
//! version, so a round never selects a file its merge cannot find. What
//! another process committed is in the published copy once the datastore's
//! periodic refresh has read it; a round has the table refreshed only after
//! a commit conflict. Configured timer ticks and explicit `COMPACT` commands
//! share the actor's queue, so their rewrites never race in one process. The
//! actor stops before the datastore's dispatch pool shuts down.
//!
//! [`MAX_SMALL_FILES_PER_MERGE`]: small_files::MAX_SMALL_FILES_PER_MERGE
//! [`MAX_SMALL_FILE_MERGES_PER_COMMIT`]: small_files::MAX_SMALL_FILE_MERGES_PER_COMMIT
//! [`MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER`]: small_files::MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use object_storage::ObjectPath;
use planner::catalog::SchemaQualifiedTableName;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::maintenance::StopSignal;

use crate::table_format::PartitionHandle;
use crate::{CatalogTable, Error, PivotlakeDatastore, Result};

mod execute;
mod index;
mod overlap;
mod small_files;
pub(crate) mod sort_key;

use execute::{MergeBatch, MergeKind, MergeOutcome, compact_batch};

/// Rows per row group in a merged file.
pub const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default compacted-file size target. Files smaller than half this size enter
/// small-file compaction; files at least half this size are full enough to
/// enter layout optimization instead.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default for how many times the desired output size the inputs of one
/// small-file merge may hold. A merge writes as many output files as its
/// rows fill, so several files that each nearly reach the desired size, and
/// that no smaller group could merge, are rewritten together. A merge holds
/// its whole input decoded in memory, so this is also the most one merge of
/// small files holds: see [`DEFAULT_COMPACT_PARALLELISM`].
pub const DEFAULT_MAX_MERGE_INPUT_MULTIPLIER: f64 = 5.0;

/// Default for the fewest small files one merge takes: any two.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 2;

/// Default cadence for re-checking the tables' metadata. Candidates only
/// change when a commit produces a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(30);

/// Default number of commits' worth of merges a datastore has in flight at
/// once ([`CompactionConfig::max_concurrent_merges`]). A merge holds its
/// whole input decoded on the memory ring until its output is encoded, and
/// the decoded form is many times the input (a merge of 300 MB of compressed
/// OTLP logs with variant columns pinned about 6 GB). Several large merges
/// landing together can take the entire ring, and then every read that
/// needs a buffer, the merges' own footer loads and the ingest included,
/// fails with nothing left to evict. One commit's merges at a time keeps
/// that footprint bounded: one merge of whatever it holds, or several
/// small-file merges of no more than
/// `MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER` of inputs between them. A
/// deployment that has the memory to spare raises it in its config.
pub const DEFAULT_COMPACT_PARALLELISM: usize = 1;

/// Default size of the clique a routine round merges, and the cap on a
/// final sweep's connected group: one more than the bar's reciprocal, the
/// narrowest merge that keeps the layout's scales apart. A clique merge's
/// outputs each span about `1 / clique_size` of the inputs' range, so a
/// leftover input holds at most `total / width` of its row groups inside any
/// output, below the bar's `total * percent / 100 + 1`; the extra one is a
/// full row group of slack against cut drift. Narrower merges let leftovers
/// and outputs merge across scales (files of four row groups need only one
/// contested group and break at width four). Wider merges settle rows in
/// fewer rewrites but hold every input's decoded rows in memory.
pub const DEFAULT_LAYOUT_CLIQUE_SIZE: usize = 100 / overlap::CONTESTED_ROW_GROUP_PERCENT_BAR + 1;

/// Default for the most bytes the inputs of one small-file merge hold.
pub fn default_max_merge_input_bytes(target_bytes: u64) -> u64 {
    (target_bytes as f64 * DEFAULT_MAX_MERGE_INPUT_MULTIPLIER) as u64
}

/// Tuning for a datastore's self-managed compaction loop.
#[derive(Clone)]
pub struct CompactionConfig {
    /// Compaction output target. Half of it is the boundary between small-file
    /// and layout compaction; a single encoded row group may exceed it (see
    /// [`DEFAULT_COMPACT_BYTES`]).
    pub target_bytes: u64,
    /// The most bytes the inputs of one small-file merge hold.
    pub max_merge_input_bytes: u64,
    /// The fewest small files one merge takes, at least two.
    pub min_files: usize,
    /// How many commits' worth of merges a round may have in flight at
    /// once: a layout merge, or the small-file merges of one commit, which
    /// are up to `MAX_SMALL_FILE_MERGES_PER_COMMIT` merges written at the
    /// same time, of `MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER` of
    /// inputs between them when there are several. At one, a round may
    /// still be writing that many merges at once. Selection stays serial
    /// whatever this is; only the rewrites overlap.
    pub max_concurrent_merges: usize,
    /// Files one layout merge rewrites together, at least two; see
    /// [`DEFAULT_LAYOUT_CLIQUE_SIZE`].
    pub layout_clique_size: usize,
    /// How often to re-check the tables' metadata for newly-accumulated files.
    pub poll_interval: Duration,
}

/// Default cadence for reloading the tables from the store. This bounds how
/// stale a query's view of externally committed data in a shared remote store
/// is, so it trades freshness against the store's listing traffic.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// What background maintenance a [`PivotlakeDatastore`] runs for itself once opened.
/// The datastore spawns its own tasks from this on the ambient tokio runtime;
/// they run until the datastore is stopped.
#[derive(Clone)]
pub struct MaintenanceConfig {
    /// How often the datastore reloads its tables from the store (new
    /// versions and new files' footers), so externally committed data in a
    /// shared remote store becomes visible to queries.
    pub refresh_interval: Duration,
    /// Compaction settings, or `None` to run no compacter (a read-only server
    /// or a remote-store deployment where another process owns compaction).
    pub compaction: Option<CompactionConfig>,
    /// Vacuum settings, or `None` to run no vacuumer (a read-only server, or a
    /// remote-store deployment where another process owns physical cleanup).
    /// Deletes tombstoned data files and superseded commit JSONs past their
    /// retention.
    pub vacuum: Option<crate::vacuum::VacuumConfig>,
}

/// Handle to a datastore's single compaction actor. Both periodic maintenance
/// and explicit `COMPACT` statements run through that actor, so two compaction
/// rounds in one process never overlap. Within one round, selection is still
/// serial while the rewrites it starts run concurrently.
#[derive(Clone)]
pub struct CompacterHandle {
    commands: mpsc::UnboundedSender<CompactionCommand>,
}

/// Which pass a round runs: the routine one, which keeps every guard, or
/// the final sweep, which drops them and settles the table.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum CompactionPass {
    Routine,
    Final,
}

struct CompactionCommand {
    name: SchemaQualifiedTableName,
    /// The identity of the table to compact.
    id: Uuid,
    pass: CompactionPass,
    result: oneshot::Sender<Result<u64>>,
}

pub struct CompacterActor {
    /// Output-size target; half of it is the small/layout boundary.
    target_bytes: u64,
    max_merge_input_bytes: u64,
    min_files: usize,
    /// How many commits' worth of merges a round may have in flight at
    /// once; see [`CompactionConfig::max_concurrent_merges`].
    max_concurrent_merges: usize,
    /// Files one layout merge rewrites together, at least two; see
    /// [`DEFAULT_LAYOUT_CLIQUE_SIZE`].
    layout_clique_size: usize,
    /// How often to re-check the tables' metadata, or `None` when only
    /// explicit commands drive this actor.
    poll_interval: Option<Duration>,
    datastore: Arc<PivotlakeDatastore>,
    commands: mpsc::UnboundedReceiver<CompactionCommand>,
    /// Raised when the actor is to stop: it starts no merge afterwards, and
    /// returns once the merges in flight have finished.
    stop: StopSignal,
    /// What the rounds keep of each table from one to the next: its files,
    /// and what they measured of them.
    table_files: HashMap<Uuid, index::TableFileLists>,
}

/// What a round over one table came to.
struct RoundOutcome {
    passes: u64,
    /// How many partitions the round examined.
    examined_partitions: usize,
}

impl CompacterHandle {
    /// The handle and the actor it drives, over `datastore`. The caller spawns
    /// `CompacterActor::run` on its runtime.
    pub fn new(
        target_bytes: u64,
        max_merge_input_bytes: u64,
        min_files: usize,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        poll_interval: Option<Duration>,
        datastore: Arc<PivotlakeDatastore>,
    ) -> (Self, CompacterActor) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self { commands: sender },
            CompacterActor {
                target_bytes,
                max_merge_input_bytes,
                min_files: min_files.max(2),
                max_concurrent_merges: max_concurrent_merges.max(1),
                layout_clique_size: layout_clique_size.max(2),
                poll_interval,
                datastore,
                commands: receiver,
                stop: StopSignal::never(),
                table_files: HashMap::new(),
            },
        )
    }

    /// Enqueue a round over the table with the identity `id`, known as
    /// `name`; the round works from the copy of the table the datastore
    /// has published and keeps merging until nothing qualifies.
    /// `FINAL` drops the small-file guards and the contested bar and
    /// compacts each partition to one contiguous flow, see the crate docs.
    pub async fn compact(
        &self,
        name: SchemaQualifiedTableName,
        id: Uuid,
        final_sweep: bool,
    ) -> Result<u64> {
        let (result, answer) = oneshot::channel();
        let pass = if final_sweep {
            CompactionPass::Final
        } else {
            CompactionPass::Routine
        };
        self.commands
            .send(CompactionCommand {
                name,
                id,
                pass,
                result,
            })
            .map_err(|_| Error::CompacterStopped)?;
        answer.await.map_err(|_| Error::CompacterStopped)?
    }
}

impl CompacterActor {
    /// Have [`run`](Self::run) return once `stop` is raised. An actor given
    /// no signal runs until every handle to it is dropped or its task is
    /// cancelled.
    pub(crate) fn with_stop_signal(mut self, stop: StopSignal) -> Self {
        self.stop = stop;
        self
    }

    /// Run timer ticks and commands serially. The first configured tick fires
    /// immediately, preserving the background compacter's startup sweep, and
    /// each later one fires the poll interval after the previous round
    /// finished: a round runs until its tables have no candidate left, however
    /// long that takes, and the interval is the pause between rounds.
    ///
    /// Returns once the stop signal is raised. A round in flight then starts
    /// no further merge, waits for the ones it has running and ends with
    /// [`Error::CompacterStopped`], which is what its command answers; a
    /// command still queued is dropped unanswered, which its caller reads as
    /// the same error.
    pub async fn run(mut self) {
        let mut tick = self.poll_interval.map(|d| {
            let mut t = tokio::time::interval(d);
            t.set_missed_tick_behavior(MissedTickBehavior::Skip);
            t
        });

        loop {
            tokio::select! {
                biased;
                _ = self.stop.wait_until_raised() => return,
                command = self.commands.recv() => match command {
                    Some(command) => {
                        self.forget_dropped_tables();
                        let res = self
                            .compact_table(&command.name, command.id, command.pass)
                            .await;
                        let _ = command.result.send(res.map(|round| round.passes));
                    },
                    None => return,
                },
                _ = async {
                    match tick.as_mut() {
                        Some(t) => t.tick().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.compact_all().await;
                    if let Some(t) = tick.as_mut() {
                        t.reset();
                    }
                },
            }
        }
    }

    /// Forget what the rounds keep of the tables the datastore no longer
    /// holds.
    fn forget_dropped_tables(&mut self) {
        let datastore = &self.datastore;
        self.table_files
            .retain(|id, _| datastore.table_handle_by_id(id).is_some());
    }

    /// Run a routine round over each table the datastore holds, until the
    /// stop signal is raised.
    ///
    /// Only the name and identity of each table are kept across the rounds;
    /// each round reads the published copy as it needs it. A copy held
    /// while the rounds run would be a version the rounds' commits leave
    /// behind, kept in memory with all it holds: its metadata and, for a
    /// format whose version holds its files' row groups, those of every
    /// file the commits remove, each with its file open.
    async fn compact_all(&mut self) {
        self.forget_dropped_tables();
        let tables: Vec<(SchemaQualifiedTableName, Uuid)> = self
            .datastore
            .tables()
            .into_iter()
            .map(|(name, table)| (name, table.id()))
            .collect();
        for (name, id) in tables {
            if self.stop.is_raised() {
                return;
            }
            match self.compact_table(&name, id, CompactionPass::Routine).await {
                Ok(round) => debug!(
                    table = %name,
                    partitions = round.examined_partitions,
                    passes = round.passes,
                    "compaction round examined the table"
                ),
                Err(Error::CompacterStopped) => return,
                Err(error) => error!(table = %name, error = %error, "compaction round failed"),
            }
        }
    }

    /// Compact the table `id` until it has no candidate left: its first
    /// selection lists the partitions the round examines, then each pass
    /// starts a small-file group if one qualifies, else the best
    /// overlapping large-file group. A small-file group of a routine round
    /// is selected with further merges of its partition's small files, up
    /// to [`MAX_SMALL_FILE_MERGES_PER_COMMIT`] in all
    /// ([`MergeSelection::pick_small_file_runs`]), written at the same
    /// time and swapped in by one commit ([`MergeBatch::write_merges`]).
    /// Up to `max_concurrent_merges` of those groups are in flight at once,
    /// each holding the inputs of all its merges reserved until it commits,
    /// so selection works around the merges already in flight. A pass with
    /// nothing to start and nothing in flight ends the round.
    ///
    /// A group one of whose merges failed commits the merges it wrote and
    /// hands the failure back, and the round ends with it once the groups
    /// still in flight have finished: the failed merge's inputs are still
    /// in the table, so a further selection would pick them again. No
    /// partition with a merge in flight is ever settled, so the next round
    /// examines the partition again.
    ///
    /// A routine round examines the partitions a file joined or left since
    /// a round last found nothing to merge in them, every partition of a
    /// table no round examined; a final one examines them all.
    ///
    /// The round keeps no copy of the table. Each pass brings the table's
    /// lists to the copy the datastore has published by then, which holds
    /// every merge the round committed, and every file an INSERT of this
    /// process committed meanwhile, and selects from them: a file landing
    /// in a partition the round has not finished joins it, any other waits
    /// for the next round. What another process commits waits for the
    /// datastore's periodic refresh to publish it: the round has the table
    /// read from storage only after a commit conflict, another writer
    /// having swapped a selected file out first, and carries on from what
    /// that refresh publishes, with the partitions it examines listed
    /// anew. A refresh that leaves the table at the
    /// version the conflicting merge was selected from would have the
    /// round select that merge again, so the round ends there
    /// ([`Error::CompactionConflictWithoutProgress`]).
    ///
    /// Once the actor's stop signal is raised the round starts no further
    /// group: it waits for the ones in flight, which write every merge they
    /// hold and commit them, and ends with [`Error::CompacterStopped`].
    ///
    /// [`MAX_SMALL_FILE_MERGES_PER_COMMIT`]: small_files::MAX_SMALL_FILE_MERGES_PER_COMMIT
    async fn compact_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        id: Uuid,
        pass: CompactionPass,
    ) -> Result<RoundOutcome> {
        let mut passes = 0;
        let mut in_flight = JoinSet::new();
        let mut progress = RoundProgress::default();
        // Every exit with a merge in flight drains first: a merge body runs
        // on the blocking pool, which a dropped handle does not stop.
        loop {
            passes += 1;

            while in_flight.len() < self.max_concurrent_merges {
                if self.stop.is_raised() {
                    drain_merges(&mut in_flight).await;
                    return Err(Error::CompacterStopped);
                }
                let (selected, batch) = self.select_next_batch(id, pass, progress).await;
                progress = selected;
                let batch = match batch {
                    Ok(Some(batch)) => batch,
                    Ok(None) => break,
                    Err(error) => {
                        drain_merges(&mut in_flight).await;
                        return Err(error);
                    }
                };
                progress.reserved.extend(batch.list_reserved_paths());
                progress.merging.push(batch.partition.clone());
                in_flight.spawn(compact_batch(
                    Arc::clone(&self.datastore),
                    name.clone(),
                    id,
                    batch,
                    self.build_merge_selection(),
                ));
            }

            if in_flight.is_empty() {
                let examined = progress.partitions;
                return Ok(RoundOutcome {
                    passes,
                    examined_partitions: examined.map_or(0, |partitions| partitions.len()),
                });
            }

            let merge = match in_flight.join_next().await {
                Some(Ok(merge)) => merge,
                Some(Err(error)) => {
                    drain_merges(&mut in_flight).await;
                    return Err(Error::CompactionJobPanicked(error.to_string()));
                }
                None => unreachable!("the set is not empty"),
            };
            for path in &merge.reserved_paths {
                progress.reserved.remove(path);
            }
            let finished_merge = progress
                .merging
                .iter()
                .position(|merging| merging == &merge.partition)
                .expect("a merge in flight is counted in its partition");
            progress.merging.swap_remove(finished_merge);
            match merge.committed {
                // The commit published the version the next pass selects
                // from, which brings the lists to it.
                Ok(()) => {}
                Err(Error::CommitConflict { file, .. }) => {
                    // The merge's outputs are already deleted; have the table
                    // refreshed to see what the conflicting writer left.
                    let table = match self.refresh_published_table(id) {
                        Ok(table) => table,
                        Err(error) => {
                            drain_merges(&mut in_flight).await;
                            return Err(error);
                        }
                    };
                    if table.version() == merge.selected_at_version {
                        drain_merges(&mut in_flight).await;
                        return Err(Error::CompactionConflictWithoutProgress {
                            table: name.to_string(),
                            version: merge.selected_at_version,
                            file,
                        });
                    }
                    // The refreshed table may hold work in a partition the
                    // round was done with, or had no reason to examine: the
                    // next selection lists the round's partitions anew.
                    progress.finished.clear();
                    progress.partitions = None;
                }
                Err(error) => {
                    drain_merges(&mut in_flight).await;
                    return Err(error);
                }
            }
        }
    }

    /// The sizes this compacter selects its merges by.
    fn build_merge_selection(&self) -> MergeSelection {
        MergeSelection {
            target_bytes: self.target_bytes,
            max_merge_input_bytes: self.max_merge_input_bytes,
            min_files: self.min_files,
            layout_clique_size: self.layout_clique_size,
        }
    }

    /// The copy of the table `id` the datastore has published.
    fn find_published_table(&self, id: Uuid) -> Result<CatalogTable> {
        self.datastore
            .table_handle_by_id(&id)
            .ok_or_else(|| Error::TableNotFound(id.to_string()))
    }

    /// Have the datastore refresh the table `id` from storage and publish
    /// it, and hand back the published copy.
    fn refresh_published_table(&self, id: Uuid) -> Result<CatalogTable> {
        self.datastore.refresh_table_from_store(&id)?;
        self.find_published_table(id)
    }

    /// The next merge a pass can start in the table `id` as the datastore
    /// has it published (see [`MergeSelection::next_batch`]), selected on
    /// the blocking pool: selection brings the table's lists to the
    /// published version, which may read the table's metadata, and, for a
    /// layout pair it has to measure, reads footers over the worker pool,
    /// neither of which belongs on an async task. Hands `progress` back as
    /// the selection left it.
    async fn select_next_batch(
        &mut self,
        id: Uuid,
        pass: CompactionPass,
        mut progress: RoundProgress,
    ) -> (RoundProgress, Result<Option<MergeBatch>>) {
        let table = match self.find_published_table(id) {
            Ok(table) => table,
            Err(error) => return (progress, Err(error)),
        };
        let selection = self.build_merge_selection();
        let mut table_files = std::mem::take(self.table_files.entry(id).or_default());
        let selected = tokio::task::spawn_blocking(move || {
            let batch = selection.next_batch(&mut table_files, &table, pass, &mut progress);
            (table_files, progress, batch)
        })
        .await;
        match selected {
            Ok((table_files, progress, batch)) => {
                self.table_files.insert(id, table_files);
                (progress, batch)
            }
            Err(error) => (
                RoundProgress::default(),
                Err(Error::CompactionJobPanicked(error.to_string())),
            ),
        }
    }
}

/// Where a round stands among the partitions it examines.
#[derive(Default)]
struct RoundProgress {
    /// The partitions the round examines, in the order it does; `None`
    /// until a selection lists them: the round's first, and the first
    /// after a commit conflict.
    partitions: Option<Vec<PartitionHandle>>,
    /// The partitions the round found nothing left to do in.
    finished: HashSet<PartitionHandle>,
    /// The partition of each merge in flight.
    merging: Vec<PartitionHandle>,
    /// The inputs of the merges in flight, which no second merge may
    /// claim.
    reserved: HashSet<ObjectPath>,
}

/// The sizes a round selects its merges by.
#[derive(Clone, Copy)]
struct MergeSelection {
    target_bytes: u64,
    max_merge_input_bytes: u64,
    min_files: usize,
    layout_clique_size: usize,
}

impl MergeSelection {
    /// The next merge a pass can start, in the first partition that has
    /// one: a small-file group if one qualifies, else the best overlapping
    /// large-file group, grown from the most contested pair. No merge
    /// claims a file `progress` holds reserved.
    ///
    /// The pass first brings `table_files` to the version `table` is at,
    /// the copy the datastore has published, and lists the partitions the
    /// round examines when `progress` holds none
    /// ([`index::TableFileLists::list_round_partitions`]). It then looks at
    /// those partitions one at a time and never at one `progress` records
    /// as finished. A partition with nothing to start and no merge in
    /// flight is finished, and is settled until a file joins or leaves
    /// it.
    fn next_batch(
        &self,
        table_files: &mut index::TableFileLists,
        table: &CatalogTable,
        pass: CompactionPass,
        progress: &mut RoundProgress,
    ) -> Result<Option<MergeBatch>> {
        table_files.apply_manifests(table, self.target_bytes)?;
        let RoundProgress {
            partitions,
            finished,
            merging,
            reserved,
        } = progress;
        let partitions = match partitions.take() {
            Some(listed) => partitions.insert(listed),
            None => partitions.insert(table_files.list_round_partitions(pass)),
        };
        for partition in partitions.iter() {
            if finished.contains(partition) {
                continue;
            }
            // A partition the table no longer has a file in holds no merge.
            let Some(partition_files) = table_files.find_partition(partition) else {
                finished.insert(partition.clone());
                continue;
            };
            if let Some(merges) = self.next_small_merges(partition_files, pass, reserved) {
                let kind = MergeKind::SmallFiles;
                let batch = MergeBatch::new(table, partition.clone(), merges, kind);
                return Ok(Some(batch));
            }
            let group = partition_files.select_layout_group(
                table,
                reserved,
                pass,
                self.layout_clique_size,
            )?;
            if let Some(inputs) = group {
                let kind = MergeKind::LayoutOptimization;
                let batch = MergeBatch::new(table, partition.clone(), vec![inputs], kind);
                return Ok(Some(batch));
            }
            if merging.contains(partition) {
                continue;
            }
            partition_files.settle();
            finished.insert(partition.clone());
        }
        Ok(None)
    }
}

/// Await the merges still in flight, reporting rather than propagating what
/// they return: a merge body runs on the blocking pool, where dropping its
/// handle does not stop it, so a round that is giving up still waits for its
/// own rewrites to finish before the next one selects.
async fn drain_merges(in_flight: &mut JoinSet<MergeOutcome>) {
    while let Some(joined) = in_flight.join_next().await {
        match joined.map(|merge| merge.committed) {
            Ok(Err(error)) => warn!(error = %error, "compaction merge failed"),
            Err(error) => warn!(error = %error, "compaction merge panicked"),
            // The next round brings the lists to the version a merge
            // committed here.
            Ok(Ok(())) => {}
        }
    }
}

/// Integer file sizes below the mathematical half-target are genuinely small.
/// `div_ceil` makes a five-byte target classify sizes 0, 1, and 2 as small,
/// while an exact half is always full enough for layout work.
pub fn is_small_file(size: u64, target_bytes: u64) -> bool {
    size < target_bytes.div_ceil(2)
}

#[cfg(test)]
mod tests;
