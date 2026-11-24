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
//! # Which partitions a round examines
//!
//! The compacter remembers, per table, how far among the table's commits
//! its rounds examined it: a round that ends with nothing left to merge has
//! examined every file committed up to the version it began on. The next
//! routine round examines only the partitions a later commit added a file
//! to, an insert's or a merge's output alike, which the table's format
//! reads from the metadata written since; a partition nothing was added to
//! is not walked, and a format that holds its files a partition at a time
//! does not load it. A format that keeps no record of what a commit added
//! tells every partition, and a round over it examines them all. Nothing
//! is remembered when the compacter starts, so its first round examines
//! every partition once. `COMPACT ... FINAL` examines every partition, one
//! after another. No round examines a partition whose files the table's
//! format does not rewrite: an Iceberg table's partition written under a
//! partition spec other than the table's default.
//!
//! # What the rounds remember of a partition's layout
//!
//! Per partition, the rounds remember which files they measured and the
//! pairs of them that clear the contested bar, each with its share
//! ([`overlap::OverlapCache`]), so a round over a partition measures only
//! the files no round measured before, and a round over an unchanged
//! partition measures nothing. A pair below the bar is not remembered: a
//! final sweep, which merges those too, measures every pair of a partition
//! again when it reaches it, and lets go of the ones below the bar when it
//! is done with the partition. A file that is gone is forgotten by the next
//! round that examines its partition, and a partition left without a file
//! a layout merge may rewrite leaves nothing remembered. A partition the
//! table no longer has is forgotten by the next round that lists every
//! partition of the table, its first or a final sweep, and a table the
//! datastore no longer holds by the next round over any table.
//!
//! # Small files
//!
//! A partition's small files are merged once their total reaches the merge
//! target, the file target with headroom for compression gains. A pile that
//! reaches the file-count threshold before the byte target is merged once
//! it is balanced: the batch must total at least five times its largest
//! file, each file weighed with a fixed cost so that many tiny files
//! qualify, and the largest file is dropped and the test repeated until the
//! rest passes or nothing useful remains. `COMPACT ... FINAL` drops both
//! guards and merges any two or more small files.
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
//! manifest bounds or no rows, pairs whose ranges are disjoint or merely
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
//! scanning), and the merge (read, upload, swap) runs through
//! [`compact_table_files`]. A table in a bucket compacts through the exact
//! same code path as a local one.
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
//! copy of it: it has the datastore refresh the table from storage and
//! publish it as it begins, and reads the published copy whenever it lists
//! partitions or selects a merge. A merge looks its inputs up in that same
//! copy and its commit publishes the next version, so a round never selects
//! a file its merge cannot find. Configured timer ticks and explicit
//! `COMPACT` commands share the actor's queue, so their rewrites never race
//! in one process. The actor stops before the datastore's dispatch pool
//! shuts down.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ops::{ControlFlow, Range};
use std::sync::Arc;
use std::time::Duration;

use object_storage::{FileRef, ObjectPath};
use parquet_engine::file_entry::FileEntry;
use planner::catalog::SchemaQualifiedTableName;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::table_format::RemovedFiles;
use crate::{CatalogTable, Error, PivotlakeDatastore, Result, TableFile};

mod overlap;

/// Rows per row group in a merged file.
pub const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default compacted-file size target. Files smaller than half this size enter
/// small-file compaction; files at least half this size are full enough to
/// enter layout optimization instead.
pub const DEFAULT_COMPACT_BYTES: u64 = 1024 * 1024 * 1024;

/// Default for how many times the desired output size the inputs of one
/// small-file merge may hold. A merge writes as many output files as its
/// rows fill, so several files that each nearly reach the desired size, and
/// that no smaller group could merge, are rewritten together.
pub const DEFAULT_MERGE_TARGET_MULTIPLIER: f64 = 5.0;

/// Default for the fewest small files one merge takes: any two.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 2;

/// The most small files one merge takes. A merge holds rows of every input
/// file in memory at once, so its memory grows with their number. A larger
/// pile is merged in steps.
pub const MAX_SMALL_FILES_PER_MERGE: usize = 10;

/// A merge takes more than [`MAX_SMALL_FILES_PER_MERGE`] files while they
/// hold no more than these bytes and are no more than this many: files this
/// small take little memory each, and merging few of them at a time removes
/// fewer files than a busy table gains.
const MAX_BYTES_OF_A_MERGE_OF_MANY_FILES: u64 = 64 * 1024 * 1024;
const MAX_FILES_OF_A_MERGE_OF_FEW_BYTES: usize = 100;

/// The most merges of small files a round swaps into the table in one
/// commit. A commit costs the same whatever it holds, so a table that gains
/// small files quickly loses them faster when one commit swaps in several
/// merges.
pub const MAX_SMALL_FILE_MERGES_PER_COMMIT: usize = 15;

/// The most bytes of inputs the merges of one commit hold between them when
/// there are several: they are written at the same time, each holding rows
/// of its inputs in memory. A merge alone may hold more.
pub const MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER: u64 = 100 * 1024 * 1024;

/// What merging one more file costs whatever its size, added to each file's
/// bytes wherever a merge of small files is weighed. Without it many
/// similarly small files would be refused because their bytes are few.
const FIXED_FILE_COST_BYTES: f64 = (5 * 1024 * 1024) as f64;

/// How many times the size of its largest file a run of small files must be
/// to merge, in a partition with few small files. A run of equally sized
/// files then needs this many files. The more small files the partition has,
/// the less it asks, down to [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`].
const MIN_BALANCE_RATIO: f64 = 5.0;
const MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION: f64 = 2.0;

/// The counts of small files in a partition between which the balance a merge
/// must have falls from [`MIN_BALANCE_RATIO`] to
/// [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`].
const SMALL_FILES_TO_START_LOWERING_BALANCE: f64 = 10.0;
const SMALL_FILES_OF_A_CROWDED_PARTITION: f64 = 50.0;

/// A run whose files have all waited is asked for less balance too, down to
/// [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`], so the files left in a
/// partition that gains no more are merged in the end. The balance starts to
/// fall once the run's youngest file is this many seconds old.
const AGE_SECONDS_TO_START_LOWERING_BALANCE: f64 = 10.0;

/// How old the youngest file of a run is when the run is asked for the least
/// balance: this many seconds for a run of [`BYTES_OF_A_RUN_THAT_AGES_FASTEST`]
/// or fewer, and for a run of the desired file size or more, with the
/// logarithm of the run's bytes in between. Rewriting many bytes early costs
/// more than rewriting few.
const AGE_SECONDS_OF_AN_OLD_RUN_OF_FEW_BYTES: f64 = 10.0 * 60.0;
const AGE_SECONDS_OF_AN_OLD_RUN_OF_A_WHOLE_FILE: f64 = 6.0 * 3600.0;
const BYTES_OF_A_RUN_THAT_AGES_FASTEST: f64 = (1024 * 1024) as f64;

/// A run whose merged file would be about the size of the file before it is
/// preferred, because merging it leaves two neighbours of one size for a
/// later merge. The preference applies when the file before is larger than
/// this share of the merged file, and within this many powers of two of it;
/// a run that merges to exactly its size has its score multiplied by
/// [`BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE`].
const MIN_SHARE_OF_THE_FILE_BEFORE_A_RUN: f64 = 0.9;
const MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE: f64 = 0.5;
const BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE: f64 = 0.75;

/// How many input bytes a merge is taken to read for each byte it writes:
/// a file carries metadata and dictionaries of its own that a merged file
/// holds once, so a merged file is smaller than its inputs together.
const INPUT_BYTES_PER_MERGED_BYTE: f64 = 1.3;

/// Default cadence for re-checking the tables' metadata. Candidates only
/// change when a commit produces a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(30);

/// Default number of merges a datastore runs at once. A merge holds a few
/// decoded row groups of each of its input files on the memory ring, plus the
/// output row groups being encoded and written, and decoded rows are many
/// times their compressed size. Merges of many files landing together can
/// take a large part of the ring from the reads that need buffers, the
/// ingest included. One merge at a time keeps that footprint bounded; a
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

/// Default accumulated input size for a small-file merge.
pub fn default_merge_target_bytes(target_bytes: u64) -> u64 {
    (target_bytes as f64 * DEFAULT_MERGE_TARGET_MULTIPLIER) as u64
}

/// Tuning for a datastore's self-managed compaction loop.
#[derive(Clone)]
pub struct CompactionConfig {
    /// Compaction output target. Half of it is the boundary between small-file
    /// and layout compaction; a single encoded row group may exceed it (see
    /// [`DEFAULT_COMPACT_BYTES`]).
    pub target_bytes: u64,
    /// Accumulated small-file bytes that trigger a merge.
    pub merge_target_bytes: u64,
    /// Count at which a sub-target pile may take the balance-check path.
    pub min_files: usize,
    /// How many groups of merges a round may have committing while it merges
    /// the next one. One group is merged at a time whatever this is.
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
/// they stop when its dispatch pool begins shutting down.
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
    merge_target_bytes: u64,
    min_files: usize,
    /// How many merges a round may have in flight at once.
    max_concurrent_merges: usize,
    /// Files one layout merge rewrites together, at least two; see
    /// [`DEFAULT_LAYOUT_CLIQUE_SIZE`].
    layout_clique_size: usize,
    /// How often to re-check the tables' metadata, or `None` when only
    /// explicit commands drive this actor.
    poll_interval: Option<Duration>,
    datastore: Arc<PivotlakeDatastore>,
    commands: mpsc::UnboundedReceiver<CompactionCommand>,
    /// What the rounds remember of each table from one to the next.
    table_states: HashMap<Uuid, TableCompactionState>,
}

/// What the rounds over one table remember from one to the next.
#[derive(Default)]
struct TableCompactionState {
    /// What the rounds know of the layout of each partition they examined
    /// that holds a file a layout merge may rewrite, so a pass only measures
    /// the files no round measured before (see [`overlap::OverlapCache`]).
    overlaps: PartitionOverlaps,
    /// Where the version a round began on stood among the table's commits,
    /// for the last round that ended with nothing left to merge: no file a
    /// commit up to there added has a merge waiting. `None` until a round
    /// has ended so.
    examined_through_commit: Option<u64>,
}

/// What a round over one table came to.
struct RoundOutcome {
    passes: u64,
    /// The partitions the round examined, as a walk names them.
    examined: Vec<Vec<u8>>,
}

/// What the rounds know of the layout of one table's partitions, by the
/// partition, as a walk names it.
type PartitionOverlaps = HashMap<Vec<u8>, overlap::OverlapCache>;

impl CompacterHandle {
    /// The handle and the actor it drives, over `datastore`. The caller spawns
    /// `CompacterActor::run` on its runtime.
    pub fn new(
        target_bytes: u64,
        merge_target_bytes: u64,
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
                merge_target_bytes,
                min_files: min_files.max(2),
                max_concurrent_merges: max_concurrent_merges.max(1),
                layout_clique_size: layout_clique_size.max(2),
                poll_interval,
                datastore,
                commands: receiver,
                table_states: HashMap::new(),
            },
        )
    }

    /// Enqueue a round over the table with the identity `id`, known as
    /// `name`; the round has the datastore refresh the table to its latest
    /// version as it begins and keeps merging until nothing qualifies.
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
    /// Run timer ticks and commands serially. The first configured tick fires
    /// immediately, preserving the background compacter's startup sweep, and
    /// each later one fires the poll interval after the previous round
    /// finished: a round runs until its tables have no candidate left, however
    /// long that takes, and the interval is the pause between rounds.
    pub async fn run(mut self) {
        let mut tick = self.poll_interval.map(|d| {
            let mut t = tokio::time::interval(d);
            t.set_missed_tick_behavior(MissedTickBehavior::Skip);
            t
        });

        loop {
            tokio::select! {
                biased;
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

    /// Forget what the rounds remember of the tables the datastore no
    /// longer holds.
    fn forget_dropped_tables(&mut self) {
        let datastore = &self.datastore;
        self.table_states
            .retain(|id, _| datastore.table_handle_by_id(id).is_some());
    }

    async fn compact_all(&mut self) {
        self.forget_dropped_tables();
        // Only the names and identities are kept: a copy of a table held
        // across its round keeps every file the round's commits remove in the
        // caches.
        let tables: Vec<(SchemaQualifiedTableName, Uuid)> = self
            .datastore
            .tables()
            .into_iter()
            .map(|(name, table)| (name, table.id()))
            .collect();
        for (name, id) in tables {
            match self.compact_table(&name, id, CompactionPass::Routine).await {
                Ok(round) => debug!(
                    table = %name,
                    partitions = round.examined.len(),
                    passes = round.passes,
                    "compaction round examined the table"
                ),
                Err(error) => error!(table = %name, error = %error, "compaction round failed"),
            }
        }
    }

    /// Compact the table `id` until it has no candidate left: the round has
    /// the datastore refresh the table from storage and publish it once as
    /// it begins, then each pass starts a small-file group if one qualifies,
    /// else the best overlapping large-file group. A small-file group is
    /// joined by further merges of small files of the same partition, written
    /// at the same time as it, and one commit swaps them all in
    /// ([`MAX_SMALL_FILE_MERGES_PER_COMMIT`],
    /// [`MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER`]). One group is merged
    /// at a time. The next is selected as soon as the group before starts its
    /// commit, while up to `max_concurrent_merges` groups are committing, and
    /// a committing group holds its inputs reserved, so selection works
    /// around them. A pass with nothing to start and nothing in
    /// flight ends the round, and the table is then remembered as examined
    /// up to the version the round listed its partitions from.
    ///
    /// A routine round examines the partitions a commit added a file to
    /// since the table was last examined, every partition when it never
    /// was; a final one examines them all. A round that lists every
    /// partition of the table forgets what is remembered of any other.
    ///
    /// The round keeps no copy of the table. Each pass selects from the copy
    /// the datastore has published by then, which holds every merge the
    /// round committed, and every file an INSERT of this process committed
    /// meanwhile: one landing in a partition the round has not finished
    /// joins it, any other waits for the next round. What another process
    /// commits meanwhile waits too: the table is read from storage again
    /// only after a commit conflict, another writer having swapped a
    /// selected file out first, and the round carries on from what that
    /// refresh publishes. A refresh that leaves the table at the version
    /// the conflicting merge was selected from would have the round select
    /// that merge again, so the round ends there
    /// ([`Error::CompactionConflictWithoutProgress`]).
    async fn compact_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        id: Uuid,
        pass: CompactionPass,
    ) -> Result<RoundOutcome> {
        let mut passes = 1;
        let mut in_flight = JoinSet::new();
        // What the group being merged sends once its merges are written: the
        // inputs its commit removes.
        let mut merging: Option<oneshot::Receiver<Vec<ObjectPath>>> = None;
        // Whether a group one of whose merges failed has yet to be joined:
        // the run that failed is not reserved, and would be selected again.
        let mut is_failed_group_in_flight = false;
        let mut reserved = HashSet::new();
        let examined_through_commit = self
            .table_states
            .get(&id)
            .and_then(|state| state.examined_through_commit);
        let listed = {
            // The copy is let go of once the partitions are listed: a version
            // held for the round's length keeps every file the round's commits
            // remove in the caches.
            let table = self.refresh_published_table(id)?;
            list_round_partitions(&table, pass, examined_through_commit).await?
        };
        self.forget_partitions_not_among(id, &listed);
        let mut listed_at = listed.commit_position;
        let mut progress = RoundProgress {
            partitions: listed.partitions,
            ..RoundProgress::default()
        };
        loop {
            if merging.is_none()
                && !is_failed_group_in_flight
                && in_flight.len() <= self.max_concurrent_merges
            {
                let (selected, batch) = self.select_next_batch(id, pass, &reserved, progress).await;
                progress = selected;
                let batch = match batch {
                    Ok(batch) => batch,
                    Err(error) => {
                        drain_merges(&mut in_flight).await;
                        return Err(error);
                    }
                };
                if let Some(batch) = batch {
                    progress.merging.push(batch.partition.clone());
                    let (merged_inputs, merges_written) = oneshot::channel();
                    merging = Some(merges_written);
                    in_flight.spawn(compact_batch(
                        Arc::clone(&self.datastore),
                        name.clone(),
                        id,
                        batch,
                        self.build_merge_selection(),
                        merged_inputs,
                    ));
                }
            }

            if in_flight.is_empty() {
                self.table_states
                    .entry(id)
                    .or_default()
                    .examined_through_commit = Some(listed_at);
                return Ok(RoundOutcome {
                    passes,
                    examined: progress.partitions,
                });
            }

            // Every exit from here on drains first: a merge body runs on the
            // blocking pool, which a dropped handle does not stop.
            let joined = tokio::select! {
                // A group that has finished has sent its inputs or dropped
                // its sender, so what it sent is read before it is joined.
                biased;
                merges_written = async {
                    merging
                        .as_mut()
                        .expect("the branch is taken while a group is merging")
                        .await
                }, if merging.is_some() => {
                    // A group one of whose merges failed sends nothing, and
                    // hands its failure back when it is joined.
                    merging = None;
                    match merges_written {
                        Ok(inputs) => reserved.extend(inputs),
                        Err(_) => is_failed_group_in_flight = true,
                    }
                    continue;
                }
                joined = in_flight.join_next() => joined,
            };
            let merge = match joined {
                Some(Ok(merge)) => merge,
                Some(Err(error)) => {
                    drain_merges(&mut in_flight).await;
                    return Err(Error::CompactionJobPanicked(error.to_string()));
                }
                None => unreachable!("the set is not empty"),
            };
            passes += 1;
            if merge.has_failed_merge {
                is_failed_group_in_flight = false;
            }
            for path in &merge.reserved_paths {
                reserved.remove(path);
            }
            let finished_merge = progress
                .merging
                .iter()
                .position(|merging| merging == &merge.partition)
                .expect("a merge in flight is counted in its partition");
            progress.merging.swap_remove(finished_merge);
            match merge.committed {
                // The commit published the version the next pass selects from.
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
                    // round was done with, or had no reason to examine.
                    progress.finished.clear();
                    let listed =
                        match list_round_partitions(&table, pass, examined_through_commit).await {
                            Ok(listed) => listed,
                            Err(error) => {
                                drain_merges(&mut in_flight).await;
                                return Err(error);
                            }
                        };
                    self.forget_partitions_not_among(id, &listed);
                    listed_at = listed.commit_position;
                    progress.partitions = listed.partitions;
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
            merge_target_bytes: self.merge_target_bytes,
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

    /// Forget what is remembered of the partitions of the table `id` that
    /// are not among `listed`, when the round listed every partition of the
    /// table: the table no longer has a file a rewrite may replace in any
    /// other.
    fn forget_partitions_not_among(&mut self, id: Uuid, listed: &RoundPartitions) {
        if !listed.lists_every_partition {
            return;
        }
        let Some(state) = self.table_states.get_mut(&id) else {
            return;
        };
        let listed: HashSet<&[u8]> = listed.partitions.iter().map(Vec::as_slice).collect();
        state
            .overlaps
            .retain(|partition, _| listed.contains(partition.as_slice()));
    }

    /// The next merge a pass can start in the table `id` as the datastore
    /// has it published (see [`MergeSelection::next_batch`]), selected on
    /// the blocking pool: selection walks the files of the partitions it
    /// looks at and, for a layout pair it has to measure, reads footers over
    /// the worker pool, neither of which belongs on an async task. Hands
    /// `progress` back as the selection left it.
    async fn select_next_batch(
        &mut self,
        id: Uuid,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
        mut progress: RoundProgress,
    ) -> (RoundProgress, Result<Option<MergeBatch>>) {
        let table = match self.find_published_table(id) {
            Ok(table) => table,
            Err(error) => return (progress, Err(error)),
        };
        let selection = self.build_merge_selection();
        let mut overlaps = std::mem::take(&mut self.table_states.entry(id).or_default().overlaps);
        let reserved = reserved.clone();
        let selected = tokio::task::spawn_blocking(move || {
            let batch = selection.next_batch(&mut overlaps, &table, pass, &reserved, &mut progress);
            (overlaps, progress, batch)
        })
        .await;
        match selected {
            Ok((overlaps, progress, batch)) => {
                self.table_states.entry(id).or_default().overlaps = overlaps;
                (progress, batch)
            }
            Err(error) => (
                RoundProgress::default(),
                Err(Error::CompactionJobPanicked(error.to_string())),
            ),
        }
    }
}

/// The partitions a round examines.
struct RoundPartitions {
    /// The partitions, as a walk names them, in the order the round
    /// examines them.
    partitions: Vec<Vec<u8>>,
    /// Whether the round listed every partition of the table whose files a
    /// rewrite may replace, and not only those a commit added a file to.
    lists_every_partition: bool,
    /// Where the version the partitions were listed from stands among the
    /// table's commits: a file a later commit adds may sit in a partition
    /// that is not listed.
    commit_position: u64,
}

/// The partitions a round examines of `table`, the copy the datastore
/// published of the version just refreshed from storage, in the order the
/// round examines them: those a commit added a file to since the one
/// standing at `examined_through_commit`, for a routine round over a table
/// examined before; else every partition. A partition whose files the
/// table's format does not rewrite is in neither. Listing them may read
/// the table's metadata, so it runs on the blocking pool.
async fn list_round_partitions(
    table: &CatalogTable,
    pass: CompactionPass,
    examined_through_commit: Option<u64>,
) -> Result<RoundPartitions> {
    let table = table.clone();
    let listed = tokio::task::spawn_blocking(move || {
        let added_after = match (pass, examined_through_commit) {
            (CompactionPass::Routine, Some(commit_position)) => Some(commit_position),
            _ => None,
        };
        let partitions = match added_after {
            Some(commit_position) => table.list_partitions_added_after(commit_position),
            None => table.list_partitions(),
        }?;
        Ok::<_, Error>(RoundPartitions {
            partitions: table.select_rewritable_partitions(partitions)?,
            lists_every_partition: added_after.is_none(),
            commit_position: table.commit_position(),
        })
    })
    .await;
    listed.map_err(|error| Error::CompactionJobPanicked(error.to_string()))?
}

/// How many of the partitions a round is about to look at are indexed
/// together.
const PARTITIONS_PER_INDEXING: usize = 64;

/// Where a round stands among the partitions it examines.
#[derive(Default)]
struct RoundProgress {
    /// The partitions the round examines, in the order it does.
    partitions: Vec<Vec<u8>>,
    /// The partitions the round found nothing left to do in.
    finished: HashSet<Vec<u8>>,
    /// The partition of each merge in flight.
    merging: Vec<Vec<u8>>,
}

/// A file below the target size, as the selection of a merge weighs it.
#[derive(Clone)]
struct SmallFile {
    file: FileRef,
    /// When the commit that added the file was made, in Unix milliseconds.
    committed_at_ms: u64,
}

/// The files the merges of one commit rewrite, all of one partition.
struct MergeBatch {
    /// The partition the inputs sit in, as a walk names it.
    partition: Vec<u8>,
    /// The inputs of each merge. No file is among those of two.
    merges: Vec<Vec<FileRef>>,
    kind: MergeKind,
    /// The version of the table the inputs were selected from.
    selected_at_version: u64,
    /// The copy of the table the inputs were selected from.
    table: CatalogTable,
}

/// A run of small files one merge may take, as the picking of a commit's
/// runs orders it: the cheaper to merge is the greater, and of two as cheap
/// the one that starts later in the partition's order.
struct RunCandidate {
    /// The bytes the merge rewrites for each file it removes.
    score: f64,
    first: usize,
    end: usize,
}

impl Ord for RunCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.first.cmp(&other.first))
    }
}

impl PartialOrd for RunCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RunCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RunCandidate {}

/// What the merges of one commit wrote, before the commit.
struct WrittenMerges {
    /// The copy of the table the merges read.
    table: CatalogTable,
    /// The inputs of every merge written.
    removed_paths: Vec<ObjectPath>,
    /// The outputs of every merge written.
    outputs: Vec<TableFile>,
    /// Why a merge failed, the first one when several did.
    failure: Option<Error>,
}

impl WrittenMerges {
    /// Delete the outputs, which `error` keeps out of the table.
    fn delete_outputs(&self, error: &Error) {
        for file in &self.outputs {
            if let Err(cleanup_error) = self.table.delete_data_file(&file.file_ref().path) {
                warn!(
                    error = %error,
                    cleanup_error = %cleanup_error,
                    file = %file.file_ref().path,
                    "compaction: deleting an output that will not be committed failed (orphan left)"
                );
            }
        }
    }
}

impl MergeBatch {
    /// The merges of one commit, selected from `table`.
    fn new(
        table: &CatalogTable,
        partition: Vec<u8>,
        merges: Vec<Vec<FileRef>>,
        kind: MergeKind,
    ) -> Self {
        Self {
            partition,
            merges,
            kind,
            selected_at_version: table.version(),
            table: table.clone(),
        }
    }

    /// Write the merges at the same time, each a merge of its own. Nothing
    /// is committed. A merge that fails leaves the others to be committed.
    ///
    /// Blocking store I/O, so callers run it on the blocking pool.
    fn write_merges(
        self,
        selection: &MergeSelection,
        name: &SchemaQualifiedTableName,
    ) -> WrittenMerges {
        let MergeBatch {
            partition,
            merges,
            kind,
            table,
            ..
        } = self;

        // Each merge blocks its thread on the dataflow that writes it, and
        // reads the table's store through the runtime of the thread that
        // called.
        let runtime = tokio::runtime::Handle::try_current().ok();
        let merged: Vec<Result<Vec<TableFile>>> = std::thread::scope(|scope| {
            let writers: Vec<_> = merges
                .iter()
                .map(|inputs| {
                    scope.spawn(|| {
                        let _in_runtime = runtime.as_ref().map(|runtime| runtime.enter());
                        table.merge_files(
                            &partition,
                            inputs,
                            ROW_GROUP_ROWS,
                            selection.target_bytes,
                        )
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| {
                    writer.join().unwrap_or_else(|_| {
                        Err(Error::CompactionJobPanicked("a merge panicked".to_string()))
                    })
                })
                .collect()
        });

        let mut written = WrittenMerges {
            table,
            removed_paths: Vec::new(),
            outputs: Vec::new(),
            failure: None,
        };
        for (inputs, outputs) in merges.into_iter().zip(merged) {
            let outputs = match outputs {
                Ok(outputs) => outputs,
                Err(error) => {
                    written.failure.get_or_insert(error);
                    continue;
                }
            };
            let input_sizes: Vec<u64> = inputs.iter().map(|file| file.size).collect();
            let output_sizes: Vec<u64> = outputs.iter().map(|file| file.file_ref().size).collect();
            info!(
                table = %name,
                kind = kind.label(),
                inputs = input_sizes.len(),
                input_sizes = ?input_sizes,
                input_bytes = input_sizes.iter().sum::<u64>(),
                outputs = output_sizes.len(),
                output_sizes = ?output_sizes,
                output_bytes = output_sizes.iter().sum::<u64>(),
                "compacted batch"
            );
            written
                .removed_paths
                .extend(inputs.into_iter().map(|file| file.path));
            written.outputs.extend(outputs);
        }
        written
    }
}

/// The sizes a round selects its merges by.
#[derive(Clone, Copy)]
struct MergeSelection {
    target_bytes: u64,
    merge_target_bytes: u64,
    min_files: usize,
    layout_clique_size: usize,
}

impl MergeSelection {
    /// The next merge a pass can start, in the first partition that has
    /// one: a small-file group if one qualifies, else the best overlapping
    /// large-file group, grown from the most contested pair. `reserved`
    /// holds the files the merges already in flight are rewriting, which no
    /// second merge may claim.
    ///
    /// A pass looks at the partitions `progress` lists for the round, one
    /// at a time and never at one it records as finished. A partition with
    /// nothing to start and no merge in flight is finished, and `overlaps`
    /// keeps of it what the rounds remember between them: nothing, when it
    /// holds no file a layout merge may rewrite. The partitions are indexed
    /// a few ahead of the look, so a pass over many partitions does not
    /// read the table's metadata again for each.
    fn next_batch(
        &self,
        overlaps: &mut PartitionOverlaps,
        table: &CatalogTable,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
        progress: &mut RoundProgress,
    ) -> Result<Option<MergeBatch>> {
        let unfinished: Vec<Vec<u8>> = progress
            .partitions
            .iter()
            .filter(|partition| !progress.finished.contains(*partition))
            .cloned()
            .collect();
        for partitions in unfinished.chunks(PARTITIONS_PER_INDEXING) {
            table.index_partitions(partitions)?;
            for partition in partitions.iter().cloned() {
                if let Some(merges) = self.next_small_merges(table, &partition, pass, reserved)? {
                    let batch = MergeBatch::new(table, partition, merges, MergeKind::SmallFiles);
                    return Ok(Some(batch));
                }
                let group =
                    self.next_layout_optimization(overlaps, table, &partition, reserved, pass)?;
                if let Some(inputs) = group {
                    let kind = MergeKind::LayoutOptimization;
                    return Ok(Some(MergeBatch::new(table, partition, vec![inputs], kind)));
                }
                if progress.merging.contains(&partition) {
                    continue;
                }
                if let Some(cache) = overlaps.get_mut(&partition) {
                    cache.forget_pairs_below_bar();
                    if cache.is_empty() {
                        overlaps.remove(&partition);
                    }
                }
                progress.finished.insert(partition);
            }
        }
        Ok(None)
    }

    /// Pick the merges one commit makes of the files of `partition` below
    /// the target size, the inputs of each: in a routine round the runs of
    /// neighbours cheapest to merge
    /// ([`pick_small_file_runs`](Self::pick_small_file_runs)), in a final
    /// sweep one merge of the first files of the partition, as many as a
    /// merge takes.
    fn next_small_merges(
        &self,
        table: &CatalogTable,
        partition: &[u8],
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Result<Option<Vec<Vec<FileRef>>>> {
        let Some(files) = table.load_partition_files(partition)? else {
            return Ok(None);
        };
        let mut small_file_refs: Vec<FileRef> = Vec::new();
        let mut commit_sequences: Vec<u64> = Vec::new();
        let _ = files.visit_files(&mut |file| {
            if is_small_file(file.file().size, self.target_bytes)
                && !reserved.contains(&file.file().path)
            {
                small_file_refs.push(file.file().clone());
                commit_sequences.push(file.commit_sequence());
            }
            ControlFlow::Continue(())
        });
        if small_file_refs.len() < 2 {
            return Ok(None);
        }
        let merges: Vec<Vec<FileRef>> = match pass {
            CompactionPass::Routine => {
                let small_files: Vec<SmallFile> = small_file_refs
                    .into_iter()
                    .zip(table.read_commit_times_ms(&commit_sequences))
                    .map(|(file, committed_at_ms)| SmallFile {
                        file,
                        committed_at_ms,
                    })
                    .collect();
                self.pick_small_file_runs(
                    &small_files,
                    crate::vacuum::now_unix_ms(),
                    MAX_SMALL_FILE_MERGES_PER_COMMIT,
                )
                .into_iter()
                .map(|run| {
                    small_files[run]
                        .iter()
                        .map(|small| small.file.clone())
                        .collect()
                })
                .collect()
            }
            CompactionPass::Final => {
                small_file_refs.truncate(MAX_SMALL_FILES_PER_MERGE);
                vec![small_file_refs]
            }
        };
        Ok((!merges.is_empty()).then_some(merges))
    }

    /// How many times the size of its largest file a run of `total_bytes`
    /// must be to merge, when its youngest file was committed `age_seconds`
    /// ago in a partition `crowding` of the way from few small files to
    /// many. The crowding and the run's age each take the balance asked
    /// from [`MIN_BALANCE_RATIO`] toward
    /// [`MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION`], and together no
    /// further than that.
    fn find_min_balance_ratio(&self, crowding: f64, total_bytes: u64, age_seconds: f64) -> f64 {
        let fewest_bytes = (1.0 + BYTES_OF_A_RUN_THAT_AGES_FASTEST).ln();
        let bytes_span = (1.0 + self.target_bytes as f64).ln() - fewest_bytes;
        let size_position = if bytes_span > 0.0 {
            (((1.0 + total_bytes as f64).ln() - fewest_bytes) / bytes_span).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let age_of_an_old_run = AGE_SECONDS_OF_AN_OLD_RUN_OF_FEW_BYTES
            + (AGE_SECONDS_OF_AN_OLD_RUN_OF_A_WHOLE_FILE - AGE_SECONDS_OF_AN_OLD_RUN_OF_FEW_BYTES)
                * size_position;
        let aging = ((age_seconds - AGE_SECONDS_TO_START_LOWERING_BALANCE)
            / (age_of_an_old_run - AGE_SECONDS_TO_START_LOWERING_BALANCE))
            .clamp(0.0, 1.0);
        let lowering = (crowding + aging).min(1.0);
        MIN_BALANCE_RATIO
            + (MIN_BALANCE_RATIO_OF_A_CROWDED_PARTITION - MIN_BALANCE_RATIO) * lowering
    }

    /// The runs of neighbours among `files`, the small files of one partition
    /// in its order, that the merges of one commit take, a run each, in the
    /// order they are picked: at most `max_runs` of them, of no more than
    /// [`MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER`] between them, the
    /// first whatever it holds. None when no run may merge.
    ///
    /// Each run is the one cheapest to merge for the files it removes, of
    /// the runs that hold no file of a run picked before it. Every run of at
    /// least two neighbours whose bytes fit one merged file is weighed, of
    /// at most `MAX_SMALL_FILES_PER_MERGE` files, or of at most
    /// `MAX_FILES_OF_A_MERGE_OF_FEW_BYTES` that hold no more than
    /// `MAX_BYTES_OF_A_MERGE_OF_MANY_FILES` bytes. A run may merge when it
    /// has the configured number of files and is balanced: its bytes are
    /// enough times those of its largest file, so a large file is not
    /// rewritten to absorb a few small ones. A partition with many small
    /// files asks for less balance, so it merges sooner, and so does a run
    /// whose files have all waited
    /// ([`find_min_balance_ratio`](Self::find_min_balance_ratio)). Of the
    /// runs that may merge, the one with the fewest bytes rewritten per file
    /// removed is taken, and of equals the last in the partition's order. A
    /// run that would merge to about the size of the file before it counts
    /// as cheaper than it is.
    ///
    /// The files are gone through once, for the best run that starts at
    /// each. The best of those is picked, and the runs that start at a file
    /// are weighed again only when the best of them holds a file a picked
    /// run took.
    fn pick_small_file_runs(
        &self,
        files: &[SmallFile],
        now_ms: u64,
        max_runs: usize,
    ) -> Vec<Range<usize>> {
        let crowding = ((files.len() as f64 - SMALL_FILES_TO_START_LOWERING_BALANCE)
            / (SMALL_FILES_OF_A_CROWDED_PARTITION - SMALL_FILES_TO_START_LOWERING_BALANCE))
            .clamp(0.0, 1.0);

        let mut is_taken = vec![false; files.len()];
        let mut candidates: BinaryHeap<RunCandidate> = (0..files.len())
            .filter_map(|first| self.find_best_run_from(files, first, &is_taken, crowding, now_ms))
            .collect();
        let mut runs: Vec<Range<usize>> = Vec::new();
        let mut input_bytes = 0;
        while runs.len() < max_runs {
            let Some(candidate) = candidates.pop() else {
                break;
            };
            let run = candidate.first..candidate.end;
            if is_taken[run.clone()].contains(&true) {
                // What is left of the runs that start at the file, none when
                // the file itself was taken.
                candidates.extend(self.find_best_run_from(
                    files,
                    candidate.first,
                    &is_taken,
                    crowding,
                    now_ms,
                ));
                continue;
            }
            let run_bytes: u64 = files[run.clone()].iter().map(|small| small.file.size).sum();
            let is_first_run = runs.is_empty();
            if !is_first_run && input_bytes + run_bytes > MAX_INPUT_BYTES_OF_MERGES_WRITTEN_TOGETHER
            {
                break;
            }
            input_bytes += run_bytes;
            is_taken[run.clone()].fill(true);
            runs.push(run);
        }
        runs
    }

    /// The run cheapest to merge of those that start at `files[first]` and
    /// hold no taken file, if any may merge, weighed as
    /// [`pick_small_file_runs`](Self::pick_small_file_runs) says: of runs as
    /// cheap, the longest.
    fn find_best_run_from(
        &self,
        files: &[SmallFile],
        first: usize,
        is_taken: &[bool],
        crowding: f64,
        now_ms: u64,
    ) -> Option<RunCandidate> {
        if is_taken[first] {
            return None;
        }
        let mut best: Option<RunCandidate> = None;
        let mut total = files[first].file.size;
        let mut largest = files[first].file.size;
        let mut youngest_committed_at_ms = files[first].committed_at_ms;
        for end in first + 2..=files.len() {
            if is_taken[end - 1] {
                break;
            }
            total += files[end - 1].file.size;
            largest = largest.max(files[end - 1].file.size);
            youngest_committed_at_ms = youngest_committed_at_ms.max(files[end - 1].committed_at_ms);
            let count = end - first;
            let has_too_many_files = count > MAX_FILES_OF_A_MERGE_OF_FEW_BYTES
                || (count > MAX_SMALL_FILES_PER_MERGE
                    && total > MAX_BYTES_OF_A_MERGE_OF_MANY_FILES);
            if has_too_many_files || total > self.merge_target_bytes {
                break;
            }
            if count < self.min_files {
                continue;
            }
            let balance_ratio = (total as f64 + FIXED_FILE_COST_BYTES * count as f64)
                / (largest as f64 + FIXED_FILE_COST_BYTES);
            let age_seconds = now_ms.saturating_sub(youngest_committed_at_ms) as f64 / 1000.0;
            if balance_ratio < self.find_min_balance_ratio(crowding, total, age_seconds) {
                continue;
            }

            let mut score =
                (total as f64 + FIXED_FILE_COST_BYTES * count as f64) / (count as f64 - 1.9);
            let merged_size = total as f64 / INPUT_BYTES_PER_MERGED_BYTE;
            let size_before = match first {
                0 => 0.0,
                _ => files[first - 1].file.size as f64,
            };
            if size_before > merged_size * MIN_SHARE_OF_THE_FILE_BEFORE_A_RUN {
                let powers_of_two_apart = (merged_size / size_before).log2().abs();
                if powers_of_two_apart < MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE {
                    let closeness =
                        1.0 - powers_of_two_apart / MAX_POWERS_OF_TWO_FROM_THE_FILE_BEFORE;
                    score *= 1.0
                        - (1.0 - BEST_SCORE_FACTOR_OF_A_RUN_MERGING_TO_THE_FILE_BEFORE) * closeness;
                }
            }

            if best.as_ref().is_none_or(|best| score <= best.score) {
                best = Some(RunCandidate { score, first, end });
            }
        }
        best
    }

    /// The next group of at-least-half-full files of `partition` one merge
    /// should rewrite together, chosen by what `overlaps` knows of the
    /// partition ([`overlap::OverlapCache`]): a full clique in a routine
    /// round, any connected group in a final sweep. Reserved files stay in
    /// view so their remembered overlaps survive a failed merge; selection
    /// passes their pairs over.
    fn next_layout_optimization(
        &self,
        overlaps: &mut PartitionOverlaps,
        table: &CatalogTable,
        partition: &[u8],
        reserved: &HashSet<ObjectPath>,
        pass: CompactionPass,
    ) -> Result<Option<Vec<FileRef>>> {
        if table.sort_by().is_empty() {
            return Ok(None);
        }

        let candidates = LayoutCandidates {
            table,
            partition,
            target_bytes: self.target_bytes,
        };
        let cache = overlaps.entry(partition.to_vec()).or_default();
        cache.observe(&candidates, table.sort_by(), pass)?;
        let Some(group) = cache.select_group(reserved, pass, self.layout_clique_size) else {
            return Ok(None);
        };
        let entries = overlap::LayoutSource::find_entries(&candidates, &group)?;
        Ok(Some(entries.into_iter().map(|entry| entry.file).collect()))
    }
}

/// One partition's files at least half the target size, as a layout
/// observation reads them: walked from the table's metadata, with a file's
/// entry built and its footer read only when the observation measures it.
struct LayoutCandidates<'a> {
    table: &'a CatalogTable,
    /// The partition, as a walk names it.
    partition: &'a [u8],
    target_bytes: u64,
}

impl overlap::LayoutSource for LayoutCandidates<'_> {
    fn walk_candidates(&self, visit: &mut overlap::VisitCandidate<'_>) -> Result<()> {
        let Some(files) = self.table.load_partition_files(self.partition)? else {
            return Ok(());
        };
        let _ = files.visit_records(&mut |record| {
            if !is_small_file(record.facts().file().size, self.target_bytes) {
                visit(self.partition, record);
            }
            ControlFlow::Continue(())
        });
        Ok(())
    }

    fn find_entries(&self, paths: &[Arc<ObjectPath>]) -> Result<Vec<FileEntry>> {
        let paths: Vec<ObjectPath> = paths.iter().map(|path| path.as_ref().clone()).collect();
        Ok(self
            .table
            .find_file_entries(self.partition, &paths)?
            .into_iter()
            .map(|entry| entry.expect("a walked file is live in the version walked"))
            .collect())
    }

    /// The footers are read for this call alone: only the bounds of their
    /// row groups on the sort columns leave it.
    fn read_row_group_ranges(&self, files: &[FileRef]) -> Result<Vec<overlap::RowGroupRanges>> {
        Ok(self
            .table
            .row_groups(files)?
            .iter()
            .map(|row_groups| overlap::read_row_group_ranges(row_groups, self.table.sort_by()))
            .collect())
    }
}

/// What the merges of one commit hand back.
struct MergeOutcome {
    /// The partition the inputs sit in, as a walk names it.
    partition: Vec<u8>,
    /// The input paths the round held reserved for the commit.
    reserved_paths: Vec<ObjectPath>,
    /// The version of the table the inputs were selected from.
    selected_at_version: u64,
    /// Whether a merge failed, so the round was sent no inputs.
    has_failed_merge: bool,
    /// Whether every merge was written and their swap committed, or why
    /// not. The merges written before one that failed are committed all the
    /// same.
    committed: Result<()>,
}

/// Write the merges of `batch` ([`MergeBatch::write_merges`]) and swap their
/// outputs into the table for their inputs in one commit labelled a
/// rearrangement, which publishes the version it wrote. The inputs are sent
/// through `merged_inputs` once the merges are written, before the commit, so
/// the round can select the next merges around them while the commit runs.
///
/// The merges run off a read copy of the table with no commit lock held, so
/// they never block a concurrent INSERT's commit; only the swap takes it.
/// When a merge fails, the merges written before it are committed all the
/// same and the failure is returned, with nothing sent through
/// `merged_inputs`: a run that keeps failing does not keep the runs picked
/// before it from merging. If the swap fails, the outputs are deleted
/// before the error is returned; a swap whose outcome is unknown may have
/// succeeded, so its outputs stay.
///
/// Returns the inputs' partition and paths beside the outcome so the round
/// can release their reservations however the merges went, and the version
/// they were selected from so the round can tell a conflict it cannot get
/// past.
async fn compact_batch(
    datastore: Arc<PivotlakeDatastore>,
    name: SchemaQualifiedTableName,
    id: Uuid,
    batch: MergeBatch,
    selection: MergeSelection,
    merged_inputs: oneshot::Sender<Vec<ObjectPath>>,
) -> MergeOutcome {
    let partition = batch.partition.clone();
    let selected_at_version = batch.selected_at_version;
    let written =
        match tokio::task::spawn_blocking(move || batch.write_merges(&selection, &name)).await {
            Ok(written) => written,
            Err(error) => {
                return MergeOutcome {
                    partition,
                    reserved_paths: Vec::new(),
                    selected_at_version,
                    has_failed_merge: true,
                    committed: Err(Error::CompactionJobPanicked(error.to_string())),
                };
            }
        };
    let reserved_paths = written.removed_paths.clone();
    let has_failed_merge = written.failure.is_some();
    let commit_partition = partition.clone();
    let committed = tokio::task::spawn_blocking(move || {
        let mut written = written;
        let failure = written.failure.take();
        match &failure {
            // The round only reads what is sent while it waits for this
            // commit.
            None => {
                let _ = merged_inputs.send(written.removed_paths.clone());
            }
            Some(_) => drop(merged_inputs),
        }
        if written.removed_paths.is_empty() {
            return failure.map_or(Ok(()), Err);
        }
        let removed = RemovedFiles {
            partition: &commit_partition,
            paths: &written.removed_paths,
        };
        // The swap goes through the datastore's one in-process write path,
        // under the table's commit lock.
        let committed = datastore.commit_to_table(id, removed, written.outputs.clone(), false);
        if let Err(error) = &committed
            && !error.may_have_committed()
        {
            written.delete_outputs(error);
        }
        committed.and(failure.map_or(Ok(()), Err))
    })
    .await
    .unwrap_or_else(|error| Err(Error::CompactionJobPanicked(error.to_string())));
    MergeOutcome {
        partition,
        reserved_paths,
        selected_at_version,
        has_failed_merge,
        committed,
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
            Ok(Ok(())) => {}
        }
    }
}

#[derive(Clone, Copy)]
enum MergeKind {
    SmallFiles,
    LayoutOptimization,
}

impl MergeKind {
    fn label(self) -> &'static str {
        match self {
            Self::SmallFiles => "small files",
            Self::LayoutOptimization => "layout optimization",
        }
    }
}

/// Integer file sizes below the mathematical half-target are genuinely small.
/// `div_ceil` makes a five-byte target classify sizes 0, 1, and 2 as small,
/// while an exact half is always full enough for layout work.
pub fn is_small_file(size: u64, target_bytes: u64) -> bool {
    size < target_bytes.div_ceil(2)
}

/// Merge `inputs` of the table with identity `id` into fresh target-sized files
/// and swap them in for the inputs: see `compact_partition_files`. `inputs`
/// must share one partition, which is found by walking the table's files to
/// the first of them: for a caller that names files without having walked
/// to them.
pub fn compact_table_files(
    datastore: &PivotlakeDatastore,
    id: Uuid,
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
    let partition = match inputs.first() {
        Some(input) => {
            table
                .find_partition_of(&input.path)?
                .ok_or_else(|| Error::CommitConflict {
                    location: table.location().to_string(),
                    file: input.path.to_string(),
                })?
        }
        None => Vec::new(),
    };
    compact_partition_files(
        datastore,
        id,
        &partition,
        inputs,
        target_rows_per_group,
        max_output_file_size,
    )
}

/// Merge `inputs`, files of `partition` (as a walk names it) of the table
/// with identity `id`, into fresh target-sized files and swap them in for
/// the inputs, in one commit labelled a rearrangement, which publishes the
/// version it wrote. Returns the merged files.
///
/// The merge itself (decode, re-encode, upload) runs off a plain read copy of
/// the table with no commit lock held, so it never blocks a concurrent INSERT's
/// commit; only the swap that follows takes it. If that swap fails for any
/// reason, including another writer having swapped the inputs out first, the
/// uncommitted merge outputs are deleted before the error is returned; a swap
/// whose outcome is unknown may have succeeded, so its outputs stay.
///
/// `max_output_file_size` is a compressed row-group body target. Parquet
/// headers and footers are excluded, and an individually oversized row group
/// is emitted alone.
///
/// Blocking store I/O, so callers run it on the blocking pool.
fn compact_partition_files(
    datastore: &PivotlakeDatastore,
    id: Uuid,
    partition: &[u8],
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
    let merged = table.merge_files(
        partition,
        inputs,
        target_rows_per_group,
        max_output_file_size,
    )?;

    let outputs = merged.clone();
    let removed_paths: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    let removed = RemovedFiles {
        partition,
        paths: &removed_paths,
    };
    // The swap goes through the datastore's one in-process write path, under
    // the table's commit lock.
    if let Err(error) = datastore.commit_to_table(id, removed, merged, false) {
        if error.may_have_committed() {
            return Err(error);
        }
        for file in &outputs {
            if let Err(cleanup_error) = table.delete_data_file(&file.file_ref().path) {
                warn!(
                    commit_error = %error,
                    cleanup_error = %cleanup_error,
                    file = %file.file_ref().path,
                    "compaction: deleting output after commit failure failed (orphan left)"
                );
            }
        }
        return Err(error);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::PivotlakeDatastore;
    use crate::file_entry::scalar_values_equal;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use object_storage::ObjectPath;
    use parquet::arrow::ArrowWriter;
    use parquet_engine::ParquetTable;
    use planner::catalog::SchemaQualifiedTableName;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    /// Write `values` as a small SNAPPY Parquet file (single Int64 `Timestamp`
    /// column) into `dir` under `file_name`. A table created over `dir` picks it
    /// up as a small input for the compacter to merge.
    fn write_parquet_file(dir: &Path, file_name: &str, values: Vec<i64>) {
        let rows = values.len().max(1);
        write_parquet_file_with_row_group_size(dir, file_name, values, rows);
    }

    /// A file whose rows are cut into row groups of `max_row_group_size`, so a
    /// test file of a few rows still carries the many row groups that layout
    /// selection counts.
    fn write_parquet_file_with_row_group_size(
        dir: &Path,
        file_name: &str,
        values: Vec<i64>,
        max_row_group_size: usize,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        // SNAPPY, like every pivot-written file -- the decompressor expects it.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_row_count(Some(max_row_group_size))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// A sorted range with incompressible non-key payloads. Two seeds write
    /// the same key range at almost the same byte size while retaining distinct
    /// rows, which models two 80%-full overlapping files without footer bytes
    /// dominating the test.
    fn write_wide_parquet_file(dir: &Path, file_name: &str, rows: usize, seed: u64) {
        write_wide_parquet_row_groups(dir, file_name, rows, seed, rows);
    }

    /// [`write_wide_parquet_file`] with the rows cut into row groups of
    /// `rows_per_group`.
    fn write_wide_parquet_row_groups(
        dir: &Path,
        file_name: &str,
        rows: usize,
        seed: u64,
        rows_per_group: usize,
    ) {
        let keys: Vec<i64> = (1..=rows as i64).collect();
        write_wide_parquet_keys(dir, file_name, keys, seed, rows_per_group);
    }

    /// [`write_wide_parquet_row_groups`] with the `Timestamp` of every row
    /// given, in the order the rows are stored.
    fn write_wide_parquet_keys(
        dir: &Path,
        file_name: &str,
        keys: Vec<i64>,
        seed: u64,
        rows_per_group: usize,
    ) {
        let rows = keys.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Payload0", DataType::Int64, false),
            Field::new("Payload1", DataType::Int64, false),
            Field::new("Payload2", DataType::Int64, false),
            Field::new("Payload3", DataType::Int64, false),
        ]));
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys))];
        for column in 0..4 {
            let payload: Vec<i64> = (0..rows)
                .map(|row| {
                    let mut value = seed ^ row as u64 ^ ((column as u64) << 48);
                    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    value ^= value >> 31;
                    value as i64
                })
                .collect();
            columns.push(Arc::new(Int64Array::from(payload)));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_partitioned_parquet_file(
        dir: &Path,
        file_name: &str,
        timestamps: Vec<i64>,
        partition: i64,
    ) {
        let row_count = timestamps.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Partition", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)) as _,
                Arc::new(Int64Array::from(vec![partition; row_count])) as _,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `CREATE TABLE <name> (Timestamp Int64) WITH (with_pre_existing_parquets = dir)`
    /// against `datastore`, the way the server would run it -- or, when `dir` is
    /// `None`, an empty table adopting nothing.
    fn create_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
        sorted: bool,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let mut options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        if sorted {
            options.insert("sort_by".to_string(), "Timestamp".to_string());
        }
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            options,
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn create_wide_sorted_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: &Path,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload0".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload1".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload2".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload3".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([
                (
                    "with_pre_existing_parquets".to_string(),
                    dir.to_str().unwrap().to_string(),
                ),
                ("sort_by".to_string(), "Timestamp".to_string()),
            ]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    /// Create an empty table partitioned by its `Partition` column.
    fn create_partitioned_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Partition".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([(
                "partition_by".to_string(),
                "Partition".to_string(),
            )]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn partition_value(value: i64) -> crate::PartitionValues {
        std::collections::HashMap::from([(
            "Partition".to_string(),
            Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
        )])
    }

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the datastore's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(datastore: &PivotlakeDatastore, name: &str) -> Arc<ParquetTable> {
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists");
        table.refresh().expect("manifest reload");
        table.select_scan(&[]).expect("select scan")
    }

    /// Drive one full compaction sweep to completion on a temporary runtime.
    fn run_one_sweep(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(
            target_bytes,
            DEFAULT_COMPACT_PARALLELISM,
            DEFAULT_LAYOUT_CLIQUE_SIZE,
            datastore,
        )
    }

    /// [`run_one_sweep`] with layout cliques kept to pairs, for scenarios
    /// where two files are the full clique.
    fn run_one_sweep_with_pair_layout(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(target_bytes, DEFAULT_COMPACT_PARALLELISM, 2, datastore)
    }

    /// [`run_one_sweep`] with the merges in flight and the layout merge width
    /// spelled out.
    fn run_one_sweep_with(
        target_bytes: u64,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        datastore: Arc<PivotlakeDatastore>,
    ) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let tables = datastore.tables();
                let (compacter, actor) = CompacterHandle::new(
                    target_bytes,
                    default_merge_target_bytes(target_bytes),
                    DEFAULT_MIN_FILES_TO_MERGE,
                    max_concurrent_merges,
                    layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                for (name, table) in tables {
                    compacter.compact(name, table.id(), false).await.unwrap();
                }
                actor.abort();
            });
    }

    /// The size knobs one compaction command runs under.
    struct CompacterSizes {
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        layout_clique_size: usize,
    }

    impl CompacterSizes {
        /// The defaults for `target_bytes`.
        fn defaults(target_bytes: u64) -> Self {
            Self {
                target_bytes,
                merge_target_bytes: default_merge_target_bytes(target_bytes),
                min_files: DEFAULT_MIN_FILES_TO_MERGE,
                layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
            }
        }

        /// The defaults with the small-file strategy switched off, so a
        /// scenario exercises layout optimization alone.
        fn without_small_files(target_bytes: u64) -> Self {
            Self {
                merge_target_bytes: u64::MAX,
                min_files: usize::MAX,
                ..Self::defaults(target_bytes)
            }
        }
    }

    fn run_command(
        sizes: CompacterSizes,
        datastore: Arc<PivotlakeDatastore>,
        name: SchemaQualifiedTableName,
        final_sweep: bool,
    ) -> u64 {
        let id = datastore.table_handle(&name).unwrap().id();
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let (compacter, actor) = CompacterHandle::new(
                    sizes.target_bytes,
                    sizes.merge_target_bytes,
                    sizes.min_files,
                    DEFAULT_COMPACT_PARALLELISM,
                    sizes.layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                let sweeps = compacter.compact(name, id, final_sweep).await.unwrap();
                actor.abort();
                sweeps
            })
    }

    /// A round has the datastore refresh the table before it selects, so a
    /// file committed through another copy of the table, which the
    /// datastore's own does not list, still joins the merge.
    #[test]
    fn a_round_refreshes_the_table_from_storage_before_selecting() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![1]);
        write_parquet_file(&adopted, "b.parquet", vec![2]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            false,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut writer = datastore.table_handle(&name).unwrap();
        // Each file is small (under half the target), so a final sweep
        // merges all the table has: one file is left only if the refresh
        // brings `c.parquet` in.
        let target = writer.file_refs().unwrap()[0].size * 22 / 10;
        write_parquet_file(&adopted, "c.parquet", vec![3]);
        writer
            .append_data_file(
                ObjectPath::new("c.parquet"),
                &std::fs::read(adopted.join("c.parquet")).unwrap(),
                None,
            )
            .unwrap();

        let sweeps = run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        assert_eq!(sweeps, 2, "one merging pass and one that finds nothing");
        assert_eq!(
            datastore
                .table_handle(&name)
                .unwrap()
                .file_refs()
                .unwrap()
                .len(),
            1
        );
        dispatch.exit();
    }

    /// A round does not stop at one pair: it keeps merging, selecting from
    /// the version each merge commits, until no large pair overlaps any more.
    #[test]
    fn a_round_keeps_merging_until_no_pair_overlaps() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);
        write_parquet_file(&adopted_dir, "d.parquet", vec![1_000, 1_100]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap().unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap().unwrap();
        assert!(
            files.iter().all(|file| !matches!(
                file.path.name(),
                "a.parquet" | "b.parquet" | "c.parquet" | "d.parquet"
            )),
            "both overlapping pairs were rewritten in one round: {files:?}"
        );
        dispatch.exit();
    }

    /// Three overlapping pairs are rewritten with three merges in flight. The
    /// reservations keep every merge off the others' inputs, so all swaps
    /// commit and every row survives.
    #[test]
    fn concurrent_merges_never_share_an_input() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        for (file, rows) in [
            ("a.parquet", vec![0, 100]),
            ("b.parquet", vec![0, 100]),
            ("c.parquet", vec![1_000, 1_100]),
            ("d.parquet", vec![1_000, 1_100]),
            ("e.parquet", vec![2_000, 2_100]),
            ("f.parquet", vec![2_000, 2_100]),
        ] {
            write_parquet_file(&adopted_dir, file, rows);
        }
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap().unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with(target, 3, 2, datastore.clone());

        let files = datastore.table_files(&name).unwrap().unwrap();
        assert!(
            files
                .iter()
                .all(|file| file.path.name().starts_with("pivot-")),
            "all three pairs were rewritten: {files:?}"
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 12, "no input was merged twice or dropped");
        dispatch.exit();
    }

    #[test]
    fn final_command_bypasses_small_file_guards() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file, partition) in [("a", 1), ("b", 1), ("c", 2), ("d", 2)] {
            let file = format!("{file}.parquet");
            write_partitioned_parquet_file(encoded.path(), &file, vec![partition], partition);
            table
                .append_data_file(
                    ObjectPath::new(&file),
                    &std::fs::read(encoded.path().join(&file)).unwrap(),
                    Some(partition_value(partition)),
                )
                .unwrap();
        }
        datastore.refresh_from_store().unwrap();
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .unwrap()
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);

        let routine_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            false,
        );
        let routine_files = datastore
            .table_handle(&name)
            .unwrap()
            .file_refs()
            .unwrap()
            .len();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        assert_eq!(routine_sweeps, 1);
        assert_eq!(routine_files, 4);
        assert_eq!(final_sweeps, 3);
        assert_eq!(
            datastore
                .table_handle(&name)
                .unwrap()
                .file_refs()
                .unwrap()
                .len(),
            2
        );
        dispatch.exit();
    }

    /// The row-group rule is the whole layout bar: a pair merges when more
    /// than a fifth of each file's row groups overlaps the other file, and
    /// files that only reach each other's boundary value never merge, not even
    /// under the final command.
    #[test]
    fn layout_merges_need_overlapping_row_groups_on_both_sides() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        // One row per row group. The first two files share seven of their
        // twenty groups; the last two meet at 119 and share nothing.
        write_parquet_file_with_row_group_size(&adopted, "a.parquet", (0..20).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "b.parquet", (12..32).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "c.parquet", (100..120).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "d.parquet", (119..139).collect(), 1);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input out of the small-file path.
        let target = table
            .file_refs()
            .unwrap()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);

        run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            false,
        );
        let after_routine = datastore.table_handle(&name).unwrap();
        let routine_files: Vec<String> = after_routine
            .file_refs()
            .unwrap()
            .iter()
            .map(|file| file.path.name().to_string())
            .collect();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            true,
        );

        assert_eq!(routine_files.len(), 3);
        assert!(routine_files.contains(&"c.parquet".to_string()));
        assert!(routine_files.contains(&"d.parquet".to_string()));
        assert_eq!(
            final_sweeps, 1,
            "the final command finds nothing the routine round left behind"
        );
        assert_eq!(
            datastore
                .table_handle(&name)
                .unwrap()
                .file_refs()
                .unwrap()
                .len(),
            3
        );
        dispatch.exit();
    }

    /// End-to-end compaction: several registered small files merge into one, the
    /// datastore swaps to the merged file in one log commit, and it decodes back to
    /// all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_datastore() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![6, 7, 8, 9]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .unwrap()
                .len(),
            3
        );

        let total: u64 = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        run_one_sweep(total, datastore.clone());

        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .unwrap()
                .len(),
            1
        );
        let parquet = fresh_parquet(&datastore, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );

        dispatch.exit();
    }

    /// Small-file compaction uses the same output target as layout work. A
    /// three-file batch whose encoded row groups need two target-sized files is
    /// split rather than becoming one oversized replacement.
    #[test]
    fn small_file_compaction_splits_outputs_at_target() {
        const ROWS_PER_INPUT: usize = 64_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "a.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "b.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "c.parquet", ROWS_PER_INPUT, 47);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs().unwrap();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        assert!(inputs.iter().all(|file| is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs().unwrap();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let row_count: i64 = compacted
            .select_scan(&[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        dispatch.exit();
    }

    #[test]
    fn layout_optimization_merges_the_highest_overlap_large_pair() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap().unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap().unwrap();
        assert_eq!(files.len(), 2);
        assert!(
            files.iter().any(|file| file.path.name() == "c.parquet"),
            "the disjoint file is not part of the chosen pair"
        );
        assert!(
            files
                .iter()
                .all(|file| file.path.name() != "a.parquet" && file.path.name() != "b.parquet")
        );

        dispatch.exit();
    }

    /// Two 80%-full files are layout work, not small-file work. Their globally
    /// sorted rewrite is too large for one target file, so it becomes two
    /// balanced, disjoint ranges which neither strategy selects again.
    #[test]
    fn overlapping_eighty_percent_files_become_two_stable_range_files() {
        // Exactly four normal compaction row groups across the two inputs, so
        // the encoded result has a natural balanced boundary for two files.
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs().unwrap();
        assert_eq!(inputs.len(), 2);
        let largest_input = inputs.iter().map(|file| file.size).max().unwrap();
        let target = largest_input.saturating_mul(5).div_ceil(4);
        for input in &inputs {
            let fullness = input.size as f64 / target as f64;
            assert!(
                (0.75..=0.81).contains(&fullness),
                "input {} is {:.1}% of target",
                input.path,
                fullness * 100.0
            );
        }

        // Verify that an effectively unbounded encoding, not merely the sum of the
        // input sizes, requires two files under this target.
        let partition = table.list_partitions().unwrap().remove(0);
        let unbounded = table
            .merge_files(&partition, &inputs, ROW_GROUP_ROWS, u64::MAX)
            .unwrap();
        assert_eq!(unbounded.len(), 1);
        let unbounded_size = unbounded[0].file_ref().size;
        for file in unbounded {
            table.delete_data_file(&file.file_ref().path).unwrap();
        }
        assert!(unbounded_size > target);
        assert!(unbounded_size <= target * 2);

        let before_version = table.version();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs().unwrap();
        assert_eq!(
            outputs.len(),
            2,
            "target={target}, unbounded={unbounded_size}, outputs={outputs:?}"
        );
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );

        let mut ranges: Vec<(i64, i64)> = compacted
            .live_entries()
            .unwrap()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        assert!(ranges[0].1 < ranges[1].0, "ranges are disjoint: {ranges:?}");

        let row_count: i64 = compacted
            .select_scan(&[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (2 * ROWS_PER_INPUT) as i64);

        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            2
        );
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("add").is_some())
                .count(),
            2
        );

        let stable_version = compacted.version();
        let stable_paths: std::collections::HashSet<_> =
            outputs.iter().map(|file| file.path.clone()).collect();
        run_one_sweep_with_pair_layout(target, datastore.clone());
        let after_normal = datastore.table_handle(&name).unwrap();
        assert_eq!(after_normal.version(), stable_version);
        assert_eq!(
            after_normal
                .file_refs()
                .unwrap()
                .into_iter()
                .map(|file| file.path)
                .collect::<std::collections::HashSet<_>>(),
            stable_paths
        );
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name,
            true,
        );
        assert_eq!(final_sweeps, 1);
        assert_eq!(
            datastore.table_handle_by_id(&table.id()).unwrap().version(),
            stable_version
        );

        dispatch.exit();
    }

    /// Three mutually overlapping 80%-full files are exactly the configured
    /// clique, so a single merge and one commit rewrite all of them together
    /// into sorted, range-disjoint files that no later pass selects again.
    #[test]
    fn layout_optimization_merges_a_group_of_three_in_one_pass() {
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "first.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "second.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "third.parquet", ROWS_PER_INPUT, 47);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs().unwrap();
        assert_eq!(inputs.len(), 3);
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);
        assert!(inputs.iter().all(|file| !is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(
            compacted.version(),
            before_version + 1,
            "one merge rewrites all three inputs"
        );
        let outputs = compacted.file_refs().unwrap();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        let mut ranges: Vec<(i64, i64)> = compacted
            .live_entries()
            .unwrap()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        for window in ranges.windows(2) {
            assert!(window[0].1 < window[1].0, "ranges are disjoint: {ranges:?}");
        }
        let row_count: i64 = compacted
            .select_scan(&[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            3
        );

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        assert_eq!(
            datastore.table_handle(&name).unwrap().version(),
            compacted.version()
        );

        dispatch.exit();
    }

    /// Two overlapping large files are less than the clique of three, so a
    /// routine round stands them deliberately; the final sweep settles them.
    #[test]
    fn routine_round_stands_a_partial_clique_and_the_final_sweep_settles_it() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .unwrap()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap();
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 3,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            true,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 3);
        let settled = datastore.table_handle(&name).unwrap();
        assert_eq!(settled.version(), before_version + 1);
        assert!(
            settled
                .file_refs()
                .unwrap()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 4);
        dispatch.exit();
    }

    /// A wide file covers several mutually disjoint narrow files, none of
    /// which contests more than a fifth of its row groups: no pair clears
    /// the bar, so routine rounds stand still, and the final sweep's relaxed
    /// connected group settles the whole neighborhood into files that share
    /// no stretch of the key.
    #[test]
    fn final_sweep_settles_a_hub_no_routine_round_can_see() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file_with_row_group_size(&adopted, "hub.parquet", (0..20).collect(), 1);
        write_parquet_file(&adopted, "s1.parquet", vec![2, 3]);
        write_parquet_file(&adopted, "s2.parquet", vec![8, 9]);
        write_parquet_file(&adopted, "s3.parquet", vec![14, 15]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input, satellites included, out of the small-file path.
        let target = table
            .file_refs()
            .unwrap()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 4, "relaxed merging converges quickly");
        let settled = datastore.table_handle(&name).unwrap();
        assert!(settled.version() > before_version);
        assert!(
            settled
                .file_refs()
                .unwrap()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let mut ranges: Vec<(i64, i64)> = settled
            .live_entries()
            .unwrap()
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        for window in ranges.windows(2) {
            assert!(
                window[0].1 <= window[1].0,
                "no two files share a stretch of the key: {ranges:?}"
            );
        }
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 26);
        dispatch.exit();
    }

    /// Two inputs of many row groups each, covering the same keys, merge
    /// into one file of many row groups that holds every row in key order.
    #[test]
    fn a_merge_of_many_row_group_inputs_keeps_every_row_in_order() {
        const ROWS_PER_INPUT: usize = 40_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_row_groups(&adopted, "left.parquet", ROWS_PER_INPUT, 11, 3_000);
        write_wide_parquet_row_groups(&adopted, "right.parquet", ROWS_PER_INPUT, 29, 3_000);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();

        let outputs = table
            .merge_files(
                &table.list_partitions().unwrap().remove(0),
                &table.file_refs().unwrap(),
                5_000,
                u64::MAX,
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let output = std::fs::File::open(
            db.path()
                .join(table.location())
                .join(outputs[0].file_ref().path.as_str()),
        )
        .unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(output).unwrap();
        assert_eq!(reader.metadata().num_row_groups(), 16);
        let keys: Vec<i64> = reader
            .build()
            .unwrap()
            .flat_map(|batch| {
                batch
                    .unwrap()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        let expected: Vec<i64> = (1..=ROWS_PER_INPUT as i64)
            .flat_map(|key| [key, key])
            .collect();
        assert_eq!(keys, expected);
        dispatch.exit();
    }

    /// Write a file of `(Timestamp, Doc)` rows whose `Doc` is the variant
    /// `{"age": <timestamp>}`.
    fn write_variant_parquet_file(dir: &Path, file_name: &str, timestamps: Vec<i64>) {
        use arrow_array::StringArray;
        let documents: ArrayRef = Arc::new(StringArray::from(
            timestamps
                .iter()
                .map(|timestamp| format!(r#"{{"age":{timestamp}}}"#))
                .collect::<Vec<_>>(),
        ));
        let variant = parquet_variant_compute::json_to_variant(&documents).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            variant.field("Doc"),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(variant.into_inner()),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// A merge of files with a variant column reads only that column to plan
    /// the output's layout, shreds it, and keeps every row in key order.
    #[test]
    fn a_merge_shreds_a_variant_column_from_a_sample_of_it() {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_variant_parquet_file(
            &adopted,
            "odd.parquet",
            (0..500).map(|n| 2 * n + 1).collect(),
        );
        write_variant_parquet_file(&adopted, "even.parquet", (0..500).map(|n| 2 * n).collect());
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "events".to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Doc".to_string(),
                    col_type: planner::types::Type::Variant,
                },
            ],
            options: std::collections::HashMap::from([
                (
                    "with_pre_existing_parquets".to_string(),
                    adopted.to_str().unwrap().to_string(),
                ),
                ("sort_by".to_string(), "Timestamp".to_string()),
            ]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatch.dispatcher())
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();

        let outputs = table
            .merge_files(
                &table.list_partitions().unwrap().remove(0),
                &table.file_refs().unwrap(),
                300,
                u64::MAX,
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let output = std::fs::File::open(
            db.path()
                .join(table.location())
                .join(outputs[0].file_ref().path.as_str()),
        )
        .unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(output).unwrap();
        let document_fields = match reader.schema().field(1).data_type() {
            DataType::Struct(fields) => fields.clone(),
            other => panic!("a variant column is a struct, not {other:?}"),
        };
        assert!(document_fields.find("typed_value").is_some());
        let timestamps: Vec<i64> = reader
            .build()
            .unwrap()
            .flat_map(|batch| {
                batch
                    .unwrap()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(timestamps, (0..1_000).collect::<Vec<i64>>());
        dispatch.exit();
    }

    /// A file whose rows are not in the table's sort order cannot be merged
    /// into sorted output, so the merge fails instead of writing a file.
    #[test]
    fn a_merge_of_an_input_out_of_sort_order_fails() {
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "sorted.parquet", 1_000, 11);
        let descending: Vec<i64> = (1..=1_000).rev().collect();
        write_wide_parquet_keys(&adopted, "unsorted.parquet", descending, 29, 1_000);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();

        let outcome = table.merge_files(
            &table.list_partitions().unwrap().remove(0),
            &table.file_refs().unwrap(),
            500,
            u64::MAX,
        );

        let error = outcome.err().expect("the merge fails").to_string();
        assert!(error.contains("not sorted"), "{error}");
        dispatch.exit();
    }

    /// A table without sort columns keeps its rows in file order, and an
    /// input row group can hold rows of two output row groups.
    #[test]
    fn a_merge_without_sort_columns_cuts_input_row_groups_across_output_ones() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        let values: Vec<i64> = (0..14).collect();
        write_parquet_file_with_row_group_size(&adopted, "rows.parquet", values.clone(), 7);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            false,
        );
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();

        let outputs = table
            .merge_files(
                &table.list_partitions().unwrap().remove(0),
                &table.file_refs().unwrap(),
                5,
                u64::MAX,
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        let output = std::fs::File::open(
            db.path()
                .join(table.location())
                .join(outputs[0].file_ref().path.as_str()),
        )
        .unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(output).unwrap();
        assert_eq!(reader.metadata().num_row_groups(), 3);
        let merged: Vec<i64> = reader
            .build()
            .unwrap()
            .flat_map(|batch| {
                batch
                    .unwrap()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(merged, values);
        dispatch.exit();
    }

    /// The byte target is applied only between independently encoded row
    /// groups. A group that crosses it is emitted alone without changing the
    /// normal compaction row-group size or re-encoding the input.
    #[test]
    fn one_oversized_row_group_remains_one_output_file() {
        const ROWS_PER_INPUT: usize = 8_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        let inputs = table.file_refs().unwrap();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);

        let partition = table.list_partitions().unwrap().remove(0);

        let outputs = table
            .merge_files(&partition, &inputs, ROW_GROUP_ROWS, target)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].file_ref().size > target);
        for output in outputs {
            table.delete_data_file(&output.file_ref().path).unwrap();
        }
        dispatch.exit();
    }

    /// Partitioned compaction never mixes partitions, and both the replacement
    /// Adds and input tombstones retain the partition values needed to reload a
    /// valid Delta snapshot.
    #[test]
    fn compaction_preserves_partitioned_table_metadata_and_rows() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");

        // Two inputs in each partition. If batching crossed the partition
        // boundary, the merged file could not truthfully carry one constant.
        let inputs = [
            ("one-a.parquet", 1),
            ("one-b.parquet", 1),
            ("two-a.parquet", 2),
            ("two-b.parquet", 2),
        ];
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file_name, value) in inputs {
            write_partitioned_parquet_file(
                encoded.path(),
                file_name,
                vec![value * 10, value * 10 + 1],
                value,
            );
            table
                .append_data_file(
                    ObjectPath::new(file_name),
                    &std::fs::read(encoded.path().join(file_name)).unwrap(),
                    Some(partition_value(value)),
                )
                .unwrap();
        }
        let before_version = table.version();
        assert_eq!(table.file_refs().unwrap().len(), 4);
        assert!(datastore.refresh_from_store().unwrap());

        let target = table
            .file_refs()
            .unwrap()
            .into_iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        // One round merges each partition's small files in turn; a second round
        // then finds nothing left to do.
        assert_eq!(
            datastore
                .table_handle(&name)
                .unwrap()
                .file_refs()
                .unwrap()
                .len(),
            2
        );
        run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        // Reload from Delta rather than trusting the in-memory swap. Exactly one
        // output per partition remains, and every original row is still present.
        let mut reloaded = datastore.table_handle(&name).unwrap();
        reloaded.refresh().unwrap();
        assert_eq!(reloaded.file_refs().unwrap().len(), 2);
        let partitions = reloaded.file_partitions().unwrap();
        assert_eq!(partitions.len(), 2);
        for expected in [partition_value(1), partition_value(2)] {
            assert!(partitions.iter().any(|(_, actual)| {
                actual
                    .as_ref()
                    .is_some_and(|actual| scalar_values_equal(actual, &expected))
            }));
        }
        let parquet = reloaded.select_scan(&[]).unwrap();
        assert_eq!(
            parquet
                .row_groups()
                .iter()
                .map(|group| group.num_rows)
                .sum::<i64>(),
            8
        );

        // Inspect every compaction commit. Each removes and adds files from one
        // partition only, and Kernel serialized that value into both actions.
        let log_dir = db.path().join(reloaded.location()).join("_delta_log");
        let mut compacted_partitions = std::collections::BTreeSet::new();
        for version in before_version + 1..=reloaded.version() {
            let commit =
                std::fs::read_to_string(log_dir.join(format!("{version:020}.json"))).unwrap();
            let actions: Vec<serde_json::Value> = commit
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let remove_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("remove"))
                .map(|remove| remove["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("remove").is_some())
                    .count(),
                2
            );
            let add_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("add"))
                .map(|add| add["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("add").is_some())
                    .count(),
                1
            );
            assert_eq!(remove_values.len(), 1, "one partition per compacted batch");
            assert_eq!(
                add_values, remove_values,
                "replacement stays in its partition"
            );
            compacted_partitions.extend(remove_values.into_iter().map(str::to_string));
        }
        assert_eq!(
            compacted_partitions,
            std::collections::BTreeSet::from(["1".to_string(), "2".to_string()])
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a table whose adopted files were listed
    /// **store-relative** (a directory under the database root, resolved through
    /// the store -- the same path a remote `s3://` root takes) compacts through
    /// the exact same code, with writes and deletes going through the table's
    /// store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();

        // Two small files under a prefix inside the database root.
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![3]);

        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        // A relative adoption path: the files are read at `events` under the
        // store root, rather than at an absolute path of their own.
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        let table_dir = db.path().join(
            datastore
                .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .location(),
        );

        let target = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .unwrap()
            .iter()
            .map(|f| f.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            true,
        );

        // One merged object replaced the two inputs, in the store and in the
        // datastore's table view.
        let files = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().starts_with("pivot-"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        // The merged object is written to the table's own directory; the two
        // adopted inputs are left where they were.
        assert!(on_disk.contains(&files[0].path.as_str().to_string()));
        assert!(adopted_dir.join("a.parquet").exists());
        assert!(adopted_dir.join("b.parquet").exists());

        let rows = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>();
        assert_eq!(rows, 3);

        dispatch.exit();
    }

    /// The selection of small-file merges of at most `merge_target_bytes`
    /// and at least `min_files` files.
    fn small_file_selection(merge_target_bytes: u64, min_files: usize) -> MergeSelection {
        MergeSelection {
            target_bytes: DEFAULT_COMPACT_BYTES,
            merge_target_bytes,
            min_files,
            layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
        }
    }

    /// When the tests' small files were committed, and the time selection
    /// is asked at unless a test says otherwise.
    const COMMITTED_AT_MS: u64 = 1_000_000_000_000;

    fn file_of_size(path: &str, size: u64) -> SmallFile {
        SmallFile {
            file: FileRef {
                path: ObjectPath::new(path),
                size,
            },
            committed_at_ms: COMMITTED_AT_MS,
        }
    }

    /// A Delta datastore holding the sorted table `events` over one file,
    /// with the directory that keeps it alive.
    fn datastore_with_one_sorted_file(
        dispatch: &Dispatch,
    ) -> (tempfile::TempDir, Arc<PivotlakeDatastore>) {
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![0, 100]);
        let datastore = PivotlakeDatastore::open_with_format(
            db.path().to_str().unwrap(),
            dispatch.dispatcher(),
            Some(crate::TableFormat::Delta),
        )
        .unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        (db, datastore)
    }

    /// A compaction actor over `datastore` that nothing drives but the
    /// test, for which every file is full enough for layout work.
    fn layout_actor(datastore: Arc<PivotlakeDatastore>) -> CompacterActor {
        let (_, actor) = CompacterHandle::new(
            2,
            u64::MAX,
            usize::MAX,
            DEFAULT_COMPACT_PARALLELISM,
            2,
            None,
            datastore,
        );
        actor
    }

    /// Run `rounds`, what a test drives a compaction actor through, to
    /// completion on a temporary runtime.
    fn block_on_rounds<Outcome>(rounds: impl Future<Output = Outcome>) -> Outcome {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(rounds)
    }

    /// One round of `actor` running `pass` over the table `name` as its
    /// datastore holds it.
    fn run_round(
        actor: &mut CompacterActor,
        name: &SchemaQualifiedTableName,
        pass: CompactionPass,
    ) {
        let id = actor.datastore.table_handle(name).unwrap().id();
        block_on_rounds(actor.compact_table(name, id, pass)).unwrap();
    }

    /// Whether `actor` remembers anything of the layout of the table `id`.
    fn remembers_a_layout_of(actor: &CompacterActor, id: Uuid) -> bool {
        actor
            .table_states
            .get(&id)
            .is_some_and(|state| !state.overlaps.is_empty())
    }

    #[test]
    fn a_final_sweep_forgets_the_partition_its_table_no_longer_has() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let (_db, datastore) = datastore_with_one_sorted_file(&dispatch);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        let mut actor = layout_actor(datastore.clone());
        run_round(&mut actor, &name, CompactionPass::Routine);
        let remembered_with_the_file = remembers_a_layout_of(&actor, table.id());
        let files: Vec<ObjectPath> = table
            .file_refs()
            .unwrap()
            .into_iter()
            .map(|file| file.path)
            .collect();
        table.replace_data_files(&files, &[]).unwrap();
        datastore.refresh_from_store().unwrap();

        run_round(&mut actor, &name, CompactionPass::Final);

        assert!(remembered_with_the_file);
        assert!(!remembers_a_layout_of(&actor, table.id()));
        dispatch.exit();
    }

    #[test]
    fn a_round_forgets_the_table_its_datastore_no_longer_holds() {
        use catalog::datastore::DatastoreTransaction as _;
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let (_db, datastore) = datastore_with_one_sorted_file(&dispatch);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let id = datastore.table_handle(&name).unwrap().id();
        let mut actor = layout_actor(datastore.clone());
        run_round(&mut actor, &name, CompactionPass::Routine);
        let remembered_before_the_drop = actor.table_states.contains_key(&id);
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_drop_table(planner::catalog::DropTableRequest {
                datastore_name: None,
                schema_name: None,
                name: "events".to_string(),
                if_exists: false,
            })
            .unwrap()
            .compile(dispatch.dispatcher())
            .unwrap()
            .collect()
            .unwrap();
        block_on_rounds(transaction.commit()).unwrap();

        block_on_rounds(actor.compact_all());

        assert!(remembered_before_the_drop);
        assert!(!actor.table_states.contains_key(&id));
        dispatch.exit();
    }

    #[test]
    fn only_files_strictly_below_half_target_are_small() {
        assert!(is_small_file(49, 100));
        assert!(!is_small_file(50, 100));
        assert!(!is_small_file(80, 100));

        assert!(is_small_file(2, 5));
        assert!(!is_small_file(3, 5));
    }

    #[test]
    fn a_crowd_of_equal_small_files_merges_the_last_of_them_a_merge_takes() {
        const MIB: u64 = 1024 * 1024;
        let files: Vec<_> = (0..120)
            .map(|index| file_of_size(&format!("small-{index:03}"), MIB))
            .collect();

        let run = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run, Some(56..120));
    }

    #[test]
    fn a_large_file_is_left_out_of_the_merge_of_its_small_neighbours() {
        const MIB: u64 = 1024 * 1024;
        let mut files = vec![file_of_size("outlier", 500 * MIB)];
        files.extend((0..9).map(|index| file_of_size(&format!("small-{index}"), MIB)));

        let run = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run, Some(1..10));
    }

    #[test]
    fn few_equal_small_files_merge_once_there_are_five() {
        const MIB: u64 = 1024 * 1024;
        let files: Vec<_> = (0..5)
            .map(|index| file_of_size(&format!("small-{index}"), MIB))
            .collect();

        let run_of_four = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files[..4], COMMITTED_AT_MS, 1)
            .into_iter()
            .next();
        let run_of_five = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run_of_four, None);
        assert_eq!(run_of_five, Some(0..5));
    }

    #[test]
    fn a_run_merges_with_the_tiny_files_at_its_end() {
        const MIB: u64 = 1024 * 1024;
        let mut files: Vec<_> = (0..4)
            .map(|index| file_of_size(&format!("small-{index}"), 10 * MIB))
            .collect();
        files.extend((0..3).map(|index| file_of_size(&format!("tiny-{index}"), MIB / 4)));

        let run = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run, Some(0..7));
    }

    #[test]
    fn a_merge_takes_no_more_bytes_than_one_merged_file() {
        const MIB: u64 = 1024 * 1024;
        let files: Vec<_> = (0..60)
            .map(|index| file_of_size(&format!("small-{index:02}"), 200 * MIB))
            .collect();

        let run = small_file_selection(1300 * MIB, 2)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run.unwrap().len(), 6);
    }

    #[test]
    fn a_merge_takes_no_fewer_files_than_configured() {
        const MIB: u64 = 1024 * 1024;
        let files: Vec<_> = (0..5)
            .map(|index| file_of_size(&format!("small-{index}"), MIB))
            .collect();

        let allowed = small_file_selection(1300 * MIB, 5)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();
        let waiting = small_file_selection(1300 * MIB, 6)
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();

        assert_eq!(allowed, Some(0..5));
        assert_eq!(waiting, None);
    }

    #[test]
    fn small_files_of_uneven_sizes_merge_once_they_have_waited() {
        const MIB: u64 = 1024 * 1024;
        const DAY_MS: u64 = 24 * 3600 * 1000;
        let files: Vec<_> = [219, 205, 263, 205]
            .into_iter()
            .enumerate()
            .map(|(index, mebibytes)| file_of_size(&format!("left-{index}"), mebibytes * MIB))
            .collect();
        let selection = small_file_selection(5120 * MIB, 2);

        let run_when_committed = selection
            .pick_small_file_runs(&files, COMMITTED_AT_MS, 1)
            .into_iter()
            .next();
        let run_a_day_later = selection
            .pick_small_file_runs(&files, COMMITTED_AT_MS + DAY_MS, 1)
            .into_iter()
            .next();

        assert_eq!(run_when_committed, None);
        assert_eq!(run_a_day_later, Some(0..4));
    }

    #[test]
    fn the_runs_of_one_commit_are_each_the_best_of_what_the_runs_before_left() {
        const KIB: u64 = 1024;
        let files: Vec<_> = (0..250)
            .map(|index| file_of_size(&format!("small-{index:03}"), 256 * KIB))
            .collect();

        let runs = small_file_selection(1300 * 1024 * KIB, 2).pick_small_file_runs(
            &files,
            COMMITTED_AT_MS,
            MAX_SMALL_FILE_MERGES_PER_COMMIT,
        );

        assert_eq!(runs, [150..250, 50..150, 0..50]);
    }

    #[test]
    fn a_commit_takes_no_more_runs_than_it_is_asked_for() {
        const KIB: u64 = 1024;
        let files: Vec<_> = (0..250)
            .map(|index| file_of_size(&format!("small-{index:03}"), 256 * KIB))
            .collect();

        let runs = small_file_selection(1300 * 1024 * KIB, 2).pick_small_file_runs(
            &files,
            COMMITTED_AT_MS,
            2,
        );

        assert_eq!(runs, [150..250, 50..150]);
    }

    #[test]
    fn a_run_that_would_take_a_commits_inputs_past_what_is_written_together_ends_the_picking() {
        const MIB: u64 = 1024 * 1024;
        let files: Vec<_> = (0..120)
            .map(|index| file_of_size(&format!("small-{index:03}"), MIB))
            .collect();

        let runs = small_file_selection(1300 * MIB, 2).pick_small_file_runs(
            &files,
            COMMITTED_AT_MS,
            MAX_SMALL_FILE_MERGES_PER_COMMIT,
        );

        assert_eq!(runs, [56..120]);
    }

    #[test]
    fn a_run_picked_around_a_large_file_leaves_the_runs_on_either_side_of_it() {
        const MIB: u64 = 1024 * 1024;
        let mut files: Vec<_> = (0..8)
            .map(|index| file_of_size(&format!("before-{index}"), MIB))
            .collect();
        files.push(file_of_size("outlier", 400 * MIB));
        files.extend((0..9).map(|index| file_of_size(&format!("after-{index}"), MIB)));

        let runs = small_file_selection(1300 * MIB, 2).pick_small_file_runs(
            &files,
            COMMITTED_AT_MS,
            MAX_SMALL_FILE_MERGES_PER_COMMIT,
        );

        assert_eq!(runs, [9..18, 0..8]);
    }
}
