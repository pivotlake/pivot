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
//! The compacter keeps, per table, the table's files as its own lists
//! ([`files::TableFileLists`]): a partition at a time, its small files and
//! its large ones each in the order a partition's files stand in. A table's
//! format hands its files out in groups it never changes, a manifest of an
//! Iceberg table or the whole of one version of a Delta table, and the
//! lists are brought to the version the datastore has published by the
//! groups that version has and the lists do not hold, and the groups the
//! lists hold and it no longer has. That happens before every selection,
//! a round's first included, so a selection sees what was committed since
//! the one before, and reads nothing of a table whose groups it already
//! holds. A merge of the round's own is applied as it commits, by its
//! inputs and its outputs alone, when the table's format says how the
//! commit's groups carry forward
//! ([`CarriedFileGroups`](crate::table_format::CarriedFileGroups)): the
//! selection that follows then finds the lists at the version the merge
//! published, and compares nothing.
//! Nothing is kept when the compacter starts, so its first round over a
//! table takes every group the table has.
//!
//! # Which partitions a round examines
//!
//! A partition a file joined or left is marked as changed until a round
//! finds nothing left to merge in it. A routine round examines the
//! partitions marked so as it begins, and no other: a partition nothing
//! changed in is not looked at. A round that fails leaves its partitions
//! marked for the next. `COMPACT ... FINAL` examines every partition, one
//! after another. No round examines a partition whose files the table's
//! format does not rewrite: an Iceberg table's partition written under a
//! partition spec other than the table's default.
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
//! selects a merge. A merge looks its inputs up in that same copy and its
//! commit publishes the next version, so a round never selects a file its
//! merge cannot find. What another process committed is in the published
//! copy once the datastore's periodic refresh has read it; a round has the
//! table refreshed only after a commit conflict. Configured timer ticks and
//! explicit `COMPACT` commands share the actor's queue, so their rewrites
//! never race in one process. The actor stops before the datastore's
//! dispatch pool shuts down.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use object_storage::{FileRef, ObjectPath};
use planner::catalog::SchemaQualifiedTableName;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::table_format::{CarriedFileGroups, CompactionFile, RemovedFiles};
use crate::{CatalogTable, Error, PivotlakeDatastore, Result};

mod files;
mod overlap;
pub(crate) mod sort_key;

/// Rows per row group in a merged file.
pub const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default compacted-file size target. Files smaller than half this size enter
/// small-file compaction; files at least half this size are full enough to
/// enter layout optimization instead.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default headroom for compression gains: a group whose input bytes only just
/// reach the desired output size can encode to another undersized file.
pub const DEFAULT_MERGE_TARGET_MULTIPLIER: f64 = 1.3;

/// Default count at which a sub-target pile may take the balance-check path.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 100;

/// Per-file fixed cost in the balance test. This prevents many similarly small
/// files from being rejected merely because their byte total is low.
const FIXED_FILE_COST_BYTES: f64 = (5 * 1024 * 1024) as f64;
const MIN_BALANCE_RATIO: f64 = 5.0;

/// Default cadence for re-checking the tables' metadata. Candidates only
/// change when a commit produces a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(30);

/// Default number of merges a datastore runs at once. A merge holds its whole
/// input decoded on the memory ring until its output is encoded, and the
/// decoded form is many times the input (a merge of 300 MB of compressed
/// OTLP logs with variant columns pinned about 6 GB). Several merges landing
/// together can take the entire ring, and then every read that needs a
/// buffer, the merges' own footer loads and the ingest included, fails with
/// nothing left to evict. One merge at a time keeps that footprint bounded;
/// a deployment that has the memory to spare raises it in its config.
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
    /// How many merges a round may have in flight at once. Selection stays
    /// serial whatever this is; only the rewrites overlap.
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
    /// What the rounds keep of each table from one to the next: its files,
    /// and what they measured of them.
    table_files: HashMap<Uuid, files::TableFileLists>,
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

    /// Forget what the rounds keep of the tables the datastore no longer
    /// holds.
    fn forget_dropped_tables(&mut self) {
        let datastore = &self.datastore;
        self.table_files
            .retain(|id, _| datastore.table_handle_by_id(id).is_some());
    }

    async fn compact_all(&mut self) {
        self.forget_dropped_tables();
        for (name, table) in self.datastore.tables() {
            match self
                .compact_table(&name, table.id(), CompactionPass::Routine)
                .await
            {
                Ok(round) => debug!(
                    table = %name,
                    partitions = round.examined_partitions,
                    passes = round.passes,
                    "compaction round examined the table"
                ),
                Err(error) => error!(table = %name, error = %error, "compaction round failed"),
            }
        }
    }

    /// Compact the table `id` until it has no candidate left: its first
    /// selection lists the partitions the round examines, then each pass
    /// starts a small-file group if one qualifies, else the best
    /// overlapping large-file group. Up to `max_concurrent_merges` of those rewrites run
    /// at once, each holding its inputs reserved until it commits, so
    /// selection works around the merges already in flight. A pass with
    /// nothing to start and nothing in flight ends the round.
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
                let inputs = batch.inputs.iter();
                let input_paths = inputs.map(|file| file.file().path.clone());
                progress.reserved.extend(input_paths);
                progress.merging.push(batch.partition.clone());
                in_flight.spawn(compact_batch(
                    Arc::clone(&self.datastore),
                    name.clone(),
                    id,
                    batch,
                    self.target_bytes,
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
                // from: the lists are brought to it by what the merge
                // changed, when its format says how its groups carry.
                Ok(None) => {}
                Ok(Some(carried_groups)) => {
                    let table_files = self.table_files.entry(id).or_default();
                    let carried = table_files.carry_merge(
                        &merge.partition,
                        &merge.inputs,
                        carried_groups,
                        self.target_bytes,
                    );
                    if let Err(file) = carried {
                        self.table_files.remove(&id);
                        drain_merges(&mut in_flight).await;
                        return Err(Error::CompactionListsOutOfStep {
                            table: name.to_string(),
                            file: file.to_string(),
                        });
                    }
                }
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
        let selection = MergeSelection {
            target_bytes: self.target_bytes,
            merge_target_bytes: self.merge_target_bytes,
            min_files: self.min_files,
            layout_clique_size: self.layout_clique_size,
        };
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
    partitions: Option<Vec<Vec<u8>>>,
    /// The partitions the round found nothing left to do in.
    finished: HashSet<Vec<u8>>,
    /// The partition of each merge in flight.
    merging: Vec<Vec<u8>>,
    /// The files the merges in flight are rewriting, which no second merge
    /// may claim.
    reserved: HashSet<ObjectPath>,
}

/// The files one merge rewrites together, all of one partition.
struct MergeBatch {
    /// The key of the partition the inputs sit in.
    partition: Vec<u8>,
    /// The inputs, each as the lists it was selected from hold it.
    inputs: Vec<Arc<dyn CompactionFile>>,
    kind: MergeKind,
    /// The version of the table the inputs were selected from.
    selected_at_version: u64,
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
    /// large-file group, grown from the most contested pair. No merge
    /// claims a file `progress` holds reserved.
    ///
    /// The pass first brings `table_files` to the version `table` is at,
    /// the copy the datastore has published, and lists the partitions the
    /// round examines when `progress` holds none
    /// ([`files::TableFileLists::list_round_partitions`]). It then looks at
    /// those partitions one at a time and never at one `progress` records
    /// as finished. A partition with nothing to start and no merge in
    /// flight is finished, and counts as unchanged until a file joins or
    /// leaves it.
    fn next_batch(
        &self,
        table_files: &mut files::TableFileLists,
        table: &CatalogTable,
        pass: CompactionPass,
        progress: &mut RoundProgress,
    ) -> Result<Option<MergeBatch>> {
        table_files.apply_file_groups(table, self.target_bytes)?;
        let RoundProgress {
            partitions,
            finished,
            merging,
            reserved,
        } = progress;
        let partitions = match partitions.take() {
            Some(listed) => partitions.insert(listed),
            None => partitions.insert(table_files.list_round_partitions(table, pass)?),
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
            let build_batch = |inputs, kind| MergeBatch {
                partition: partition.clone(),
                inputs,
                kind,
                selected_at_version: table.version(),
            };
            if let Some(inputs) = self.next_small_batch(partition_files, pass, reserved) {
                return Ok(Some(build_batch(inputs, MergeKind::SmallFiles)));
            }
            let group = partition_files.select_layout_group(
                table,
                reserved,
                pass,
                self.layout_clique_size,
            )?;
            if let Some(inputs) = group {
                return Ok(Some(build_batch(inputs, MergeKind::LayoutOptimization)));
            }
            if merging.contains(partition) {
                continue;
            }
            partition_files.settle();
            finished.insert(partition.clone());
        }
        Ok(None)
    }

    /// Pick one group of the small files of a partition, from the start of
    /// their list. A normal merge stops as soon as its input reaches the
    /// configured merge target. A pile that reaches the configured
    /// file-count threshold first uses the balance test, repeatedly
    /// discarding its largest file until the remainder is balanced enough to
    /// merge or no useful group remains.
    fn next_small_batch(
        &self,
        partition_files: &files::PartitionFileLists,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<Vec<Arc<dyn CompactionFile>>> {
        let small_files = partition_files.list_small_files(reserved);
        let batch = match pass {
            CompactionPass::Routine => small_file_batch(
                small_files,
                |file| file.file().size,
                self.merge_target_bytes,
                self.min_files,
            ),
            CompactionPass::Final => {
                let small_files: Vec<&Arc<dyn CompactionFile>> = small_files.collect();
                (small_files.len() >= 2).then_some(small_files)
            }
        };
        batch.map(|batch| batch.into_iter().cloned().collect())
    }
}

/// What one merge hands back.
struct MergeOutcome {
    /// The key of the partition the inputs sit in.
    partition: Vec<u8>,
    /// The inputs, each as the lists it was selected from held it.
    inputs: Vec<Arc<dyn CompactionFile>>,
    /// The input paths the merge held reserved.
    reserved_paths: Vec<ObjectPath>,
    /// The version of the table the inputs were selected from.
    selected_at_version: u64,
    /// What the commit of the merge's swap did to the table's groups, when
    /// the table's format says so, or why the merge did not commit.
    committed: Result<Option<CarriedFileGroups>>,
}

/// Rewrite the inputs of `batch` into target-sized outputs and swap them
/// into the table in one commit. Returns the inputs' partition and paths
/// beside the outcome so the round can release their reservations however
/// the merge went, and the version they were selected from so the round
/// can tell a conflict it cannot get past.
async fn compact_batch(
    datastore: Arc<PivotlakeDatastore>,
    name: SchemaQualifiedTableName,
    id: Uuid,
    batch: MergeBatch,
    target_bytes: u64,
) -> MergeOutcome {
    let MergeBatch {
        partition,
        inputs,
        kind,
        selected_at_version,
    } = batch;
    let input_files: Vec<FileRef> = inputs.iter().map(|file| file.file().clone()).collect();
    let reserved_paths: Vec<ObjectPath> =
        input_files.iter().map(|file| file.path.clone()).collect();
    let input_sizes: Vec<u64> = input_files.iter().map(|file| file.size).collect();
    let merged_partition = partition.clone();
    let build_outcome = |committed| MergeOutcome {
        partition,
        inputs,
        reserved_paths,
        selected_at_version,
        committed,
    };
    // `spawn_blocking` needs a `'static` closure, so the merge takes the
    // datastore handle rather than a borrow of it.
    let merged = match tokio::task::spawn_blocking(move || {
        compact_partition_files(
            datastore.as_ref(),
            id,
            &merged_partition,
            &input_files,
            target_bytes,
        )
    })
    .await
    {
        Ok(Ok(merged)) => merged,
        Ok(Err(error)) => return build_outcome(Err(error)),
        Err(error) => {
            return build_outcome(Err(Error::CompactionJobPanicked(error.to_string())));
        }
    };
    let output_sizes = merged.output_sizes;
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
    build_outcome(Ok(merged.carried_groups))
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
            // A merge that commits here is not applied to the lists: the
            // next round brings them to the table's version by its groups.
            Ok(Ok(_)) => {}
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

/// The small files one merge takes of `files`, a partition's in order, each
/// of the size `read_size` reads of it: the first of them up to the merge
/// target, else all of them once there are `min_files` and they are
/// balanced enough. The files past the merge target are not looked at.
fn small_file_batch<File>(
    files: impl IntoIterator<Item = File>,
    read_size: impl Fn(&File) -> u64,
    merge_target_bytes: u64,
    min_files: usize,
) -> Option<Vec<File>> {
    let mut total = 0u64;
    let mut batch = Vec::new();
    for file in files {
        total += read_size(&file);
        batch.push(file);
        if batch.len() >= 2 && total >= merge_target_bytes {
            return Some(batch);
        }
    }

    if batch.len() < min_files.max(2) {
        return None;
    }

    // A few disproportionately large files can make the whole pile a poor
    // merge. Remove the current largest and run the same balance test again on
    // everything that remains; the count threshold is only the entry condition.
    while batch.len() >= 2 {
        let largest = batch
            .iter()
            .enumerate()
            .max_by_key(|(_, file)| read_size(file))
            .map(|(index, _)| index)
            .expect("a non-empty batch has a largest file");
        let largest_size = read_size(&batch[largest]);
        let ratio = (total as f64 + FIXED_FILE_COST_BYTES * batch.len() as f64)
            / (largest_size as f64 + FIXED_FILE_COST_BYTES);
        if ratio >= MIN_BALANCE_RATIO {
            return Some(batch);
        }
        total -= largest_size;
        batch.swap_remove(largest);
    }
    None
}

/// A merge that committed its swap.
struct CommittedMerge {
    /// The size of each file the merge wrote in place of its inputs.
    output_sizes: Vec<u64>,
    /// What the commit did to the groups of the version it was made on,
    /// when the table's format says so.
    carried_groups: Option<CarriedFileGroups>,
}

/// Merge `inputs`, files of the partition with the key `partition` of the
/// table with identity `id`, into fresh target-sized files and swap them in for
/// the inputs, in one commit labelled a rearrangement, which publishes the
/// version it wrote. Returns the sizes of the merged files, with what the
/// commit did to the table's groups.
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
    max_output_file_size: u64,
) -> Result<CommittedMerge> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
    let merged = table.merge_files(partition, inputs, max_output_file_size)?;

    let outputs: Vec<FileRef> = merged.iter().map(|file| file.file_ref().clone()).collect();
    let removed_paths: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    let removed = RemovedFiles {
        partition,
        paths: &removed_paths,
    };
    // The swap goes through the datastore's one in-process write path, under
    // the table's commit lock.
    let carried_groups = match datastore.commit_to_table(id, removed, merged, false) {
        Ok(carried_groups) => carried_groups,
        Err(error) => {
            if error.may_have_committed() {
                return Err(error);
            }
            for file in &outputs {
                if let Err(cleanup_error) = table.delete_data_file(&file.path) {
                    warn!(
                        commit_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %file.path,
                        "compaction: deleting output after commit failure failed (orphan left)"
                    );
                }
            }
            return Err(error);
        }
    };
    Ok(CommittedMerge {
        output_sizes: outputs.iter().map(|file| file.size).collect(),
        carried_groups,
    })
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Payload0", DataType::Int64, false),
            Field::new("Payload1", DataType::Int64, false),
            Field::new("Payload2", DataType::Int64, false),
            Field::new("Payload3", DataType::Int64, false),
        ]));
        let keys: Vec<i64> = (1..=rows as i64).collect();
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
        let options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        let sorted_by = if sorted {
            vec!["Timestamp".to_string()]
        } else {
            Vec::new()
        };
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            partitioned_by: Vec::new(),
            sorted_by,
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
            partitioned_by: Vec::new(),
            sorted_by: vec!["Timestamp".to_string()],
            options: std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
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
            partitioned_by: vec!["Partition".to_string()],
            sorted_by: Vec::new(),
            options: std::collections::HashMap::new(),
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

    /// Refresh `name` to the latest committed version (a writer evolves a
    /// cloned-out handle, so the datastore's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(datastore: &PivotlakeDatastore, name: &str) -> Arc<ParquetTable> {
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists");
        table.refresh().expect("refresh the table");
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
        let target = (total as f64 / DEFAULT_MERGE_TARGET_MULTIPLIER).floor() as u64;
        run_one_sweep(target, datastore.clone());

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
        let partition = find_only_partition(&table);
        let unbounded = table.merge_files(&partition, &inputs, u64::MAX).unwrap();
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

        let partition = find_only_partition(&table);

        let outputs = table.merge_files(&partition, &inputs, target).unwrap();

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
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            false,
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
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            false,
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

    fn file_of_size(path: &str, size: u64) -> FileRef {
        FileRef {
            path: ObjectPath::new(path),
            size,
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

    /// The key of the one partition `table` has a file in.
    fn find_only_partition(table: &CatalogTable) -> Vec<u8> {
        let groups = table.load_file_groups(&table.list_file_groups()).unwrap();
        let mut partitions = groups.into_iter().flatten();
        partitions.next().unwrap().partition
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

    /// Whether `actor` keeps a file of the partition `partition` of the
    /// table `id`.
    fn keeps_a_file_of(actor: &mut CompacterActor, id: Uuid, partition: &[u8]) -> bool {
        let table_files = actor.table_files.get_mut(&id);
        table_files.is_some_and(|files| files.find_partition(partition).is_some())
    }

    #[test]
    fn a_round_forgets_the_partition_its_table_no_longer_has_a_file_in() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let (_db, datastore) = datastore_with_one_sorted_file(&dispatch);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        let partition = find_only_partition(&table);
        let mut actor = layout_actor(datastore.clone());
        run_round(&mut actor, &name, CompactionPass::Routine);
        let kept_with_the_file = keeps_a_file_of(&mut actor, table.id(), &partition);
        let files: Vec<ObjectPath> = table
            .file_refs()
            .unwrap()
            .into_iter()
            .map(|file| file.path)
            .collect();
        table.replace_data_files(&files, &[]).unwrap();
        datastore.refresh_from_store().unwrap();

        run_round(&mut actor, &name, CompactionPass::Routine);

        assert!(kept_with_the_file);
        assert!(!keeps_a_file_of(&mut actor, table.id(), &partition));
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
        let remembered_before_the_drop = actor.table_files.contains_key(&id);
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
        assert!(!actor.table_files.contains_key(&id));
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
    fn small_files_merge_at_configured_target() {
        let files = [
            file_of_size("a", 70),
            file_of_size("b", 59),
            file_of_size("c", 1),
        ];

        let batch = small_file_batch(files.to_vec(), |file| file.size, 130, 100).unwrap();

        assert_eq!(
            batch.iter().map(|file| file.size).collect::<Vec<_>>(),
            [70, 59, 1]
        );
    }

    #[test]
    fn small_file_fallback_removes_largest_until_balanced() {
        const MIB: u64 = 1024 * 1024;
        let mut files = vec![file_of_size("outlier", 500 * MIB)];
        files.extend((0..99).map(|index| file_of_size(&format!("small-{index}"), MIB)));

        let batch =
            small_file_batch(files.to_vec(), |file| file.size, 2 * 1024 * MIB, 100).unwrap();

        assert_eq!(batch.len(), 99, "the check continues below 100 files");
        assert!(batch.iter().all(|file| file.path.as_str() != "outlier"));
    }

    #[test]
    fn small_file_fallback_uses_configured_minimum() {
        let files: Vec<_> = (0..5)
            .map(|index| file_of_size(&format!("small-{index}"), 1))
            .collect();

        let allowed = small_file_batch(files.to_vec(), |file| file.size, 100, 5);
        let waiting = small_file_batch(files.to_vec(), |file| file.size, 100, 6);

        assert_eq!(allowed.unwrap().len(), 5);
        assert!(waiting.is_none());
    }
}
