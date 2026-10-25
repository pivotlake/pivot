//! Background compaction of a table's Parquet files, for whichever datastore
//! and table format holds the table.
//!
//! Frequent writes litter a table with small files, and scans pay per file.
//! The [`CompacterHandle`] runs two strategies, split by file size: files
//! below half the configured target are merged as *small files*, files at
//! least half full are candidates for *layout optimization*. Both rewrite
//! their inputs into target-sized outputs and swap them into the table in a
//! single commit. A row group larger than the target stays one oversized
//! file, since row groups are the smallest independently encoded unit. Each
//! pass tries small files first, then layout.
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
//! [`CONTESTED_ROW_GROUP_PERCENT_BAR`] percent of each file's row
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
//! A datastore runs one [`CompacterHandle`] actor over its
//! [`CompactionHost`]: what names its tables, finds one by identity and
//! commits a swap. Each table reaches the compacter as a
//! [`CompactionTable`]: a copy at one committed version the round refreshes,
//! selects from, merges through and folds its own swaps into. Configured
//! timer ticks and explicit `COMPACT` commands share the actor's queue, so
//! their rewrites never race in one process. A timed round reloads each
//! table to its latest version; commands carry the table copy captured by
//! their query transaction. The actor stops before the datastore's dispatch
//! pool shuts down.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use arrow_schema::SchemaRef;
use dispatch::{DataFlowDispatcher, DataFlowError};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::RowGroupMetadata;
use parquet_engine::file_entry::{FileEntry, PartitionValues, scalar_values_equal};
use planner::catalog::{Column, SchemaQualifiedTableName};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};
use uuid::Uuid;

mod merge;
mod overlap;

pub use merge::merge_files;
pub use overlap::CONTESTED_ROW_GROUP_PERCENT_BAR;

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
/// change when a commit lands a new version, so seconds-scale is plenty.
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

/// What can go wrong inside the compacter itself. A host's own failures
/// travel as the host's error type, which carries these through
/// [`CompactionError`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("table at `{location}` commit conflict: input file `{file}` is no longer active")]
    CommitConflict { location: String, file: String },
    #[error("running a compaction merge: {0}")]
    Merge(#[source] DataFlowError),
    #[error("table `{0}` does not exist")]
    TableNotFound(String),
    #[error("compaction actor stopped")]
    CompacterStopped,
    #[error("compaction job panicked: {0}")]
    JobPanicked(String),
}

/// The error type of a datastore the compacter serves: it carries the
/// compacter's own errors, and it tells a lost swap apart from every other
/// failure, since a round reloads and carries on after one but gives up
/// after the other.
pub trait CompactionError: std::error::Error + From<Error> + Send + Sync + 'static {
    /// Whether this error says another writer swapped one of a merge's
    /// inputs out first.
    fn is_commit_conflict(&self) -> bool;
}

/// One data file of a table, materialized: its metadata entry ([`FileEntry`]:
/// identity, partition tuple, stats) paired with the row groups read from its
/// footer (file-local order). What a writer hands a commit: the entry is the
/// metadata the commit records, the row groups the content the next scan
/// reads.
#[derive(Clone)]
pub struct TableFile {
    pub entry: FileEntry,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's entry with the row groups read from its footer.
    ///
    /// Derive the file's stats from its footers when the entry carries none, so
    /// file-level range pruning always has min/max bounds. Usually the entry
    /// already has them (a fresh INSERT records them, and a reload reads the
    /// persisted stats), so this covers only a file with neither: one adopted
    /// at CREATE, or whose metadata recorded no stats.
    pub fn new(mut entry: FileEntry, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        if entry.stats.is_none() && !row_groups.is_empty() {
            entry.stats = Some(Arc::new(parquet_engine::aggregate_file_stats(&row_groups)));
        }
        Self { entry, row_groups }
    }

    /// The file's store identity (path and size), for a caller that names the
    /// file outside the table's own module (reporting a merge's outputs,
    /// deleting one whose commit failed).
    pub fn file_ref(&self) -> &FileRef {
        &self.entry.file
    }
}

/// One table at one committed version, as the compacter holds it: the copy a
/// round refreshes, selects candidates from, merges through, and folds its
/// own committed swaps into. A copy is a read view, never a commit base; the
/// swap itself goes through the [`CompactionHost`].
pub trait CompactionTable: Clone + Send + Sync + 'static {
    type Error: CompactionError;

    /// The table's durable identity, stable across versions and renames.
    fn id(&self) -> Uuid;
    fn columns(&self) -> &[Column];
    /// The partition columns, in order (empty = unpartitioned).
    fn partition_by(&self) -> &[String];
    /// The sort columns, in order (empty = unsorted).
    fn sort_by(&self) -> &[String];
    /// Every live data file at this version, with its partition tuple and
    /// stats as the metadata recorded them.
    fn entries(&self) -> &[FileEntry];
    /// The row groups of each of `entries` (entries of this very version), in
    /// the order given: what a merge and a layout measurement read.
    fn row_groups(
        &self,
        entries: &[&FileEntry],
    ) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>, Self::Error>;
    /// Advance this copy to the latest committed version. Returns whether it
    /// advanced; `Ok(false)` means this copy was already current.
    fn refresh(&mut self) -> Result<bool, Self::Error>;
    /// Fold a swap this process already committed through the host into this
    /// read copy without a metadata read: drop `removed` and adopt `added`,
    /// whose footers the merge holds.
    fn apply_file_swap(&mut self, removed: &[ObjectPath], added: Vec<TableFile>);
    /// The pool a merge decodes and encodes on.
    fn dispatcher(&self) -> &DataFlowDispatcher;
    /// The store a merge reads its inputs from and writes its outputs to.
    fn store(&self) -> Arc<dyn ObjectStore>;
    /// The table's directory under the store: what a merge's outputs are
    /// written under, and what their paths are reported relative to.
    fn directory(&self) -> ObjectPath;
    /// Where below the directory the table keeps its data files: empty for a
    /// table whose files sit flat in it.
    fn file_prefix(&self) -> ObjectPath;
    /// The Arrow schema merged files are encoded under, when the table names
    /// its columns differently from how its files carry them (a format that
    /// resolves columns by field id); `None` keeps the inputs' own schema.
    fn written_schema(&self) -> Option<SchemaRef>;
    /// Delete a data file under the table's directory that its metadata does
    /// not list: an output of a merge that will not commit.
    fn delete_data_file(&self, path: &ObjectPath) -> Result<(), Self::Error>;
    /// Where the table lives, for messages.
    fn location(&self) -> String;
}

/// The datastore a compacter serves: what names its tables, finds one by
/// identity, and lands a swap as one rearrangement commit on the live table.
pub trait CompactionHost: Send + Sync + 'static {
    type Table: CompactionTable<Error = Self::Error>;
    type Error: CompactionError;

    /// A copy of every table the datastore currently holds, for a sweep that
    /// refreshes and evolves each one independently.
    fn tables(&self) -> Vec<(SchemaQualifiedTableName, Self::Table)>;
    /// A copy of the table with identity `id`, or `None` if it is gone.
    fn table_by_id(&self, id: &Uuid) -> Option<Self::Table>;
    /// In one commit on the live table `id`, remove every path in `removed`
    /// and add every file in `added`, labelled a rearrangement: the rows are
    /// as they were. A commit that finds a removed path no longer live fails
    /// with an error whose [`is_commit_conflict`](CompactionError::is_commit_conflict)
    /// is true.
    fn commit_swap(
        &self,
        id: Uuid,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Result<(), Self::Error>;
}

/// Handle to a datastore's single compaction actor. Both periodic maintenance
/// and explicit `COMPACT` statements run through that actor, so two compaction
/// rounds in one process never overlap. Within one round, selection is still
/// serial while the rewrites it starts run concurrently.
pub struct CompacterHandle<H: CompactionHost> {
    commands: mpsc::UnboundedSender<CompactionCommand<H>>,
}

impl<H: CompactionHost> Clone for CompacterHandle<H> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
        }
    }
}

/// Which pass a round runs: the routine one, which keeps every guard, or
/// the final sweep, which drops them and settles the table.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum CompactionPass {
    Routine,
    Final,
}

struct CompactionCommand<H: CompactionHost> {
    name: SchemaQualifiedTableName,
    table: H::Table,
    pass: CompactionPass,
    result: oneshot::Sender<Result<u64, H::Error>>,
}

pub struct CompacterActor<H: CompactionHost> {
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
    host: Arc<H>,
    commands: mpsc::UnboundedReceiver<CompactionCommand<H>>,
    /// Each table's pair overlaps, remembered from round to round so a
    /// pass only measures the files it has not seen before (see
    /// [`overlap::OverlapCache`]).
    overlap_caches: HashMap<Uuid, overlap::OverlapCache>,
}

impl<H: CompactionHost> CompacterHandle<H> {
    /// The handle and the actor it drives, over `host`. The caller spawns
    /// [`run`](CompacterActor::run) on its runtime.
    pub fn new(
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        poll_interval: Option<Duration>,
        host: Arc<H>,
    ) -> (Self, CompacterActor<H>) {
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
                host,
                commands: receiver,
                overlap_caches: HashMap::new(),
            },
        )
    }

    /// Enqueue one table from the caller's transaction snapshot; the round
    /// refreshes it to the latest version as it begins and keeps merging until
    /// nothing qualifies. `FINAL` drops the small-file guards and the
    /// contested bar and compacts each partition to one contiguous flow, see
    /// the crate docs.
    pub async fn compact(
        &self,
        name: SchemaQualifiedTableName,
        table: H::Table,
        final_sweep: bool,
    ) -> Result<u64, H::Error> {
        let (result, answer) = oneshot::channel();
        let pass = if final_sweep {
            CompactionPass::Final
        } else {
            CompactionPass::Routine
        };
        self.commands
            .send(CompactionCommand {
                name,
                table,
                pass,
                result,
            })
            .map_err(|_| Error::CompacterStopped)?;
        answer.await.map_err(|_| Error::CompacterStopped)?
    }
}

impl<H: CompactionHost> CompacterActor<H> {
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
                        let res = self
                            .compact_table(&command.name, command.table, command.pass)
                            .await;
                        let _ = command.result.send(res.map(|(_, count)| count));
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

    async fn compact_all(&mut self) {
        let tables = self.host.tables();
        let live: HashSet<Uuid> = tables.iter().map(|(_, table)| table.id()).collect();
        self.overlap_caches.retain(|id, _| live.contains(id));
        for (name, table) in tables {
            if let Err(error) = self
                .compact_table(&name, table, CompactionPass::Routine)
                .await
            {
                error!(table = %name, error = %error, "compaction round failed");
            }
        }
    }

    /// Compact `table` until it has no candidate left: the round refreshes the
    /// table to the latest committed version once as it begins, then each pass
    /// starts a small-file group if one qualifies, else the best overlapping
    /// large-file group. Up to `max_concurrent_merges` of those rewrites run at
    /// once, each holding its inputs reserved until it commits, so selection
    /// works around the merges already in flight. A pass with nothing to start
    /// and nothing in flight ends the round. Returns the table as it stands and
    /// the number of passes.
    ///
    /// The round works off its one refreshed copy rather than reloading the
    /// metadata per pass: each of its own merges is folded in as it commits,
    /// so files landing meanwhile (an INSERT in this process, or another
    /// writer) wait for the next round. Only a commit conflict, another writer
    /// having swapped a selected file out first, reloads the copy, and the
    /// round carries on from what it finds.
    async fn compact_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        mut table: H::Table,
        pass: CompactionPass,
    ) -> Result<(H::Table, u64), H::Error> {
        let mut passes = 0;
        let mut in_flight = JoinSet::new();
        let mut reserved = HashSet::new();
        table.refresh()?;
        loop {
            passes += 1;
            let id = table.id();

            while in_flight.len() < self.max_concurrent_merges {
                let Some((inputs, kind)) = self.next_batch(&table, pass, &reserved)? else {
                    break;
                };
                reserved.extend(inputs.iter().map(|file| file.path.clone()));
                in_flight.spawn(compact_batch(
                    Arc::clone(&self.host),
                    name.clone(),
                    id,
                    inputs,
                    kind,
                    self.target_bytes,
                ));
            }

            if in_flight.is_empty() {
                return Ok((table, passes));
            }

            // Every exit from here on drains first: a merge body runs on the
            // blocking pool, which a dropped handle does not stop.
            let (reserved_paths, outcome) = match in_flight.join_next().await {
                Some(Ok(merge)) => merge,
                Some(Err(error)) => {
                    drain_merges(&mut in_flight).await;
                    return Err(Error::JobPanicked(error.to_string()).into());
                }
                None => unreachable!("the set is not empty"),
            };
            for path in &reserved_paths {
                reserved.remove(path);
            }
            match outcome {
                Ok(outputs) => table.apply_file_swap(&reserved_paths, outputs),
                Err(error) if error.is_commit_conflict() => {
                    // The merge's outputs are already deleted; reload the copy
                    // to see what the conflicting writer left and keep going.
                    if let Err(error) = table.refresh() {
                        drain_merges(&mut in_flight).await;
                        return Err(error);
                    }
                }
                Err(error) => {
                    drain_merges(&mut in_flight).await;
                    return Err(error);
                }
            }
        }
    }

    /// The next merge a pass can start: a small-file group if one qualifies,
    /// else the best overlapping large-file group grown from the most
    /// contested pair. `reserved` holds the files the merges already in
    /// flight are rewriting, which no second merge may claim.
    fn next_batch(
        &mut self,
        table: &H::Table,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Result<Option<(Vec<FileRef>, MergeKind)>, H::Error> {
        if let Some(inputs) = self.next_small_batch(table, pass, reserved) {
            return Ok(Some((inputs, MergeKind::SmallFiles)));
        }
        Ok(self
            .next_layout_optimization(table, reserved, pass)?
            .map(|inputs| (inputs, MergeKind::LayoutOptimization)))
    }

    /// Pick one group of files below the target size from a single partition.
    /// A normal merge stops as soon as its input reaches the configured merge
    /// target. A pile that reaches the configured file-count threshold first
    /// uses the balance test, repeatedly discarding its largest file until the
    /// remainder is balanced enough to merge or no useful group remains.
    fn next_small_batch(
        &self,
        table: &H::Table,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<Vec<FileRef>> {
        let mut by_partition: Vec<(Option<PartitionValues>, Vec<&FileEntry>)> = Vec::new();
        for entry in table.entries().iter() {
            if !is_small_file(entry.file.size, self.target_bytes)
                || reserved.contains(&entry.file.path)
            {
                continue;
            }
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &entry.partition))
            {
                Some((_, files)) => files.push(entry),
                None => by_partition.push((entry.partition.clone(), vec![entry])),
            }
        }

        by_partition.into_iter().find_map(|(_, files)| match pass {
            CompactionPass::Routine => {
                small_file_batch(files, self.merge_target_bytes, self.min_files)
            }
            CompactionPass::Final => (files.len() >= 2)
                .then(|| files.into_iter().map(|entry| entry.file.clone()).collect()),
        })
    }

    /// The next group of at-least-half-full files one merge should rewrite
    /// together, chosen within one partition by the table's
    /// [`overlap::OverlapCache`]: a full clique in a routine round, any
    /// connected group in a final sweep. Reserved files stay in view so
    /// their remembered overlaps survive a failed merge; selection passes
    /// their pairs over.
    fn next_layout_optimization(
        &mut self,
        table: &H::Table,
        reserved: &HashSet<ObjectPath>,
        pass: CompactionPass,
    ) -> Result<Option<Vec<FileRef>>, H::Error> {
        if table.sort_by().is_empty() {
            return Ok(None);
        }

        // A candidate is measured on its row groups' bounds, which the table
        // reads for the large files (in memory for a format that holds every
        // footer, once per version for one that reads them on demand).
        let large: Vec<&FileEntry> = table
            .entries()
            .iter()
            .filter(|entry| !is_small_file(entry.file.size, self.target_bytes))
            .collect();
        let footers = table.row_groups(&large)?;
        let mut by_partition: Vec<(Option<PartitionValues>, Vec<overlap::LayoutCandidate<'_>>)> =
            Vec::new();
        for (entry, row_groups) in large.iter().zip(&footers) {
            let candidate =
                overlap::LayoutCandidate::from_entry(entry, row_groups, table.sort_by());
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &entry.partition))
            {
                Some((_, files)) => files.push(candidate),
                None => by_partition.push((entry.partition.clone(), vec![candidate])),
            }
        }

        let partitions: Vec<&[overlap::LayoutCandidate<'_>]> = by_partition
            .iter()
            .map(|(_, files)| files.as_slice())
            .collect();
        let cache = self.overlap_caches.entry(table.id()).or_default();
        cache.observe(&partitions, table.sort_by());
        let Some(group) = cache.select_group(reserved, pass, self.layout_clique_size) else {
            return Ok(None);
        };
        let files_by_path: HashMap<&str, &FileRef> = by_partition
            .iter()
            .flat_map(|(_, candidates)| candidates.iter())
            .map(|candidate| (candidate.entry.file.path.as_str(), &candidate.entry.file))
            .collect();
        Ok(Some(
            group
                .iter()
                .map(|path| files_by_path[path.as_str()].clone())
                .collect(),
        ))
    }
}

/// What one merge hands back: the input paths it held reserved, beside the
/// output files it committed or why it did not.
type MergeOutcome<E> = (Vec<ObjectPath>, Result<Vec<TableFile>, E>);

/// Rewrite `inputs` into target-sized outputs and swap them into the table in
/// one commit. Returns the input paths beside the outcome so the round can
/// release their reservations however the merge went; a committed merge's
/// outcome carries the output files for the round to fold into its table copy.
async fn compact_batch<H: CompactionHost>(
    host: Arc<H>,
    name: SchemaQualifiedTableName,
    id: Uuid,
    inputs: Vec<FileRef>,
    kind: MergeKind,
    target_bytes: u64,
) -> MergeOutcome<H::Error> {
    let reserved_paths: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    let input_sizes: Vec<u64> = inputs.iter().map(|file| file.size).collect();
    // `spawn_blocking` needs a `'static` closure, so the merge takes the
    // host handle rather than a borrow of it.
    let merged = match tokio::task::spawn_blocking(move || {
        compact_table_files(host.as_ref(), id, &inputs, ROW_GROUP_ROWS, target_bytes)
    })
    .await
    {
        Ok(Ok(merged)) => merged,
        Ok(Err(error)) => return (reserved_paths, Err(error)),
        Err(error) => {
            return (
                reserved_paths,
                Err(Error::JobPanicked(error.to_string()).into()),
            );
        }
    };
    let output_sizes: Vec<u64> = merged.iter().map(|file| file.file_ref().size).collect();
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
    (reserved_paths, Ok(merged))
}

/// Await the merges still in flight, reporting rather than propagating what
/// they return: a merge body runs on the blocking pool, where dropping its
/// handle does not stop it, so a round that is giving up still waits for its
/// own rewrites to finish before the next one selects.
async fn drain_merges<E: std::fmt::Display + 'static>(in_flight: &mut JoinSet<MergeOutcome<E>>) {
    while let Some(joined) = in_flight.join_next().await {
        match joined {
            Ok((_, Err(error))) => warn!(error = %error, "compaction merge failed"),
            Err(error) => warn!(error = %error, "compaction merge panicked"),
            Ok((_, Ok(_))) => {}
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

fn small_file_batch(
    files: Vec<&FileEntry>,
    merge_target_bytes: u64,
    min_files: usize,
) -> Option<Vec<FileRef>> {
    let mut total = 0u64;
    let mut batch = Vec::with_capacity(files.len());
    for file in files {
        total += file.file.size;
        batch.push(file);
        if batch.len() >= 2 && total >= merge_target_bytes {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
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
            .max_by_key(|(_, entry)| entry.file.size)
            .map(|(index, _)| index)
            .expect("a non-empty batch has a largest file");
        let largest_size = batch[largest].file.size;
        let ratio = (total as f64 + FIXED_FILE_COST_BYTES * batch.len() as f64)
            / (largest_size as f64 + FIXED_FILE_COST_BYTES);
        if ratio >= MIN_BALANCE_RATIO {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
        }
        total -= largest_size;
        batch.swap_remove(largest);
    }
    None
}

/// Merge `inputs` of the table with identity `id` into fresh target-sized files
/// and swap them in for the inputs, in one commit labelled a rearrangement.
/// `inputs` must share one partition tuple; the caller batches them so. Returns
/// the merged files with their footers in hand, so a caller holding its own
/// table copy can fold the swap in without reloading the metadata.
///
/// The merge itself (decode, re-encode, upload) runs off a plain read copy of
/// the table with no commit lock held, so it never blocks a concurrent INSERT's
/// commit; only the swap that follows takes it. If that swap fails for any
/// reason, including another writer having swapped the inputs out first, the
/// uncommitted merge outputs are deleted before the error is returned.
///
/// `max_output_file_size` is a compressed row-group body target. Parquet
/// headers and footers are excluded, and an individually oversized row group
/// is emitted alone.
///
/// Blocking store I/O, so callers run it on the blocking pool.
pub fn compact_table_files<H: CompactionHost>(
    host: &H,
    id: Uuid,
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>, H::Error> {
    let table = host
        .table_by_id(&id)
        .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
    let merged = merge_files(&table, inputs, target_rows_per_group, max_output_file_size)?;

    let outputs = merged.clone();
    let removed: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    if let Err(error) = host.commit_swap(id, &removed, merged) {
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

fn partition_values_equal(left: &Option<PartitionValues>, right: &Option<PartitionValues>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => scalar_values_equal(left, right),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_entry(path: &str, size: u64) -> FileEntry {
        FileEntry::new(FileRef {
            path: ObjectPath::new(path),
            size,
        })
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
        let files = [file_entry("a", 70), file_entry("b", 59), file_entry("c", 1)];

        let batch = small_file_batch(files.iter().collect(), 130, 100).unwrap();

        assert_eq!(
            batch.iter().map(|file| file.size).collect::<Vec<_>>(),
            [70, 59, 1]
        );
    }

    #[test]
    fn small_file_fallback_removes_largest_until_balanced() {
        const MIB: u64 = 1024 * 1024;
        let mut files = vec![file_entry("outlier", 500 * MIB)];
        files.extend((0..99).map(|index| file_entry(&format!("small-{index}"), MIB)));

        let batch = small_file_batch(files.iter().collect(), 2 * 1024 * MIB, 100).unwrap();

        assert_eq!(batch.len(), 99, "the check continues below 100 files");
        assert!(batch.iter().all(|file| file.path.as_str() != "outlier"));
    }

    #[test]
    fn small_file_fallback_uses_configured_minimum() {
        let files: Vec<_> = (0..5)
            .map(|index| file_entry(&format!("small-{index}"), 1))
            .collect();

        let allowed = small_file_batch(files.iter().collect(), 100, 5);
        let waiting = small_file_batch(files.iter().collect(), 100, 6);

        assert_eq!(allowed.unwrap().len(), 5);
        assert!(waiting.is_none());
    }
}
