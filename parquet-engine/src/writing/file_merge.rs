//! Two-level NUMA-aware ordering for Parquet file candidates.
//!
//! A local planner and executor merge all runs belonging to one node. The
//! global planner then merges the single result from each participating node.
//! Both planners use [`dispatch::KWayMergePlan`], so an empty or one-run level is
//! an identity operation with no Arrow allocation.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use dispatch::memory::SlabAllocator;
use dispatch::{
    KWayMergePlan, LocatedBatch, MergeRun, MergedMapping, Sender, Unary, UnaryResult, key_order_by,
};

use super::types::{
    FileMergeContext, FileOrderInput, FileRows, GlobalMergeJob, LocalMergeJob, LocalMergeResult,
    ReadyFile,
};

#[derive(Default)]
pub(super) struct LocalMergePlanner;

impl Unary<FileOrderInput, LocalMergeJob> for LocalMergePlanner {
    fn consume(
        &mut self,
        input: FileOrderInput,
        sender: &mut dyn Sender<LocalMergeJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let request = match input {
            FileOrderInput::Ready(file) => {
                sender.send(LocalMergeJob::Ready(file))?;
                return Ok(());
            }
            FileOrderInput::Merge(request) => request,
        };
        let plan = KWayMergePlan::try_new(
            request.context.order_by.clone(),
            request.runs,
            request.context.local_outputs.len(),
        )?;
        match plan {
            KWayMergePlan::Empty => sender.send(LocalMergeJob::Identity {
                context: request.context,
                node_id: request.node_id,
                output: MergedMapping::identity(&[], Vec::new())?,
            })?,
            KWayMergePlan::Identity(output) => {
                let batches = output
                    .into_batches()
                    .into_iter()
                    .map(LocatedBatch::into_batch)
                    .collect();
                sender.send(LocalMergeJob::Identity {
                    context: request.context.clone(),
                    node_id: request.node_id,
                    output: MergedMapping::identity(&request.context.order_by, batches)?,
                })?
            }
            KWayMergePlan::Parallel(tasks) => {
                for task in tasks {
                    debug_assert_eq!(dispatch::NodeIdOutput::node_id(&task), request.node_id);
                    sender.send(LocalMergeJob::Task {
                        context: request.context.clone(),
                        node_id: request.node_id,
                        task,
                    })?;
                }
            }
        }
        Ok(())
    }
}

/// Up to this many decoded bytes, a file's rows are gathered into sorted
/// batches by the merge tasks themselves, as the slices come, and the row
/// groups are cut from those: the copy is small and the gather runs on the
/// merge stage's spare cores. A larger file, a compaction merging a batch of
/// files, is only ordered by the merge, and each column job gathers its own
/// row group's rows when it is encoded, so the file is never held sorted a
/// second time.
const GATHER_WHOLE_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Default)]
pub(super) struct LocalMergeExecutor {
    allocator: Option<SlabAllocator>,
}

impl Unary<LocalMergeJob, LocalMergeResult> for LocalMergeExecutor {
    fn consume(
        &mut self,
        job: LocalMergeJob,
        sender: &mut dyn Sender<LocalMergeResult>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        match job {
            LocalMergeJob::Ready(file) => sender.send(LocalMergeResult::Ready(file))?,
            LocalMergeJob::Identity {
                context,
                node_id,
                output,
            } => sender.send(LocalMergeResult::Merged {
                context,
                node_id,
                output,
            })?,
            LocalMergeJob::Task {
                context,
                node_id,
                task,
            } => {
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                let output = if context.in_memory_bytes <= GATHER_WHOLE_BYTES {
                    match task.execute(allocator)? {
                        Some(sorted) => {
                            let batches = sorted
                                .into_batches()
                                .into_iter()
                                .map(LocatedBatch::into_batch)
                                .collect();
                            Some(MergedMapping::identity(&context.order_by, batches)?)
                        }
                        None => None,
                    }
                } else {
                    task.execute_mapping(allocator)?
                };
                if let Some(output) = output {
                    sender.send(LocalMergeResult::Merged {
                        context,
                        node_id,
                        output,
                    })?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct GlobalMergePlanner;

impl Unary<LocalMergeResult, GlobalMergeJob> for GlobalMergePlanner {
    fn consume(
        &mut self,
        result: LocalMergeResult,
        sender: &mut dyn Sender<GlobalMergeJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let (context, node_id, output) = match result {
            LocalMergeResult::Ready(file) => {
                sender.send(GlobalMergeJob::Ready(file))?;
                return Ok(());
            }
            LocalMergeResult::Merged {
                context,
                node_id,
                output,
            } => (context, node_id, output),
        };
        context.local_outputs[node_id]
            .set(output)
            .unwrap_or_else(|_| panic!("file node {node_id} completed its local merge twice"));
        if context.nodes_remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return Ok(());
        }

        let participating: Vec<usize> = (0..context.local_outputs.len())
            .filter(|&node_id| context.local_outputs[node_id].get().is_some())
            .collect();
        if let [node_id] = participating[..] {
            sender.send(GlobalMergeJob::Ready(ready_file_of_one_node(
                &context, node_id,
            )))?;
            return Ok(());
        }
        // The nodes' outputs are ordered by their keys alone; the rows stay
        // where each node decoded them until a row group gathers them.
        let key_order_by = key_order_by(&context.order_by);
        let node_runs = context
            .local_outputs
            .iter()
            .enumerate()
            .filter_map(|(node_id, output)| {
                output
                    .get()
                    .map(|output| MergeRun::new(output.keys.clone(), node_id))
            })
            .collect();
        let plan =
            KWayMergePlan::try_new(key_order_by.clone(), node_runs, context.local_outputs.len())?;
        match plan {
            KWayMergePlan::Empty => unreachable!("a completed file has at least one row"),
            KWayMergePlan::Identity(output) => {
                let node_id = output
                    .batches()
                    .first()
                    .expect("a nonempty identity has a batch")
                    .node_id();
                let keys = output
                    .into_batches()
                    .into_iter()
                    .map(LocatedBatch::into_batch)
                    .collect();
                sender.send(GlobalMergeJob::Identity {
                    context,
                    output: MergedMapping::identity(&key_order_by, keys)?,
                    node_id,
                })?;
            }
            KWayMergePlan::Parallel(tasks) => {
                for task in tasks {
                    sender.send(GlobalMergeJob::Task {
                        context: context.clone(),
                        task,
                    })?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct GlobalMergeExecutor {
    allocator: Option<SlabAllocator>,
}

impl Unary<GlobalMergeJob, ReadyFile> for GlobalMergeExecutor {
    fn consume(
        &mut self,
        job: GlobalMergeJob,
        sender: &mut dyn Sender<ReadyFile>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        match job {
            GlobalMergeJob::Ready(file) => sender.send(file)?,
            GlobalMergeJob::Identity {
                context, output, ..
            } => sender.send(ready_file(&context, output))?,
            GlobalMergeJob::Task { context, task } => {
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                if let Some(output) = task.execute_mapping(allocator)? {
                    sender.send(ready_file(&context, output))?;
                }
            }
        }
        Ok(())
    }
}

/// The file's rows over every node's decoded batches. The global order
/// ranks rows of the nodes' key batches, and each of those ranks a row of that
/// node's own batches, so the two orders compose into one over all batches.
fn ready_file(context: &Arc<FileMergeContext>, global: MergedMapping) -> ReadyFile {
    debug_assert_eq!(global.row_count, context.row_count);
    let mut rows_by_node = vec![0usize; context.local_outputs.len()];
    let mut batches: Vec<LocatedBatch> = Vec::new();
    // Per participating node, in the order its keys entered the global merge:
    // where its key batches and its own batches start in the flattened lists,
    // and the first row of each of its key batches.
    let mut first_batch = Vec::new();
    let mut key_batch_first_row: Vec<Vec<usize>> = Vec::new();
    let mut locals = Vec::new();
    for (node_id, local) in context.local_outputs.iter().enumerate() {
        let Some(local) = local.get() else {
            continue;
        };
        rows_by_node[node_id] += local.row_count;
        let mut first_row = 0;
        key_batch_first_row.push(
            local
                .keys
                .iter()
                .map(|keys| {
                    let row = first_row;
                    first_row += keys.num_rows();
                    row
                })
                .collect(),
        );
        first_batch.push(batches.len());
        batches.extend(
            local
                .batches
                .iter()
                .map(|batch| LocatedBatch::new(batch.clone(), node_id)),
        );
        locals.push(local);
    }
    let mut first_key_batch = Vec::with_capacity(locals.len());
    let mut key_batches = 0;
    for local in &locals {
        first_key_batch.push(key_batches);
        key_batches += local.keys.len();
    }
    let mapping = global
        .mapping
        .iter()
        .map(|&(key_batch, row)| {
            let key_batch = key_batch as usize;
            let node = first_key_batch.partition_point(|&first| first <= key_batch) - 1;
            let key_batch_of_node = key_batch - first_key_batch[node];
            // A node's key batches are its batches projected, so in order a
            // key batch's row is that batch's row.
            let (batch, row) = if locals[node].in_order {
                (key_batch_of_node as u32, row)
            } else {
                let local_row = key_batch_first_row[node][key_batch_of_node] + row as usize;
                locals[node].mapping[local_row]
            };
            ((first_batch[node] + batch as usize) as u32, row)
        })
        .collect();
    ReadyFile {
        plan: context.plan.clone(),
        batches,
        rows: FileRows::Mapped(Arc::new(mapping)),
        row_count: context.row_count,
        target_node: dispatch::dominant_node(&rows_by_node),
    }
}

/// The file's rows when one node held them all: that node's order is the
/// file's, with nothing to compose.
fn ready_file_of_one_node(context: &Arc<FileMergeContext>, node_id: usize) -> ReadyFile {
    let local = context.local_outputs[node_id]
        .get()
        .expect("the node completed its local merge");
    ReadyFile {
        plan: context.plan.clone(),
        batches: local
            .batches
            .iter()
            .map(|batch| LocatedBatch::new(batch.clone(), node_id))
            .collect(),
        rows: if local.in_order {
            FileRows::InOrder
        } else {
            FileRows::Mapped(local.mapping.clone())
        },
        row_count: context.row_count,
        target_node: node_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::CollectSender;
    use dispatch::{MergeRun, OrderBy};

    fn batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    #[test]
    fn local_results_feed_one_global_merge() {
        init_test_free_pool(16);
        let context = Arc::new(FileMergeContext {
            plan: super::super::types::FilePlan {
                file_id: 0,
                base_row_group_id: 0,
                assembly_worker: 0,
                partition: None,
                target_rows_per_group: 1_000,
                max_file_size: None,
            },
            order_by: Arc::from([OrderBy::new(0, false, true)]),
            row_count: 8,
            // Past the whole-gather size, so the merge is exercised by mapping.
            in_memory_bytes: usize::MAX,
            local_outputs: (0..2).map(|_| OnceLock::new()).collect(),
            nodes_remaining: 2.into(),
        });
        let inputs = [
            FileOrderInput::Merge(super::super::types::NodeMergeRequest {
                context: context.clone(),
                node_id: 0,
                runs: vec![
                    MergeRun::new(vec![batch(&[1, 7])], 0),
                    MergeRun::new(vec![batch(&[3, 5])], 0),
                ],
            }),
            FileOrderInput::Merge(super::super::types::NodeMergeRequest {
                context: context.clone(),
                node_id: 1,
                runs: vec![
                    MergeRun::new(vec![batch(&[2, 8])], 1),
                    MergeRun::new(vec![batch(&[4, 6])], 1),
                ],
            }),
        ];

        let mut local_planner = LocalMergePlanner;
        let mut local_jobs = CollectSender::new();
        for input in inputs {
            local_planner
                .consume(
                    input,
                    &mut local_jobs,
                    &mut dispatch::TestOperatorIO::default().io(),
                )
                .unwrap();
        }
        let mut local_executor = LocalMergeExecutor::default();
        let mut local_results = CollectSender::new();
        for job in local_jobs.items {
            local_executor
                .consume(
                    job,
                    &mut local_results,
                    &mut dispatch::TestOperatorIO::default().io(),
                )
                .unwrap();
        }
        assert_eq!(local_results.items.len(), 2);

        let mut global_planner = GlobalMergePlanner;
        let mut global_jobs = CollectSender::new();
        for result in local_results.items {
            global_planner
                .consume(
                    result,
                    &mut global_jobs,
                    &mut dispatch::TestOperatorIO::default().io(),
                )
                .unwrap();
        }
        let mut global_executor = GlobalMergeExecutor::default();
        let mut ready_files = CollectSender::new();
        for job in global_jobs.items {
            global_executor
                .consume(
                    job,
                    &mut ready_files,
                    &mut dispatch::TestOperatorIO::default().io(),
                )
                .unwrap();
        }

        assert_eq!(ready_files.items.len(), 1);
        let file = &ready_files.items[0];
        let FileRows::Mapped(mapping) = &file.rows else {
            panic!("a merged file's rows are mapped");
        };
        let keys: Vec<i64> = mapping
            .iter()
            .map(|&(batch, row)| {
                file.batches[batch as usize]
                    .batch()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(row as usize)
            })
            .collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
