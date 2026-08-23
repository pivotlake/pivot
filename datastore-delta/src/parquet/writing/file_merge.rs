//! Two-level NUMA-aware ordering for Parquet file candidates.
//!
//! A local planner and executor merge all runs belonging to one node. The
//! global planner then merges the single result from each participating node.
//! Both planners use [`dispatch::KWayMergePlan`], so an empty or one-run level is
//! an identity operation with no Arrow allocation.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use dispatch::memory::SlabAllocator;
use dispatch::{KWayMergePlan, MergedOutput, Sender, Unary, UnaryResult};

use super::types::{
    FileMergeContext, FileOrderInput, GlobalMergeJob, LocalMergeJob, LocalMergeResult, ReadyFile,
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
                output: MergedOutput::empty(),
            })?,
            KWayMergePlan::Identity(output) => sender.send(LocalMergeJob::Identity {
                context: request.context,
                node_id: request.node_id,
                output,
            })?,
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
                if let Some(output) = task.execute(allocator)? {
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

        let node_runs = context
            .local_outputs
            .iter()
            .enumerate()
            .filter_map(|(node_id, output)| {
                output.get().cloned().map(|output| output.into_run(node_id))
            })
            .collect();
        let plan = KWayMergePlan::try_new(
            context.order_by.clone(),
            node_runs,
            context.local_outputs.len(),
        )?;
        match plan {
            KWayMergePlan::Empty => unreachable!("a completed file has at least one row"),
            KWayMergePlan::Identity(output) => {
                let node_id = output
                    .batches()
                    .first()
                    .expect("a nonempty identity has a batch")
                    .node_id();
                sender.send(GlobalMergeJob::Identity {
                    context,
                    output,
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
                if let Some(output) = task.execute(allocator)? {
                    sender.send(ready_file(&context, output))?;
                }
            }
        }
        Ok(())
    }
}

fn ready_file(context: &Arc<FileMergeContext>, output: MergedOutput) -> ReadyFile {
    debug_assert_eq!(output.row_count(), context.row_count);
    let mut rows_by_node = vec![0usize; context.local_outputs.len()];
    for batch in output.batches() {
        rows_by_node[batch.node_id()] += batch.batch().num_rows();
    }
    let target_node = dispatch::dominant_node(&rows_by_node);
    ReadyFile {
        plan: context.plan.clone(),
        batches: output.into_batches(),
        row_count: context.row_count,
        target_node,
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
        let keys: Vec<i64> = ready_files.items[0]
            .batches
            .iter()
            .flat_map(|batch| {
                batch
                    .batch()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
