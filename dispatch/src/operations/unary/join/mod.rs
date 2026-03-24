mod factory;
mod directory;
mod probe;
mod build;

use std::cell::UnsafeCell;
use std::sync::Arc;
pub use factory::{JoinBuildFactory, JoinProbeFactory, create_for_workers as create_join_factories};
use crate::operations::unary::join::directory::Directory;


pub(crate) type Value = (u64, u64);

/// Shared hash table state returned by [`JoinBuildFactory::create_for_workers`].
/// Hand this to the probe side after the build pipeline completes.
pub struct JoinTable {
    pub directory: Arc<UnsafeCell<Directory>>,
    pub arena: Arc<UnsafeCell<Vec<Value>>>,
}

unsafe impl Send for JoinTable {}

unsafe impl Sync for JoinTable {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use std::sync::atomic::Ordering;

    use crate::operations::channels::VoidSender;
    use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
    use crate::operations::unary::test_utils::CollectSender;
    use crate::operations::{Unary, UnaryFactory};

    use super::build::JoinBuildConsumer;
    use super::factory;

    fn int64_batch(keys: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(keys.to_vec()))],
        )
        .unwrap()
    }

    fn extract_consumer(
        breaker: PipelineBreaker<RecordBatch, (), JoinBuildConsumer>,
    ) -> JoinBuildConsumer {
        match breaker {
            PipelineBreaker::Consuming(c) => c,
            _ => unreachable!(),
        }
    }

    fn build_and_probe(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> Vec<RecordBatch> {
        let workers = build_worker_batches.len();
        let (builds, probes, _gate) = factory::create_for_workers(0, 0, workers);

        let mut consumers: Vec<_> = builds
            .into_iter()
            .map(|f| extract_consumer(f.build_unary()))
            .collect();

        let mut void = VoidSender;
        for (consumer, batches) in consumers.iter_mut().zip(&build_worker_batches) {
            for batch in batches {
                consumer.consume(batch.clone(), &mut void).unwrap();
            }
        }

        let mut outputters: Vec<_> = consumers
            .into_iter()
            .filter_map(|c| c.into_outputter().unwrap())
            .collect();

        loop {
            let mut all_done = true;
            for o in &mut outputters {
                if !o.output(&mut void).unwrap() {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }

        let mut probe = probes.into_iter().next().unwrap().build_unary();
        let mut sender = CollectSender::new();
        for batch in probe_batches {
            probe.consume(batch, &mut sender).unwrap();
        }
        sender.items
    }

    fn collect_build_keys(results: &[RecordBatch]) -> Vec<i64> {
        results
            .iter()
            .flat_map(|b| {
                b.column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    #[test]
    fn single_key_match() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
        );

        let keys = collect_build_keys(&results);
        assert_eq!(keys, vec![20]);
    }

    #[test]
    fn multiple_key_matches() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30, 40, 50])]],
            vec![int64_batch(&[20, 40])],
        );

        let mut keys = collect_build_keys(&results);
        keys.sort();
        assert_eq!(keys, vec![20, 40]);
    }

    #[test]
    fn no_matches() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99, 100])],
        );

        assert!(results.is_empty());
    }

    #[test]
    fn duplicate_build_keys_produce_multiple_matches() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 10, 20])]],
            vec![int64_batch(&[10])],
        );

        let mut keys = collect_build_keys(&results);
        keys.sort();
        assert_eq!(keys, vec![10, 10]);
    }

    #[test]
    fn multiple_build_batches() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 20]), int64_batch(&[30, 40])]],
            vec![int64_batch(&[20, 30])],
        );

        let mut keys = collect_build_keys(&results);
        keys.sort();
        assert_eq!(keys, vec![20, 30]);
    }

    #[test]
    fn two_build_workers() {
        let results = build_and_probe(
            vec![
                vec![int64_batch(&[10, 20])],
                vec![int64_batch(&[30, 40])],
            ],
            vec![int64_batch(&[10, 30])],
        );

        let mut keys = collect_build_keys(&results);
        keys.sort();
        assert_eq!(keys, vec![10, 30]);
    }

    #[test]
    fn empty_build_no_matches() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[])]],
            vec![int64_batch(&[10])],
        );

        assert!(results.is_empty());
    }

    #[test]
    fn empty_probe_no_output() {
        let results = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[])],
        );

        assert!(results.is_empty());
    }
}