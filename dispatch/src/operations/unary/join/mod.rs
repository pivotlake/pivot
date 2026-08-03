//! Parallel hash join: a radix-partitioned build phase populating a shared
//! directory + arena, and a prefetch-pipelined probe phase emitting the joined
//! rows.
//!
//! Build and probe are separate roots in one dataflow. Every probe-input root
//! is gated on a publication flag set after all build partition jobs complete,
//! then the probe reads the populated [`JoinTable`].
//!
//! Output layout: the probe-side columns listed in [`JoinOutputColumns`] (in
//! list order) followed by the listed build-side columns. The supported shape
//! is an equi-join on a single fixed-width key, in one of the three
//! [`JoinKind`]s below.
//!
//! A build-side outer join ([`JoinKind::BuildOuter`]) also emits every build
//! row no probe row matched, its probe columns null-filled. Which rows those
//! are is only known once every worker has stopped probing, so the probe marks
//! each matched build row in a shared flag array and the workers scan it
//! together behind a barrier ([`UnmatchedScan`]).
//!
//! A probe-side semi join ([`JoinKind::ProbeSemi`]) emits a probe row once if
//! the build side holds its key at all. Each probe row's candidates live in one
//! arena range, so the probe stops walking that range at the first key that
//! matches; nothing else about the phases changes, and no cross-worker state is
//! involved.

mod build;
mod directory;
mod factory;
pub use factory::JoinRecordBatchOperatorFactory;
mod probe;

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use arrow_schema::Field;

use crate::memory::MultiSlabBuffer;
use crate::operations::unary::join::directory::JoinDirectory;
pub(crate) use factory::create_for_workers as create_join_factories;

/// Which columns of each side the join emits, as indices into the probe and
/// build input schemas. Downstream operators that ignore some join columns
/// declare that here so the probe never materializes values nobody reads.
#[derive(Debug, Clone)]
pub struct JoinOutputColumns {
    pub probe: Vec<usize>,
    pub build: Vec<usize>,
}

impl JoinOutputColumns {
    /// Keep every column of both sides: probe columns then build columns.
    pub fn keep_all(probe_column_count: usize, build_column_count: usize) -> Self {
        Self {
            probe: (0..probe_column_count).collect(),
            build: (0..build_column_count).collect(),
        }
    }
}

/// Which rows a join emits.
#[derive(Debug, Clone)]
pub enum JoinKind {
    /// One output row per matching (probe row, build row) pair.
    Inner,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus one row per build
    /// row nothing matched, its probe columns null-filled.
    BuildOuter {
        /// The fields the probe columns take in the output. A worker can reach
        /// the unmatched pass without ever having seen a probe batch to read a
        /// schema off — an empty probe side, or simply a peer having taken all
        /// the work — so the caller states their shape up front. Every worker
        /// shaping its output from the same fields is also what keeps their
        /// batches concatenable downstream.
        probe_fields: Vec<Field>,
    },
    /// One output row per probe row that has at least one matching build row,
    /// with no duplicates for a probe row that matches several. That is also
    /// why such a join emits no build columns: there is no single build row to
    /// take them from, so [`JoinOutputColumns::build`] must be empty.
    ProbeSemi,
}

/// How a join is configured beyond the key type it is instantiated for.
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// The key column's index in the build input's schema.
    pub build_key_column: usize,
    /// The key column's index in the probe input's schema.
    pub probe_key_column: usize,
    /// Which columns of each side the join emits.
    pub output_columns: JoinOutputColumns,
    /// Which rows reach the output.
    pub kind: JoinKind,
}

/// The cross-worker state of a build-side outer join's unmatched pass.
pub(crate) struct UnmatchedScan {
    /// Probe workers that may still mark a matched build row. Each decrements
    /// it once, on entering `finish`, after which that worker never consumes
    /// again; at zero the flags are final and the scan below can start.
    probes_live: AtomicUsize,
    /// The next build payload row the scan hands out, so workers claim
    /// disjoint chunks of the flag array.
    cursor: AtomicUsize,
}

impl UnmatchedScan {
    fn new(worker_count: usize) -> Self {
        Self {
            probes_live: AtomicUsize::new(worker_count),
            cursor: AtomicUsize::new(0),
        }
    }
}

/// Interior-mutable storage shared by partitioned build workers.
pub(crate) struct JoinCell<T>(UnsafeCell<T>);

impl<T> JoinCell<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    pub(crate) fn get(&self) -> *mut T {
        self.0.get()
    }
}

// Build workers mutate disjoint directory partitions and arena ranges. The API
// keeps the raw cell private so only that partitioned build protocol can write.
unsafe impl<T: Send> Send for JoinCell<T> {}
unsafe impl<T: Send> Sync for JoinCell<T> {}

/// Shared hash-table state handed from the build factories to the probe
/// factories. `keys` and `rows` are parallel arenas indexed by the directory's
/// slot cursors: the full-width join key, and the row's index into
/// `build_rows` (the concatenated build-side payload batch). All fields are
/// populated by the build phase and published to probe only after every
/// partition job has run.
#[derive(Clone)]
pub(crate) struct JoinTable<K> {
    pub(crate) directory: Arc<JoinCell<JoinDirectory>>,
    pub(crate) keys: Arc<JoinCell<MultiSlabBuffer<K>>>,
    pub(crate) rows: Arc<JoinCell<MultiSlabBuffer<u32>>>,
    pub(crate) build_rows: Arc<JoinCell<Option<RecordBatch>>>,
    /// One zeroed byte per `build_rows` row, set to 1 by whichever probe worker
    /// matches that row. Only allocated for a build-side outer join, which is
    /// the only reader; the bytes are written as [`AtomicU8`](std::sync::atomic::AtomicU8)
    /// because several probe workers can match the same build row at once.
    pub(crate) matched: Arc<JoinCell<MultiSlabBuffer<u8>>>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    use crate::memory::init_test_free_pool;
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

    fn keyed_names_batch(keys: &[i64], names: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("key", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
            ])),
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(StringViewArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    fn extract_consumer<const BUILD_OUTER: bool>(
        breaker: PipelineBreaker<
            RecordBatch,
            (),
            JoinBuildConsumer<arrow_array::types::Int64Type, BUILD_OUTER>,
        >,
    ) -> JoinBuildConsumer<arrow_array::types::Int64Type, BUILD_OUTER> {
        match breaker {
            PipelineBreaker::Consuming(c) => c,
            _ => unreachable!(),
        }
    }

    struct JoinResult {
        batches: Vec<RecordBatch>,
    }

    impl JoinResult {
        fn rows(&self) -> usize {
            self.batches.iter().map(|b| b.num_rows()).sum()
        }
    }

    fn build_and_probe(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<false>(build_worker_batches, probe_batches, None)
    }

    /// Build-side outer: every build row reaches the output, with null probe
    /// columns when nothing matched.
    fn build_and_probe_outer(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
        probe_fields: Vec<Field>,
    ) -> JoinResult {
        run_join::<true>(build_worker_batches, probe_batches, Some(probe_fields))
    }

    /// Run one join to completion: one build worker per entry of
    /// `build_worker_batches`, and as many probe workers, all of whose `finish`
    /// is driven (the unmatched pass only starts once every one has arrived).
    /// The probe batches all go to the first, so the rest exercise a worker
    /// reaching that pass with no batch of its own.
    fn run_join<const BUILD_OUTER: bool>(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
        probe_fields: Option<Vec<Field>>,
    ) -> JoinResult {
        init_test_free_pool(16);
        let workers = build_worker_batches.len();
        let probe_column_count = probe_batches[0].num_columns();
        let build_column_count = build_worker_batches[0][0].num_columns();
        let spec = super::JoinSpec {
            build_key_column: 0,
            probe_key_column: 0,
            output_columns: super::JoinOutputColumns::keep_all(
                probe_column_count,
                build_column_count,
            ),
            kind: match probe_fields {
                Some(probe_fields) => super::JoinKind::BuildOuter { probe_fields },
                None => super::JoinKind::Inner,
            },
        };
        let (builds, probes, _) =
            factory::create_for_workers::<arrow_array::types::Int64Type, BUILD_OUTER, false>(
                spec, workers,
            );

        let mut consumers: Vec<_> = builds
            .into_iter()
            .map(|f| extract_consumer(f.build_unary()))
            .collect();

        let mut void = CollectSender::<()>::new();
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

        let mut probers: Vec<_> = probes.into_iter().map(|p| p.build_unary()).collect();
        let mut sender = CollectSender::new();
        for batch in probe_batches {
            probers[0].consume(batch, &mut sender).unwrap();
        }
        loop {
            let mut all_done = true;
            for probe in &mut probers {
                if !probe.finish(&mut sender).unwrap() {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }
        JoinResult {
            batches: sender.items,
        }
    }

    fn collect_i64_column(results: &[RecordBatch], col: usize) -> Vec<i64> {
        results
            .iter()
            .flat_map(|b| {
                b.column(col)
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
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys, vec![20]);
    }

    #[test]
    fn multiple_key_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30, 40, 50])]],
            vec![int64_batch(&[20, 40])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![20, 40]);
    }

    #[test]
    fn no_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99, 100])],
        );

        assert!(r.batches.is_empty());
    }

    #[test]
    fn duplicate_build_keys_produce_multiple_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 10, 20])]],
            vec![int64_batch(&[10])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 10]);
    }

    #[test]
    fn multiple_build_batches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20]), int64_batch(&[30, 40])]],
            vec![int64_batch(&[20, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![20, 30]);
    }

    #[test]
    fn two_build_workers() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20])], vec![int64_batch(&[30, 40])]],
            vec![int64_batch(&[10, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 30]);
    }

    #[test]
    fn empty_build_no_matches() {
        let r = build_and_probe(vec![vec![int64_batch(&[])]], vec![int64_batch(&[10])]);

        assert!(r.batches.is_empty());
    }

    #[test]
    fn empty_probe_no_output() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[])],
        );

        assert!(r.batches.is_empty());
    }

    #[test]
    fn duplicate_probe_keys() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20])]],
            vec![int64_batch(&[10, 10, 10])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys, vec![10, 10, 10]);
    }

    #[test]
    fn many_to_many() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 10])]],
            vec![int64_batch(&[10, 10])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys.len(), 4);
        assert!(keys.iter().all(|&k| k == 10));
    }

    #[test]
    fn multiple_probe_batches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[10]), int64_batch(&[30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 30]);
    }

    #[test]
    fn all_keys_match() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[1, 2, 3, 4, 5])]],
            vec![int64_batch(&[1, 2, 3, 4, 5])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn probe_handles_matches_duplicates_and_misses() {
        let build_batches = vec![vec![int64_batch(&[10, 10, 20, 30, 40])]];
        let probe_batches = vec![int64_batch(&[99, 10, 30, 99, 10])];

        let r = build_and_probe(build_batches, probe_batches);

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 10, 10, 10, 30]);
    }

    #[test]
    fn probe_columns_pass_through() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![keyed_names_batch(&[99, 20, 99, 30], &["a", "b", "c", "d"])],
        );

        // Output: probe columns (key, name), then build columns (key).
        let batch = &r.batches[0];
        assert_eq!(batch.num_columns(), 3);
        let names: Vec<&str> = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["b", "d"]);
        assert_eq!(collect_i64_column(&r.batches, 2), vec![20, 30]);
    }

    #[test]
    fn build_payload_columns_are_gathered() {
        let r = build_and_probe(
            vec![vec![keyed_names_batch(&[10, 20, 30], &["x", "y", "z"])]],
            vec![int64_batch(&[30, 10])],
        );

        // Output: probe columns (key), then build columns (key, name).
        let batch = &r.batches[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(collect_i64_column(&r.batches, 0), vec![30, 10]);
        assert_eq!(collect_i64_column(&r.batches, 1), vec![30, 10]);
        let names: Vec<&str> = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["z", "x"]);
    }

    #[test]
    fn build_payload_spans_workers_and_batches() {
        let r = build_and_probe(
            vec![
                vec![
                    keyed_names_batch(&[1, 2], &["a", "b"]),
                    keyed_names_batch(&[3], &["c"]),
                ],
                vec![keyed_names_batch(&[4, 5], &["d", "e"])],
            ],
            vec![int64_batch(&[5, 3, 1])],
        );

        let mut pairs: Vec<(i64, String)> = r
            .batches
            .iter()
            .flat_map(|b| {
                let keys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let names = b
                    .column(2)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..b.num_rows())
                    .map(|i| (keys.value(i), names.value(i).to_string()))
                    .collect::<Vec<_>>()
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (1, "a".to_string()),
                (3, "c".to_string()),
                (5, "e".to_string())
            ]
        );
    }

    #[test]
    fn keys_beyond_u32_do_not_alias() {
        // Two keys that collide when truncated to 32 bits.
        let low = 7;
        let high = 7 + (1i64 << 32);
        let r = build_and_probe(
            vec![vec![int64_batch(&[high])]],
            vec![int64_batch(&[low, high])],
        );

        assert_eq!(collect_i64_column(&r.batches, 0), vec![high]);
    }

    #[test]
    fn null_keys_never_match() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![Some(10), None]))],
        )
        .unwrap();
        let probe = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10), None]))],
        )
        .unwrap();

        let r = build_and_probe(vec![vec![build]], vec![probe]);

        let total: usize = r.batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn output_exceeding_batch_capacity_is_chunked() {
        // One probe row matching more build rows than one output batch holds.
        let n = crate::RECORD_BATCH_SIZE + 100;
        let build_keys = vec![10i64; n];
        let r = build_and_probe(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[10])],
        );

        let total: usize = r.batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, n);
        assert!(
            r.batches
                .iter()
                .all(|b| b.num_rows() <= crate::RECORD_BATCH_SIZE)
        );
    }

    #[test]
    fn probe_row_multiplicity_matches_build_duplicates() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99, 20, 99, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 0);
        keys.sort();
        assert_eq!(keys, vec![20, 30]);
    }

    #[test]
    fn large_build_exercises_multiple_partitions() {
        let keys: Vec<i64> = (0..1000).collect();
        let probe_keys: Vec<i64> = (500..600).collect();

        let r = build_and_probe(
            vec![vec![int64_batch(&keys)]],
            vec![int64_batch(&probe_keys)],
        );

        let mut matched = collect_i64_column(&r.batches, 1);
        matched.sort();
        assert_eq!(matched, probe_keys);
    }

    #[test]
    fn empty_build_still_completes_and_matches_nothing() {
        let r = build_and_probe(vec![vec![int64_batch(&[])]], vec![int64_batch(&[10])]);

        assert!(r.batches.is_empty());
    }

    fn nullable_key_field() -> Vec<Field> {
        vec![Field::new("key", DataType::Int64, true)]
    }

    /// Every output row as (probe key, build key), each null-aware.
    fn collect_key_pairs(results: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
        let mut pairs: Vec<(Option<i64>, Option<i64>)> = results
            .iter()
            .flat_map(|batch| {
                let key = |col: usize, row: usize| {
                    let column = batch
                        .column(col)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    column.is_valid(row).then(|| column.value(row))
                };
                (0..batch.num_rows())
                    .map(|row| (key(0, row), key(1, row)))
                    .collect::<Vec<_>>()
            })
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn unmatched_build_rows_come_out_with_null_probe_columns() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, Some(10)), (None, Some(30)), (Some(20), Some(20))]
        );
    }

    #[test]
    fn every_build_row_survives_a_probe_side_that_matches_nothing() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, Some(10)), (None, Some(20)), (None, Some(30))]
        );
    }

    #[test]
    fn a_build_row_matched_twice_is_not_also_reported_unmatched() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10])]],
            vec![int64_batch(&[10, 10])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(Some(10), Some(10)), (Some(10), Some(10))]
        );
    }

    #[test]
    fn null_keyed_build_rows_are_emitted_unmatched() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![Some(10), None]))],
        )
        .unwrap();

        let r = build_and_probe_outer(
            vec![vec![build]],
            vec![int64_batch(&[10])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, None), (Some(10), Some(10))]
        );
    }

    #[test]
    fn the_unmatched_pass_covers_every_build_worker() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[1, 2])], vec![int64_batch(&[3, 4])]],
            vec![int64_batch(&[3])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![
                (None, Some(1)),
                (None, Some(2)),
                (None, Some(4)),
                (Some(3), Some(3))
            ]
        );
    }

    #[test]
    fn an_outer_build_larger_than_one_batch_emits_every_row() {
        let build_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[7])],
            nullable_key_field(),
        );

        assert_eq!(r.rows(), build_keys.len());
    }
}
