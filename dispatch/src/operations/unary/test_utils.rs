//! Shared test utilities for unary operators.

use crate::operations::channels::Sender;
use crate::operations::unary::Unary;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};
use crate::worker::install_test_worker_waker;
use arrow_array::{
    Decimal128Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};

/// A [`Sender`] that collects all sent items for later inspection.
pub struct CollectSender<T = RecordBatch> {
    pub items: Vec<T>,
}

impl<T> CollectSender<T> {
    pub fn new() -> Self {
        Self { items: vec![] }
    }
}

impl<T> Default for CollectSender<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectSender<RecordBatch> {
    /// Total number of rows across all collected batches.
    pub fn total_rows(&self) -> usize {
        self.items.iter().map(|b| b.num_rows()).sum()
    }

    /// All values from column `col` as sorted i32s.
    pub fn sorted_i32_column(&self, col: usize) -> Vec<i32> {
        let mut values = self.i32_column(col);
        values.sort();
        values
    }

    /// All values from column `col` as i32s, preserving order.
    pub fn i32_column(&self, col: usize) -> Vec<i32> {
        self.items
            .iter()
            .flat_map(|b| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// All values from column `col` as i64s, preserving order (e.g. aggregate
    /// outputs like `COUNT(*)`, which are `Int64`).
    pub fn i64_column(&self, col: usize) -> Vec<i64> {
        self.items
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

    /// All values from a `Decimal128` column `col` as `i128`, in order — a wide
    /// (`i128`) value slot renders here (e.g. a numeric extreme co-located with a
    /// string extreme in a `Dynamic`, or a wide grouped `SUM`).
    pub fn decimal128_column(&self, col: usize) -> Vec<i128> {
        self.items
            .iter()
            .flat_map(|b| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// All values from a `Float64` column `col` as `f64`, in order (e.g. a
    /// grouped float `SUM`/`MIN`/`MAX` output).
    pub fn f64_column(&self, col: usize) -> Vec<f64> {
        self.items
            .iter()
            .flat_map(|b| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// All values from a `Utf8View` column `col` as owned strings, in order
    /// (e.g. string group-by keys).
    pub fn string_column(&self, col: usize) -> Vec<String> {
        self.items
            .iter()
            .flat_map(|b| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl<T> Sender<T> for CollectSender<T> {
    fn send(&mut self, item: T) -> std::result::Result<(), crate::operations::channels::Error> {
        self.items.push(item);
        Ok(())
    }
}

/// Feed `inputs` through a [`Unary`] operator and return all output items.
pub fn run_unary<I, O, U: Unary<I, O>>(mut unary: U, inputs: Vec<I>) -> Vec<O> {
    let mut sender = CollectSender::new();
    for item in inputs {
        unary.consume(item, &mut sender).unwrap();
    }
    sender.items
}

/// Feed `inputs` through a [`Unary`] operator, then drain via `finish()`.
///
/// Unlike [`run_unary`], this also calls `finish()` in a loop to collect
/// output that the operator produces lazily after consumption (e.g. the
/// Decoder which batches across multiple pages).
pub fn run_unary_to_completion<I, O, U: Unary<I, O>>(mut unary: U, inputs: Vec<I>) -> Vec<O> {
    let mut sender = CollectSender::new();
    for item in inputs {
        unary.consume(item, &mut sender).unwrap();
    }
    loop {
        unary.run(&mut sender).unwrap();
        if unary.finish(&mut sender).unwrap() {
            break;
        }
    }
    sender.items
}

/// Drive a set of [`Consumer`]s through the full consume → output lifecycle.
///
/// 1. Feeds `worker_batches[i]` into `consumers[i]`
/// 2. Calls `into_outputter()` on all consumers (dropping internal senders)
/// 3. Drives all outputters until done
///
/// Returns a [`CollectSender`] containing all output batches.
pub fn run_consumers<C: Consumer<RecordBatch, RecordBatch>>(
    consumers: Vec<C>,
    worker_batches: Vec<Vec<RecordBatch>>,
) -> CollectSender<RecordBatch> {
    install_test_worker_waker();
    let mut dummy = CollectSender::new();

    let mut consumers = consumers;
    for (consumer, batches) in consumers.iter_mut().zip(&worker_batches) {
        for batch in batches {
            consumer.consume(batch.clone(), &mut dummy).unwrap();
        }
    }

    let mut outputters: Vec<_> = consumers
        .into_iter()
        .filter_map(|c| c.into_outputter().unwrap())
        .collect();

    let mut sender = CollectSender::new();
    let mut all_done = false;
    while !all_done {
        all_done = true;
        for outputter in &mut outputters {
            if !outputter.output(&mut sender).unwrap() {
                all_done = false;
            }
        }
    }
    sender
}
