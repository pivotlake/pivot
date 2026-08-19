//! The generic integer-series functions `range` (exclusive upper bound) and
//! `generate_series` (inclusive). They depend on nothing but their arguments.
//!
//! The series streams from a lazy source operator that emits one
//! [`SERIES_CHUNK_ROWS`] batch each time it is polled, so a huge range never
//! materializes at once and a downstream `LIMIT`/aggregate stops it early.

use super::{BoundTableFunction, TableFunction, TableFunctionRows, invalid_argument};
use crate::catalog::{Column, Result as CatalogResult};
use crate::compile::Error;
use crate::types::Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RecordBatchOperatorSpec, Sender,
    WorkStatus,
};
use duckdb_planner::ScalarValue;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Rows per batch the series source emits each time it is polled.
const SERIES_CHUNK_ROWS: usize = 8192;

/// `range`/`generate_series`. `inclusive` is the only difference: `range` stops
/// before `stop`, `generate_series` includes it.
pub(super) struct SeriesTableFunction {
    name: &'static str,
    inclusive: bool,
}

impl SeriesTableFunction {
    pub(super) fn range() -> Self {
        Self {
            name: "range",
            inclusive: false,
        }
    }

    pub(super) fn generate_series() -> Self {
        Self {
            name: "generate_series",
            inclusive: true,
        }
    }
}

impl TableFunction for SeriesTableFunction {
    fn name(&self) -> &str {
        self.name
    }

    fn argument_types(&self) -> Vec<Type> {
        // Only consulted for catalog-resolved functions; `range`/`generate_series`
        // are DuckDB built-ins, so the bridge never registers them through this.
        vec![Type::Int64]
    }

    fn bind(&self, arguments: &[ScalarValue]) -> CatalogResult<BoundTableFunction> {
        let nums = self.parse_i64_args(arguments)?;
        let (start, stop, step) = match nums.as_slice() {
            // A single argument is the stop; the series starts at 0, step 1.
            [stop] => (0, *stop, 1),
            [start, stop] => (*start, *stop, 1),
            [start, stop, step] => (*start, *stop, *step),
            _ => {
                return Err(invalid_argument(
                    self.name,
                    format!("expected 1 to 3 arguments, got {}", nums.len()),
                ));
            }
        };
        if step == 0 {
            return Err(invalid_argument(
                self.name,
                "step must not be zero".to_string(),
            ));
        }
        Ok(BoundTableFunction::Rows(Box::new(BoundSeries {
            name: self.name,
            start,
            stop,
            step,
            inclusive: self.inclusive,
        })))
    }
}

impl SeriesTableFunction {
    fn parse_i64_args(&self, args: &[ScalarValue]) -> CatalogResult<Vec<i64>> {
        // DuckDB binds `range`/`generate_series`'s integer overload as BIGINT, so
        // the arguments always arrive as `Int64`; anything else (e.g. the
        // `TIMESTAMP, INTERVAL` overload) pivot doesn't support here.
        args.iter()
            .map(|arg| match arg {
                ScalarValue::Int64(v) => Ok(*v),
                other => Err(invalid_argument(
                    self.name,
                    format!("expected an integer, got '{other}'"),
                )),
            })
            .collect()
    }
}

/// One `range`/`generate_series` call, its bounds resolved.
struct BoundSeries {
    name: &'static str,
    start: i64,
    stop: i64,
    step: i64,
    inclusive: bool,
}

impl BoundSeries {
    /// The one column, named after the function so `SELECT *` reports it as
    /// `range` / `generate_series`, matching DuckDB.
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            self.name,
            DataType::Int64,
            false,
        )]))
    }
}

impl TableFunctionRows for BoundSeries {
    fn columns(&self) -> Vec<Column> {
        vec![Column {
            name: self.name.to_string(),
            col_type: Type::Int64,
        }]
    }

    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec, Error> {
        let schema = self.schema();
        // The series is a single running counter, so exactly one worker produces
        // it (claiming the shared flag) and the rest idle, mirroring `DummyScan`.
        let claimed = Arc::new(AtomicBool::new(false));
        let factories = (0..dispatcher.worker_count()).map(|_| SeriesSourceFactory {
            claimed: claimed.clone(),
            schema: schema.clone(),
            start: self.start,
            stop: self.stop,
            step: self.step,
            inclusive: self.inclusive,
        });
        Ok(RecordBatchOperatorSpec::from_nullary(dispatcher, factories))
    }
}

/// Builds one worker's [`SeriesSource`]. All share `claimed` so exactly one wins
/// the right to produce the series.
struct SeriesSourceFactory {
    claimed: Arc<AtomicBool>,
    schema: SchemaRef,
    start: i64,
    stop: i64,
    step: i64,
    inclusive: bool,
}

impl NullaryFactory<RecordBatch> for SeriesSourceFactory {
    type Nullary = SeriesSource;

    fn build_nullary(self) -> SeriesSource {
        SeriesSource {
            claimed: self.claimed,
            is_producer: None,
            schema: self.schema,
            current: self.start,
            stop: self.stop,
            step: self.step,
            inclusive: self.inclusive,
            done: false,
        }
    }
}

/// Streams `range`/`generate_series` in [`SERIES_CHUNK_ROWS`] chunks: each poll
/// emits one chunk and advances the counter, so the full range is never resident
/// at once.
struct SeriesSource {
    claimed: Arc<AtomicBool>,
    /// `None` until the first poll decides this worker's role; `Some(true)` for
    /// the single producer, `Some(false)` for the idle ones.
    is_producer: Option<bool>,
    schema: SchemaRef,
    current: i64,
    stop: i64,
    step: i64,
    inclusive: bool,
    done: bool,
}

impl SeriesSource {
    /// Whether `current` has passed the (exclusive for `range`, inclusive for
    /// `generate_series`) end, accounting for the step's sign.
    fn past_end(&self) -> bool {
        if self.step > 0 {
            if self.inclusive {
                self.current > self.stop
            } else {
                self.current >= self.stop
            }
        } else if self.inclusive {
            self.current < self.stop
        } else {
            self.current <= self.stop
        }
    }
}

impl Nullary<RecordBatch> for SeriesSource {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        if self.done {
            return Ok(WorkStatus::Pending);
        }
        if self.is_producer.is_none() {
            // The first worker to flip the flag from false becomes the producer.
            self.is_producer = Some(!self.claimed.swap(true, Ordering::SeqCst));
        }
        if self.is_producer != Some(true) {
            self.done = true;
            return Ok(WorkStatus::Pending);
        }

        let mut values = Vec::with_capacity(SERIES_CHUNK_ROWS);
        while values.len() < SERIES_CHUNK_ROWS {
            if self.past_end() {
                self.done = true;
                break;
            }
            values.push(self.current);
            match self.current.checked_add(self.step) {
                Some(next) => self.current = next,
                // Stepping past i64::MAX/MIN ends the series after this value.
                None => {
                    self.done = true;
                    break;
                }
            }
        }

        if values.is_empty() {
            self.done = true;
            return Ok(WorkStatus::Pending);
        }
        let array = Arc::new(Int64Array::from(values)) as ArrayRef;
        let batch = RecordBatch::try_new(self.schema.clone(), vec![array])
            .expect("single BIGINT column chunk is always well-formed");
        sender.send(batch)?;
        Ok(WorkStatus::Ran)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.done)
    }
}
