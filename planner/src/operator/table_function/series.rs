//! The series functions `range` (exclusive upper bound) and `generate_series`
//! (inclusive). Integer arguments give integers; two timestamp bounds and an
//! interval step give timestamps. A call with a NULL argument returns no rows.
//!
//! A call with constant arguments is a table of its own, bound as a
//! [`BoundSeries`]. It streams from a lazy source operator that emits one
//! [`SERIES_CHUNK_ROWS`] batch each time it is polled, so a huge range never
//! materializes at once and a downstream `LIMIT`/aggregate stops it early.
//!
//! A call whose arguments are columns of an earlier `FROM` item is lateral,
//! bound as a [`BoundLateralSeries`]: it returns one series per input row.

use super::lateral::{BoundLateralFunction, LateralRowGenerator, LateralTableFunction};
use super::{TableFunction, TableFunctionSignature, invalid_argument};
use crate::catalog::{
    BoundTable, CatalogTransaction, Column, DynamicScanPredicate, Result as CatalogResult,
    TableReference, TableRevision,
};
use crate::expression::add_months_to_timestamp;
use crate::types::{
    MICROS_PER_DAY, NANOS_PER_MICRO, Type, UTC_TIMEZONE, build_scalar_value, physical_arrow_type,
    type_from_physical,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, IntervalMonthDayNanoType, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, Scalar, TimestampMicrosecondArray,
};
use arrow_schema::{Field, Schema, SchemaRef};
use dispatch::{Built, BuiltNullary};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, Projection,
    RecordBatchOperatorSpec, Sender, WorkStatus,
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
    pub(super) const fn range() -> Self {
        Self {
            name: "range",
            inclusive: false,
        }
    }

    pub(super) const fn generate_series() -> Self {
        Self {
            name: "generate_series",
            inclusive: true,
        }
    }

    /// The type of series the overload taking `argument_types` returns.
    fn resolve_value_type(&self, argument_types: &[Type]) -> CatalogResult<SeriesValueType> {
        match argument_types {
            [Type::Int64]
            | [Type::Int64, Type::Int64]
            | [Type::Int64, Type::Int64, Type::Int64] => Ok(SeriesValueType::Integer),
            [Type::Timestamp, Type::Timestamp, Type::Interval] => Ok(SeriesValueType::Timestamp),
            [Type::TimestampTz, Type::TimestampTz, Type::Interval] => {
                Ok(SeriesValueType::TimestampTz)
            }
            other => Err(self.reject_arguments(format!("no overload takes {other:?}"))),
        }
    }

    fn reject_arguments(&self, message: String) -> crate::catalog::Error {
        crate::catalog::Error::Other(Box::new(invalid_argument(self.name, message)))
    }
}

impl TableFunction for SeriesTableFunction {
    fn name(&self) -> &str {
        self.name
    }

    fn signatures(&self) -> Vec<TableFunctionSignature> {
        let integer_bounds = (1..=3).map(|argument_count| vec![Type::Int64; argument_count]);
        let timestamp_bounds = [Type::Timestamp, Type::TimestampTz]
            .map(|bound_type| vec![bound_type.clone(), bound_type, Type::Interval]);
        integer_bounds
            .chain(timestamp_bounds)
            .map(|arguments| TableFunctionSignature { arguments })
            .collect()
    }

    fn as_lateral(&self) -> Option<&dyn LateralTableFunction> {
        Some(self)
    }

    fn bind(
        &self,
        arguments: &[ScalarValue],
        _transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<Box<dyn BoundTable>> {
        // One row of argument columns, the shape a lateral call reads its
        // arguments from.
        let argument_columns: Vec<ArrayRef> = arguments
            .iter()
            .map(|argument| build_scalar_value(argument.clone()).map(Scalar::into_inner))
            .collect::<Result<_, _>>()
            .map_err(|error| self.reject_arguments(error.to_string()))?;
        let argument_types = argument_columns
            .iter()
            .map(|column| {
                type_from_physical(column.data_type()).ok_or_else(|| {
                    self.reject_arguments(format!("no overload takes a {}", column.data_type()))
                })
            })
            .collect::<CatalogResult<Vec<Type>>>()?;
        let value_type = self.resolve_value_type(&argument_types)?;
        let series = Series::read_from_row(value_type, self.inclusive, &argument_columns, 0)
            .map_err(|message| self.reject_arguments(message))?;

        let written_arguments: Vec<String> = arguments.iter().map(ToString::to_string).collect();
        Ok(Box::new(BoundSeries {
            name: self.name,
            call: format!("{}({})", self.name, written_arguments.join(",")),
            value_type,
            series,
        }))
    }
}

impl LateralTableFunction for SeriesTableFunction {
    fn bind(&self, argument_types: &[Type]) -> CatalogResult<Box<dyn BoundLateralFunction>> {
        Ok(Box::new(BoundLateralSeries {
            name: self.name,
            inclusive: self.inclusive,
            value_type: self.resolve_value_type(argument_types)?,
        }))
    }
}

/// What a series counts in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeriesValueType {
    Integer,
    /// Microseconds since the epoch.
    Timestamp,
    /// Microseconds since the epoch, in UTC.
    TimestampTz,
}

impl SeriesValueType {
    fn column_type(self) -> Type {
        match self {
            SeriesValueType::Integer => Type::Int64,
            SeriesValueType::Timestamp => Type::Timestamp,
            SeriesValueType::TimestampTz => Type::TimestampTz,
        }
    }

    /// The series function's one column, named after the function so
    /// `SELECT *` reports it as `range` / `generate_series`, matching DuckDB.
    fn column_field(self, function_name: &str) -> Field {
        Field::new(
            function_name,
            physical_arrow_type(&self.column_type()),
            false,
        )
    }

    fn build_column(self, values: Vec<i64>) -> ArrayRef {
        match self {
            SeriesValueType::Integer => Arc::new(Int64Array::from(values)),
            SeriesValueType::Timestamp => Arc::new(TimestampMicrosecondArray::from(values)),
            SeriesValueType::TimestampTz => {
                Arc::new(TimestampMicrosecondArray::from(values).with_timezone(UTC_TIMEZONE))
            }
        }
    }
}

/// The values one call returns, from `next_value` towards `stop`. Iterating
/// yields them in order.
#[derive(Clone, Debug)]
struct Series {
    /// The value the series yields next. `None` once it has ended.
    next_value: Option<i64>,
    stop: i64,
    /// The whole calendar months each step moves a timestamp by.
    step_months: i64,
    /// What each step then adds: an integer step, or an interval's days and
    /// time in microseconds.
    step_distance: i64,
    /// Whether `stop` itself belongs to the series.
    inclusive: bool,
}

impl Series {
    /// The series that row `row` of a call's argument columns asks for: the
    /// empty one when an argument is NULL.
    fn read_from_row(
        value_type: SeriesValueType,
        inclusive: bool,
        arguments: &[ArrayRef],
        row: usize,
    ) -> Result<Series, String> {
        if arguments.iter().any(|argument| argument.is_null(row)) {
            return Ok(Series::empty());
        }
        let (start, stop, step_months, step_distance) = match value_type {
            SeriesValueType::Integer => {
                let integers: Vec<i64> = arguments
                    .iter()
                    .map(|argument| argument.as_primitive::<Int64Type>().value(row))
                    .collect();
                match integers[..] {
                    // A single argument is the stop; the series starts at 0, step 1.
                    [stop] => (0, stop, 0, 1),
                    [start, stop] => (start, stop, 0, 1),
                    [start, stop, step] => (start, stop, 0, step),
                    _ => unreachable!("an integer series takes one to three arguments"),
                }
            }
            SeriesValueType::Timestamp | SeriesValueType::TimestampTz => {
                let [start, stop] = [0, 1].map(|bound| {
                    arguments[bound]
                        .as_primitive::<TimestampMicrosecondType>()
                        .value(row)
                });
                if [start, stop]
                    .iter()
                    .any(|&bound| is_infinite_timestamp(bound))
                {
                    return Err("the bounds cannot be infinite".to_string());
                }
                let step = arguments[2]
                    .as_primitive::<IntervalMonthDayNanoType>()
                    .value(row);
                // A day is a fixed 24 hours: the session zone is UTC, which has
                // no daylight saving shift to stretch a day across.
                let step_distance = i64::from(step.days)
                    .checked_mul(MICROS_PER_DAY)
                    .and_then(|day_micros| {
                        day_micros.checked_add(step.nanoseconds / NANOS_PER_MICRO)
                    })
                    .ok_or("the step is too long to count in microseconds")?;
                let step_parts = [
                    i64::from(step.months),
                    i64::from(step.days),
                    step.nanoseconds,
                ];
                if step_parts.iter().any(|&part| part > 0)
                    && step_parts.iter().any(|&part| part < 0)
                {
                    return Err(
                        "a step whose months, days and time point in different directions is \
                         not supported"
                            .to_string(),
                    );
                }
                (start, stop, i64::from(step.months), step_distance)
            }
        };
        if step_months == 0 && step_distance == 0 {
            return Err("step must not be zero".to_string());
        }
        Ok(Series {
            next_value: Some(start),
            stop,
            step_months,
            step_distance,
            inclusive,
        })
    }

    fn empty() -> Series {
        Series {
            next_value: None,
            stop: 0,
            step_months: 0,
            step_distance: 0,
            inclusive: false,
        }
    }

    /// Whether `value` lies beyond the end of the series, accounting for the
    /// step's direction.
    fn is_past_stop(&self, value: i64) -> bool {
        let ascending = self.step_months > 0 || self.step_distance > 0;
        match (ascending, self.inclusive) {
            (true, true) => value > self.stop,
            (true, false) => value >= self.stop,
            (false, true) => value < self.stop,
            (false, false) => value <= self.stop,
        }
    }

    /// The value one step after `value`. `None` when it is past what an `i64`
    /// holds, and so past any `stop`.
    fn step_from(&self, value: i64) -> Option<i64> {
        let moved_by_months = match self.step_months {
            0 => value,
            months => add_months_to_timestamp(value, months)?,
        };
        moved_by_months.checked_add(self.step_distance)
    }
}

impl Iterator for Series {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        let value = self.next_value.filter(|&value| !self.is_past_stop(value))?;
        self.next_value = self.step_from(value);
        Some(value)
    }
}

/// DuckDB's `infinity` and `-infinity` timestamps.
fn is_infinite_timestamp(micros: i64) -> bool {
    micros == i64::MAX || micros == -i64::MAX
}

/// A series call with constant arguments.
#[derive(Clone, Debug)]
struct BoundSeries {
    name: &'static str,
    /// The call as written, which tells one series table from another.
    call: String,
    value_type: SeriesValueType,
    series: Series,
}

impl BoundSeries {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![self.value_type.column_field(self.name)]))
    }

    fn compile_full(&self, dispatcher: &DataFlowDispatcher) -> RecordBatchOperatorSpec {
        let schema = self.schema();
        // The series is a single running counter, so exactly one worker produces
        // it (claiming the shared flag) and the rest idle, mirroring `DummyScan`.
        let claimed = Arc::new(AtomicBool::new(false));
        let factories = (0..dispatcher.worker_count()).map(|_| SeriesSourceFactory {
            claimed: claimed.clone(),
            schema: schema.clone(),
            value_type: self.value_type,
            series: self.series.clone(),
        });
        RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
    }

    fn projected_schema(&self, projection: &Projection) -> SchemaRef {
        if projection.column_indices.is_empty() {
            Arc::new(Schema::empty())
        } else {
            self.schema()
        }
    }
}

impl BoundTable for BoundSeries {
    fn table_reference(&self) -> TableReference {
        TableReference {
            datastore: "system".to_string(),
            schema: crate::DEFAULT_SCHEMA_NAME.to_string(),
            table: self.name.to_string(),
        }
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: self.call.clone(),
            version: "0".to_string(),
        }
    }

    fn is_plan_cacheable(&self) -> bool {
        false
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        if !projection
            .column_indices
            .iter()
            .all(|column_index| *column_index == 0)
        {
            return Err(crate::catalog::Error::Other(
                "series projection contains a column other than 0".into(),
            ));
        }
        let projected_schema = self.projected_schema(&projection);
        let column_indices = Arc::new(projection.column_indices);
        Ok(self.compile_full(dispatcher).project(move || {
            let projected_schema = projected_schema.clone();
            let column_indices = column_indices.clone();
            move |batch| {
                Ok(if column_indices.is_empty() {
                    RecordBatch::try_new_with_options(
                        projected_schema.clone(),
                        Vec::new(),
                        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
                    )?
                } else {
                    batch.project(&column_indices)?
                })
            }
        }))
    }

    fn columns(&self) -> Vec<Column> {
        vec![Column {
            name: self.name.to_string(),
            col_type: self.value_type.column_type(),
        }]
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }
}

/// Builds one worker's [`SeriesSource`]. All share `claimed` so exactly one wins
/// the right to produce the series.
struct SeriesSourceFactory {
    claimed: Arc<AtomicBool>,
    schema: SchemaRef,
    value_type: SeriesValueType,
    series: Series,
}

impl NullaryFactory<RecordBatch> for SeriesSourceFactory {
    type Nullary = SeriesSource;

    fn build_nullary(self) -> BuiltNullary<SeriesSource> {
        Built::new(SeriesSource {
            claimed: self.claimed,
            is_producer: None,
            schema: self.schema,
            value_type: self.value_type,
            series: self.series,
            done: false,
        })
    }
}

/// Streams `range`/`generate_series` in [`SERIES_CHUNK_ROWS`] chunks: each poll
/// emits one chunk and advances the series, so the full range is never resident
/// at once.
struct SeriesSource {
    claimed: Arc<AtomicBool>,
    /// `None` until the first poll decides this worker's role; `Some(true)` for
    /// the single producer, `Some(false)` for the idle ones.
    is_producer: Option<bool>,
    schema: SchemaRef,
    value_type: SeriesValueType,
    series: Series,
    done: bool,
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
        values.extend(self.series.by_ref().take(SERIES_CHUNK_ROWS));
        if values.is_empty() {
            self.done = true;
            return Ok(WorkStatus::Pending);
        }
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![self.value_type.build_column(values)],
        )
        .expect("a chunk of the series column is always well-formed");
        sender.send(batch)?;
        Ok(WorkStatus::Ran)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.done)
    }
}

/// A lateral series call: one series per input row.
#[derive(Clone, Debug)]
struct BoundLateralSeries {
    name: &'static str,
    inclusive: bool,
    value_type: SeriesValueType,
}

impl BoundLateralFunction for BoundLateralSeries {
    fn name(&self) -> &str {
        self.name
    }

    fn columns(&self) -> Vec<Column> {
        vec![Column {
            name: self.name.to_string(),
            col_type: self.value_type.column_type(),
        }]
    }

    fn clone_box(&self) -> Box<dyn BoundLateralFunction> {
        Box::new(self.clone())
    }

    fn create_row_generator(&self) -> Box<dyn LateralRowGenerator> {
        Box::new(SeriesRowGenerator {
            function: self.clone(),
            series: Series::empty(),
            generated_values: Vec::new(),
        })
    }
}

/// Generates the series of one input row at a time.
struct SeriesRowGenerator {
    function: BoundLateralSeries,
    /// What is left of the series of the call that last started.
    series: Series,
    generated_values: Vec<i64>,
}

impl LateralRowGenerator for SeriesRowGenerator {
    fn start_call(
        &mut self,
        arguments: &[ArrayRef],
        row: usize,
    ) -> Result<(), crate::compile::Error> {
        self.series = Series::read_from_row(
            self.function.value_type,
            self.function.inclusive,
            arguments,
            row,
        )
        .map_err(|message| invalid_argument(self.function.name, message))?;
        Ok(())
    }

    fn generate_rows(&mut self, limit: usize) -> usize {
        let generated_before = self.generated_values.len();
        self.generated_values.reserve(limit);
        self.generated_values
            .extend(self.series.by_ref().take(limit));
        self.generated_values.len() - generated_before
    }

    fn take_columns(&mut self) -> Vec<ArrayRef> {
        let values = std::mem::take(&mut self.generated_values);
        vec![self.function.value_type.build_column(values)]
    }
}
