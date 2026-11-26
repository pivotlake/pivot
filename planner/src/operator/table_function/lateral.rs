//! Lateral calls of table functions, such as
//! `FROM spans, generate_series(spans.first, spans.last)`.
//!
//! A call whose arguments are columns of an earlier `FROM` item cannot bind to
//! one table: it returns different rows for every row those columns come
//! from. A [`LateralTableFunction`] binds such a call to a
//! [`BoundLateralFunction`], and a [`LateralCall`] runs it: the function is
//! called once per input row, with that row's values as its arguments.

use crate::catalog::Column;
use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use crate::types::{Type, physical_arrow_type};
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_buffer::NullBuffer;
use arrow_schema::{Field, FieldRef, Schema, SchemaRef};
use dispatch::{RECORD_BATCH_SIZE, RecordBatchOperatorSpec, UnaryError, UnaryResult};
use std::fmt;
use std::sync::Arc;

/// A table function that can be called laterally.
pub trait LateralTableFunction: Send + Sync {
    /// Bind a lateral call whose arguments have `argument_types`, the
    /// parameter types of one of the function's signatures.
    fn bind(
        &self,
        argument_types: &[Type],
    ) -> crate::catalog::Result<Box<dyn BoundLateralFunction>>;
}

/// One lateral call after DuckDB has selected an overload for it.
pub trait BoundLateralFunction: fmt::Debug + Send + Sync {
    /// The function's SQL name.
    fn name(&self) -> &str;

    /// The columns one call returns.
    fn columns(&self) -> Vec<Column>;

    fn clone_box(&self) -> Box<dyn BoundLateralFunction>;

    /// Create the generator one worker runs its calls through.
    fn create_row_generator(&self) -> Box<dyn LateralRowGenerator>;
}

/// Runs one worker's calls of a lateral function, one call at a time.
pub trait LateralRowGenerator: Send {
    /// Start the call whose arguments are row `row` of `arguments`, one
    /// column per parameter.
    fn start_call(&mut self, arguments: &[ArrayRef], row: usize) -> Result<(), Error>;

    /// Generate up to `limit` more rows of the started call and return how
    /// many. Fewer than `limit` means the call has returned its last row.
    fn generate_rows(&mut self, limit: usize) -> usize;

    /// Hand over the rows generated since the last take, as the function's
    /// columns.
    fn take_columns(&mut self) -> Vec<ArrayRef>;
}

/// Calls a table function once per input row and emits each row a call
/// returns: the function's columns, then the input row's passthrough columns.
///
/// The passthrough columns are what the plan joins a call's rows back to the
/// query's other columns on, and that join never matches a NULL. A call that
/// returns rows for an input row with a NULL passthrough value would lose
/// them there, so it fails the query instead.
#[derive(Debug)]
pub struct LateralCall {
    pub(crate) function: Box<dyn BoundLateralFunction>,
    /// The call's arguments, one expression over the input per parameter.
    pub(crate) arguments: Vec<Expression>,
    /// The function's columns to emit, by position among
    /// [`BoundLateralFunction::columns`].
    pub(crate) function_columns: Vec<usize>,
    /// The input columns repeated beside each row a call returns.
    pub(crate) passthrough_columns: Vec<usize>,
}

impl LateralCall {
    pub(crate) fn output_types(&self, input: &[Type]) -> Vec<Type> {
        let function_column_types: Vec<Type> = self
            .function
            .columns()
            .into_iter()
            .map(|column| column.col_type)
            .collect();
        self.function_columns
            .iter()
            .map(|&column| function_column_types[column].clone())
            .chain(
                self.passthrough_columns
                    .iter()
                    .map(|&column| input[column].clone()),
            )
            .collect()
    }

    /// Lateral functions do not report nullability, so their columns stay
    /// conservative; a passthrough column is as nullable as its input.
    pub(crate) fn output_nullability(&self, input: &[bool]) -> Vec<bool> {
        self.function_columns
            .iter()
            .map(|_| true)
            .chain(self.passthrough_columns.iter().map(|&column| input[column]))
            .collect()
    }

    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let argument_builders: Arc<Vec<ExprFn>> = Arc::new(
            self.arguments
                .iter()
                .map(Expression::compile)
                .collect::<Result<_, _>>()?,
        );
        let columns = self.function.columns();
        let function_fields: Arc<Vec<FieldRef>> = Arc::new(
            self.function_columns
                .iter()
                .map(|&column| {
                    let Column { name, col_type } = &columns[column];
                    Arc::new(Field::new(name, physical_arrow_type(col_type), true))
                })
                .collect(),
        );
        let function: Arc<dyn BoundLateralFunction> = Arc::from(self.function.clone_box());
        let function_columns = Arc::new(self.function_columns.clone());
        let passthrough_columns = Arc::new(self.passthrough_columns.clone());

        Ok(input.flat_map(move || {
            let mut calls = WorkerCalls {
                function_name: function.name().to_string(),
                row_generator: function.create_row_generator(),
                argument_evals: argument_builders.iter().map(|builder| builder()).collect(),
                function_columns: function_columns.clone(),
                function_fields: function_fields.clone(),
                passthrough_columns: passthrough_columns.clone(),
                held_input: None,
            };
            move |input: &RecordBatch| calls.produce_next_batch(input)
        }))
    }
}

/// One worker's calls: it steps through the rows of the input batch it holds,
/// a batch of output rows at a time.
struct WorkerCalls {
    function_name: String,
    row_generator: Box<dyn LateralRowGenerator>,
    argument_evals: Vec<ExprEvalFn>,
    function_columns: Arc<Vec<usize>>,
    /// The fields of the emitted function columns, in output order.
    function_fields: Arc<Vec<FieldRef>>,
    passthrough_columns: Arc<Vec<usize>>,
    /// Where this worker is in the input batch it holds. `None` between
    /// batches.
    held_input: Option<HeldInput>,
}

/// The state of the calls over one input batch.
struct HeldInput {
    /// The batch's argument columns, one per parameter.
    arguments: Vec<ArrayRef>,
    /// The rows with a NULL in a passthrough column, when there are any.
    passthrough_nulls: Option<NullBuffer>,
    /// The function's fields, then the batch's passthrough fields.
    output_schema: SchemaRef,
    /// The row whose call is running, or starts next.
    row: usize,
    /// Whether `row`'s call has started and may have rows left to return.
    is_call_running: bool,
}

impl WorkerCalls {
    /// The next batch of rows the calls over `input` return, or `None` once
    /// every row's call has returned its last row.
    fn produce_next_batch(&mut self, input: &RecordBatch) -> UnaryResult<Option<RecordBatch>> {
        let row_count = input.num_rows();
        let held_input = self.held_input.get_or_insert_with(|| HeldInput {
            arguments: self
                .argument_evals
                .iter_mut()
                .map(|eval| eval(input).into_array(row_count))
                .collect(),
            passthrough_nulls: NullBuffer::union_many(
                self.passthrough_columns
                    .iter()
                    .map(|&column| input.column(column).nulls()),
            ),
            output_schema: Arc::new(Schema::new(
                self.function_fields
                    .iter()
                    .cloned()
                    .chain(
                        self.passthrough_columns
                            .iter()
                            .map(|&column| input.schema_ref().fields()[column].clone()),
                    )
                    .collect::<Vec<FieldRef>>(),
            )),
            row: 0,
            is_call_running: false,
        });

        // The input row each output row came from.
        let mut source_rows: Vec<u32> = Vec::new();
        while source_rows.len() < RECORD_BATCH_SIZE && held_input.row < row_count {
            if !held_input.is_call_running {
                self.row_generator
                    .start_call(&held_input.arguments, held_input.row)
                    .map_err(|error| UnaryError::Operator(Box::new(error)))?;
                held_input.is_call_running = true;
            }
            let wanted_rows = RECORD_BATCH_SIZE - source_rows.len();
            let generated_rows = self.row_generator.generate_rows(wanted_rows);
            let has_null_passthrough = held_input
                .passthrough_nulls
                .as_ref()
                .is_some_and(|nulls| nulls.is_null(held_input.row));
            if generated_rows > 0 && has_null_passthrough {
                return Err(UnaryError::Operator(
                    format!(
                        "a lateral call of {} returned rows for a row where a column its \
                         arguments read is NULL, which is not supported",
                        self.function_name
                    )
                    .into(),
                ));
            }
            source_rows.extend(std::iter::repeat_n(held_input.row as u32, generated_rows));
            if generated_rows < wanted_rows {
                held_input.is_call_running = false;
                held_input.row += 1;
            }
        }
        if source_rows.is_empty() {
            self.held_input = None;
            return Ok(None);
        }

        let output_schema = held_input.output_schema.clone();
        let generated_columns = self.row_generator.take_columns();
        let source_rows = UInt32Array::from(source_rows);
        let mut output_columns: Vec<ArrayRef> = self
            .function_columns
            .iter()
            .map(|&column| generated_columns[column].clone())
            .collect();
        for &column in self.passthrough_columns.iter() {
            output_columns.push(arrow::compute::take(
                input.column(column),
                &source_rows,
                None,
            )?);
        }
        Ok(Some(RecordBatch::try_new(output_schema, output_columns)?))
    }
}

impl fmt::Display for LateralCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arguments: Vec<String> = self.arguments.iter().map(ToString::to_string).collect();
        write!(
            formatter,
            "LateralCall({}({}))",
            self.function.name(),
            arguments.join(", ")
        )
    }
}
