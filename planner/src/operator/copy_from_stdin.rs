//! [`CopyFromStdin`] — the `COPY <table> FROM STDIN` statement, and its
//! lowering into an ingest dataflow.

use crate::catalog::{BoundTable, Error as CatalogError};
use crate::types::physical_arrow_type;
use arrow_array::{ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{Field, Schema, SchemaRef};
use dispatch::{
    ChannelInputSender, DataFlowDispatcher, RecordBatchOperatorSpec, Unary, UnaryError,
    UnaryFactory, UnaryResult, channel_input, stealable,
};
use std::fmt;
use std::sync::Arc;

/// `COPY <table> [(columns)] FROM STDIN WITH (FORMAT arrow)`: load rows
/// arriving over the client protocol into a table.
///
/// A statement, not a query: its rows arrive later over the connection's
/// copy-in sub-protocol, so a plan never compiles it. The engine reads it off
/// the plan ([`Plan::as_copy_from_stdin`](crate::Plan::as_copy_from_stdin))
/// and lowers it through [`compile_ingest`](Self::compile_ingest), which must
/// run inside the transaction the statement was planned in: the bound table
/// stages its writes there. Everything knowable without the rows is validated
/// at plan time; the rows themselves are checked per batch as they arrive
/// (width and casts).
#[derive(Debug)]
pub struct CopyFromStdin {
    /// The bound target table, resolved by DuckDB's binder like an INSERT's.
    pub table: Box<dyn BoundTable>,
    /// The explicit column list resolved to table column positions, empty
    /// when the statement targets every column.
    pub columns: Vec<usize>,
    /// The validated data format.
    pub format: CopyFormat,
}

/// The validated data format of a COPY FROM STDIN. Only Arrow IPC is
/// supported so far; the PostgreSQL text format (the protocol's default) and
/// everything else are rejected at plan time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyFormat {
    /// An Arrow IPC stream; batches conform to the table schema by position.
    ArrowIpc,
}

impl CopyFormat {
    /// Validate a statement's format name and raw option list (names as
    /// written, each with its bound constant values; a bare flag has none).
    pub fn resolve(
        format: Option<&str>,
        options: &[(String, Vec<String>)],
    ) -> Result<Self, String> {
        match format {
            Some("arrow") => {}
            None | Some("text") => {
                return Err(
                    "the COPY text format is not supported yet; use WITH (FORMAT arrow)"
                        .to_string(),
                );
            }
            Some(other) => {
                return Err(format!(
                    "COPY FORMAT {other} is not supported; only arrow is"
                ));
            }
        }
        if let Some((name, _)) = options.first() {
            return Err(format!(
                "COPY option \"{}\" is not valid for FORMAT arrow",
                name.to_lowercase()
            ));
        }
        Ok(Self::ArrowIpc)
    }
}

impl Clone for CopyFromStdin {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone_box(),
            columns: self.columns.clone(),
            format: self.format.clone(),
        }
    }
}

impl fmt::Display for CopyFromStdin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reference = self.table.table_reference();
        write!(
            f,
            "CopyFromStdin({}.{}.{}",
            reference.datastore, reference.schema, reference.table
        )?;
        if !self.columns.is_empty() {
            let columns = self.table.columns();
            let names: Vec<String> = self
                .columns
                .iter()
                .map(|&index| columns[index].name.clone())
                .collect();
            write!(f, " ({})", names.join(", "))?;
        }
        match &self.format {
            CopyFormat::ArrowIpc => write!(f, " format: arrow")?,
        }
        write!(f, ")")
    }
}

/// Unclaimed batches the frontend may queue ahead of the workers.
const CHUNK_QUEUE_CAPACITY: usize = 1048;

/// The statement's column list resolved against the table: the full output
/// schema in table-column order (built once, shared by every conformed
/// batch), and where each output column's data comes from in an incoming
/// batch (`None` for columns the statement does not fill, which become NULL).
#[derive(Debug)]
struct ColumnLayout {
    schema: SchemaRef,
    source_positions: Vec<Option<usize>>,
}

impl ColumnLayout {
    /// Columns each incoming batch must have: the length of the COPY column
    /// list (or the full table width without one). Exactly the table columns
    /// the statement feeds, so it is the count of filled source positions.
    fn incoming_columns(&self) -> usize {
        self.source_positions.iter().flatten().count()
    }
}

impl CopyFromStdin {
    /// Lower this statement into its ingest dataflow: a batch channel fanning
    /// out to per-worker conformance stages, feeding the bound table's insert
    /// sink (which stages into the transaction the statement was planned in).
    /// Returns the producer half and the spec ready to execute; its single
    /// output row is the insert count. `on_claim` fires as workers take
    /// batches, for producer backpressure.
    pub fn compile_ingest(
        &self,
        dispatcher: &DataFlowDispatcher,
        on_claim: Box<dyn Fn() + Send + Sync>,
    ) -> Result<(ChannelInputSender<RecordBatch>, RecordBatchOperatorSpec), CatalogError> {
        let CopyFormat::ArrowIpc = self.format;
        let layout = Arc::new(self.build_layout().map_err(|message| {
            CatalogError::Other(Box::<dyn std::error::Error + Send + Sync>::from(message))
        })?);
        let (sender, batches) =
            channel_input::<RecordBatch>(dispatcher, CHUNK_QUEUE_CAPACITY, on_claim);
        let worker_count = dispatcher.worker_count();
        let conformers: Vec<_> = (0..worker_count)
            .map(|_| ConformArrowBatch {
                layout: layout.clone(),
            })
            .collect();
        let conformed = batches
            .chain(
                stealable::<RecordBatch>(dispatcher.topology())
                    .into_iter()
                    .collect(),
                conformers,
            )
            .record_batches();
        let insert = self.table.compile_insert(conformed, dispatcher)?;
        Ok((sender, insert))
    }

    /// Resolve the column list (positions resolved and validated by the
    /// binder) against the table.
    fn build_layout(&self) -> Result<ColumnLayout, String> {
        let table_columns = self.table.columns();
        let fields: Vec<Field> = table_columns
            .iter()
            .map(|column| {
                Field::new(
                    column.name.clone(),
                    physical_arrow_type(&column.col_type),
                    true,
                )
            })
            .collect();
        let mut source_positions: Vec<Option<usize>> = vec![None; table_columns.len()];
        if self.columns.is_empty() {
            for (i, source) in source_positions.iter_mut().enumerate() {
                *source = Some(i);
            }
        } else {
            // The binder guarantees valid, distinct positions; check anyway so
            // a disagreement fails loudly instead of scrambling columns.
            for (position, &index) in self.columns.iter().enumerate() {
                let slot = source_positions
                    .get_mut(index)
                    .ok_or_else(|| format!("COPY column position {index} is out of range"))?;
                if slot.is_some() {
                    return Err(format!("COPY column position {index} listed twice"));
                }
                *slot = Some(position);
            }
        }
        Ok(ColumnLayout {
            schema: Arc::new(Schema::new(fields)),
            source_positions,
        })
    }
}

/// Cast one incoming column to its table field's physical type, with a loud
/// error naming the column when a value does not convert.
fn cast_to_field(array: &ArrayRef, field: &Field) -> Result<ArrayRef, String> {
    if field.data_type() == array.data_type() {
        return Ok(array.clone());
    }
    let options = arrow::compute::CastOptions {
        safe: false,
        format_options: Default::default(),
    };
    arrow::compute::cast_with_options(array, field.data_type(), &options)
        .map_err(|e| format!("column \"{}\": {e}", field.name()))
}

/// Conform one client-schema Arrow batch to the table's physical schema:
/// columns map to the statement's column list by position, cast to the
/// table's physical types, and unlisted table columns fill with NULL.
/// Variant columns are admitted as sent: clients are trusted to encode them
/// correctly, and a malformed document surfaces wherever it is first parsed.
fn conform_arrow_batch(batch: &RecordBatch, layout: &ColumnLayout) -> Result<RecordBatch, String> {
    let incoming_columns = layout.incoming_columns();
    if batch.num_columns() != incoming_columns {
        return Err(format!(
            "COPY arrow stream has {} columns, the statement targets {incoming_columns}",
            batch.num_columns(),
        ));
    }
    let arrays = layout
        .schema
        .fields()
        .iter()
        .zip(&layout.source_positions)
        .map(|(field, source)| match source {
            Some(index) => cast_to_field(batch.column(*index), field),
            None => Ok(new_null_array(field.data_type(), batch.num_rows())),
        })
        .collect::<Result<Vec<_>, String>>()?;
    RecordBatch::try_new(layout.schema.clone(), arrays)
        .map_err(|e| format!("assembling COPY batch: {e}"))
}

/// Worker-side transform: one decoded client batch in, one table-schema batch
/// out.
struct ConformArrowBatch {
    layout: Arc<ColumnLayout>,
}

impl Unary<RecordBatch, RecordBatch> for ConformArrowBatch {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn dispatch::Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let batch = conform_arrow_batch(&batch, &self.layout)
            .map_err(|message| UnaryError::Operator(message.into()))?;
        if batch.num_rows() > 0 {
            sender.send(batch)?;
        }
        Ok(())
    }
}

// Stateless per worker, so it serves as its own factory.
impl UnaryFactory<RecordBatch, RecordBatch> for ConformArrowBatch {
    type Unary = Self;

    fn build_unary(self) -> Self {
        self
    }
}
