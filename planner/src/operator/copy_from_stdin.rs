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

/// `COPY <table> [(columns)] FROM STDIN WITH (FORMAT <format>)`: load rows
/// arriving over the client protocol into a table as Arrow IPC or CSV.
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

/// The validated data format of a COPY FROM STDIN. The PostgreSQL text format
/// (the protocol's default) and formats other than Arrow IPC and CSV are
/// rejected at plan time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyFormat {
    /// An Arrow IPC stream; batches conform to the table schema by position.
    ArrowIpc,
    /// PostgreSQL CSV data and its validated parsing options.
    Csv(CopyCsvOptions),
}

/// The supported options for `COPY ... WITH (FORMAT csv)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyCsvOptions {
    /// The single-byte field separator (comma by default).
    pub delimiter: u8,
    /// Whether to discard the first record as a header.
    pub header: bool,
    /// The single-byte character surrounding quoted fields.
    pub quote: u8,
    /// The single-byte character escaping quote and escape within a quote.
    pub escape: u8,
    /// An unquoted field exactly equal to this string is NULL.
    pub null: String,
}

impl Default for CopyCsvOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            header: false,
            quote: b'"',
            escape: b'"',
            null: String::new(),
        }
    }
}

impl CopyFormat {
    /// Validate a statement's format name and raw option list (names as
    /// written, each with its bound constant values; a bare flag has none).
    pub fn resolve(
        format: Option<&str>,
        options: &[(String, Vec<String>)],
    ) -> Result<Self, String> {
        let format = match format {
            Some("arrow") => Self::ArrowIpc,
            Some("csv") => Self::Csv(CopyCsvOptions::resolve(options)?),
            None | Some("text") => {
                return Err(
                    "the COPY text format is not supported yet; use WITH (FORMAT csv) or WITH (FORMAT arrow)"
                        .to_string(),
                );
            }
            Some(other) => {
                return Err(format!(
                    "COPY FORMAT {other} is not supported; only csv and arrow are"
                ));
            }
        };
        if matches!(format, Self::ArrowIpc)
            && let Some((name, _)) = options.first()
        {
            return Err(format!(
                "COPY option \"{}\" is not valid for FORMAT arrow",
                name.to_lowercase()
            ));
        }
        Ok(format)
    }

    /// The SQL name of this format.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ArrowIpc => "arrow",
            Self::Csv(_) => "csv",
        }
    }
}

impl CopyCsvOptions {
    fn resolve(options: &[(String, Vec<String>)]) -> Result<Self, String> {
        let mut csv = Self::default();
        for (name, values) in options {
            let name = name.to_ascii_lowercase();
            match name.as_str() {
                "delimiter" => csv.delimiter = csv_option_byte(&name, values)?,
                "header" => csv.header = csv_header_option(values)?,
                "quote" => csv.quote = csv_option_byte(&name, values)?,
                "escape" => csv.escape = csv_option_byte(&name, values)?,
                "null" => csv.null = csv_option_value(&name, values)?.to_string(),
                _ => {
                    return Err(format!(
                        "COPY option \"{name}\" is not valid for FORMAT csv"
                    ));
                }
            }
        }

        for (name, byte) in [
            ("delimiter", csv.delimiter),
            ("quote", csv.quote),
            ("escape", csv.escape),
        ] {
            if matches!(byte, b'\r' | b'\n') {
                return Err(format!("COPY CSV {name} cannot be a newline"));
            }
        }
        if csv.delimiter == csv.quote {
            return Err("COPY CSV delimiter and quote must be different".to_string());
        }
        if csv.null.contains('\r') || csv.null.contains('\n') {
            return Err("COPY CSV null string cannot contain a newline".to_string());
        }
        if csv.null.as_bytes().contains(&csv.delimiter) {
            return Err("COPY CSV null string cannot contain the delimiter".to_string());
        }
        if csv.null.as_bytes().contains(&csv.quote) {
            return Err("COPY CSV null string cannot contain the quote character".to_string());
        }
        Ok(csv)
    }
}

fn csv_option_value<'a>(name: &str, values: &'a [String]) -> Result<&'a str, String> {
    let [value] = values else {
        return Err(format!("COPY CSV option \"{name}\" requires one value"));
    };
    Ok(value)
}

fn csv_option_byte(name: &str, values: &[String]) -> Result<u8, String> {
    let value = csv_option_value(name, values)?;
    let [byte] = value.as_bytes() else {
        return Err(format!(
            "COPY CSV option \"{name}\" must be a single one-byte character"
        ));
    };
    Ok(*byte)
}

fn csv_header_option(values: &[String]) -> Result<bool, String> {
    let value = match values {
        [] => return Ok(true),
        [value] => value.to_ascii_lowercase(),
        _ => return Err("COPY CSV option \"header\" accepts at most one value".to_string()),
    };
    match value.as_str() {
        "true" | "t" | "1" | "on" | "yes" => Ok(true),
        "false" | "f" | "0" | "off" | "no" => Ok(false),
        _ => Err(format!(
            "COPY CSV option \"header\" expects a boolean, got {value:?}"
        )),
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
        write!(f, " format: {})", self.format.name())
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
        let layout = Arc::new(self.build_layout().map_err(|message| {
            CatalogError::Other(Box::<dyn std::error::Error + Send + Sync>::from(message))
        })?);
        let (sender, batches) =
            channel_input::<RecordBatch>(dispatcher, CHUNK_QUEUE_CAPACITY, on_claim);
        let worker_count = dispatcher.worker_count();
        let conformers: Vec<_> = (0..worker_count)
            .map(|_| ConformCopyBatch {
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

    /// Columns each incoming row or batch must carry: the explicit COPY
    /// column list's length, or the full table width without one.
    pub fn incoming_column_count(&self) -> usize {
        if self.columns.is_empty() {
            self.table.columns().len()
        } else {
            self.columns.len()
        }
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
    if *field.data_type() == physical_arrow_type(&crate::types::Type::Variant)
        && matches!(
            array.data_type(),
            arrow_schema::DataType::Utf8
                | arrow_schema::DataType::LargeUtf8
                | arrow_schema::DataType::Utf8View
        )
    {
        return crate::expression::json_to_canonical_variant(array)
            .map_err(|e| format!("column \"{}\": {e}", field.name()));
    }
    let options = arrow::compute::CastOptions {
        safe: false,
        format_options: Default::default(),
    };
    arrow::compute::cast_with_options(array, field.data_type(), &options)
        .map_err(|e| format!("column \"{}\": {e}", field.name()))
}

/// Conform one decoded client batch to the table's physical schema:
/// columns map to the statement's column list by position, cast to the
/// table's physical types, and unlisted table columns fill with NULL.
/// Variant structs from Arrow IPC are admitted as sent; string values from
/// CSV are parsed as JSON documents.
fn conform_copy_batch(batch: &RecordBatch, layout: &ColumnLayout) -> Result<RecordBatch, String> {
    let incoming_columns = layout.incoming_columns();
    if batch.num_columns() != incoming_columns {
        return Err(format!(
            "COPY input has {} columns, the statement targets {incoming_columns}",
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
struct ConformCopyBatch {
    layout: Arc<ColumnLayout>,
}

impl Unary<RecordBatch, RecordBatch> for ConformCopyBatch {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn dispatch::Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let batch = conform_copy_batch(&batch, &self.layout)
            .map_err(|message| UnaryError::Operator(message.into()))?;
        if batch.num_rows() > 0 {
            sender.send(batch)?;
        }
        Ok(())
    }
}

// Stateless per worker, so it serves as its own factory.
impl UnaryFactory<RecordBatch, RecordBatch> for ConformCopyBatch {
    type Unary = Self;

    fn build_unary(self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_csv_defaults_and_options() {
        assert_eq!(
            CopyFormat::resolve(Some("csv"), &[]).unwrap(),
            CopyFormat::Csv(CopyCsvOptions::default())
        );

        let format = CopyFormat::resolve(
            Some("csv"),
            &[
                ("DELIMITER".into(), vec!["|".into()]),
                ("escape".into(), vec!["\\".into()]),
                ("header".into(), Vec::new()),
                ("null".into(), vec!["NULL".into()]),
                ("quote".into(), vec!["'".into()]),
            ],
        )
        .unwrap();
        assert_eq!(
            format,
            CopyFormat::Csv(CopyCsvOptions {
                delimiter: b'|',
                header: true,
                quote: b'\'',
                escape: b'\\',
                null: "NULL".into(),
            })
        );
    }

    #[test]
    fn validates_csv_options() {
        let error =
            CopyFormat::resolve(Some("csv"), &[("compression".into(), vec!["gzip".into()])])
                .unwrap_err();
        assert!(error.contains("not valid for FORMAT csv"), "{error}");

        let error = CopyFormat::resolve(Some("csv"), &[("delimiter".into(), vec!["||".into()])])
            .unwrap_err();
        assert!(error.contains("single one-byte character"), "{error}");

        let error =
            CopyFormat::resolve(Some("csv"), &[("header".into(), vec!["sometimes".into()])])
                .unwrap_err();
        assert!(error.contains("expects a boolean"), "{error}");
    }
}
