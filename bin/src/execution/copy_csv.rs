//! Streaming PostgreSQL CSV decoding for `COPY ... FROM STDIN`.
//!
//! Protocol frames may split anywhere, including in a quoted field or a UTF-8
//! character. This decoder therefore keeps parser state and field bytes across
//! calls, and emits string-typed Arrow batches for the shared COPY conformance
//! stage to cast to the table schema.

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use planner::CopyCsvOptions;

const ROWS_PER_BATCH: usize = dispatch::RECORD_BATCH_SIZE;

/// PostgreSQL's optional in-band end-of-copy marker. Clients normally turn it
/// into `CopyDone`, but accepting it also lets dump-style streams work.
const END_OF_DATA_MARKER: &[u8] = b"\\.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    FieldStart,
    Unquoted,
    Quoted,
    /// A quote may have closed the field, or may be the first half of a
    /// doubled quote when quote and escape are the same (the default).
    AfterQuote,
    /// A distinct escape character was seen inside a quoted field.
    Escaped,
}

#[derive(Debug)]
struct RawField {
    bytes: Vec<u8>,
    quoted: bool,
}

pub(super) struct CsvRowDecoder {
    options: CopyCsvOptions,
    schema: SchemaRef,
    builders: Vec<StringBuilder>,
    buffered_rows: usize,
    current_field: Vec<u8>,
    field_quoted: bool,
    current_row: Vec<RawField>,
    state: ParseState,
    /// A CR outside quotes already ended a record; swallow an immediately
    /// following LF so CRLF is one terminator even across protocol frames.
    swallow_lf: bool,
    header_consumed: bool,
    record_number: usize,
    saw_end_marker: bool,
}

impl CsvRowDecoder {
    pub(super) fn new(column_count: usize, options: CopyCsvOptions) -> Self {
        let fields: Vec<Field> = (0..column_count)
            .map(|index| Field::new(format!("field_{index}"), DataType::Utf8, true))
            .collect();
        Self {
            options,
            schema: Arc::new(Schema::new(fields)),
            builders: (0..column_count).map(|_| StringBuilder::new()).collect(),
            buffered_rows: 0,
            current_field: Vec::new(),
            field_quoted: false,
            current_row: Vec::with_capacity(column_count),
            state: ParseState::FieldStart,
            swallow_lf: false,
            header_consumed: false,
            record_number: 0,
            saw_end_marker: false,
        }
    }

    /// Consume one pgwire `CopyData` frame and return every full batch it
    /// completes. Partial fields and records remain buffered for the next
    /// frame.
    pub(super) fn decode(&mut self, bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
        let mut batches = Vec::new();
        for &byte in bytes {
            if self.swallow_lf {
                self.swallow_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if self.saw_end_marker {
                return Err("data after the end-of-copy marker".to_string());
            }
            self.consume_byte(byte, &mut batches)?;
        }
        Ok(batches)
    }

    /// Finish a stream, validating any partial record and returning the last
    /// non-empty batch.
    pub(super) fn finish(&mut self) -> Result<Option<RecordBatch>, String> {
        if !self.saw_end_marker {
            match self.state {
                ParseState::Quoted => {
                    return Err(format!(
                        "record {} has an unterminated quoted field",
                        self.record_number + 1
                    ));
                }
                ParseState::Escaped => {
                    return Err(format!(
                        "record {} ends with an incomplete escape",
                        self.record_number + 1
                    ));
                }
                ParseState::AfterQuote | ParseState::Unquoted => {
                    self.finish_field();
                    self.finish_record()?;
                }
                ParseState::FieldStart if !self.current_row.is_empty() => {
                    // A delimiter was the stream's final byte.
                    self.finish_field();
                    self.finish_record()?;
                }
                ParseState::FieldStart => {}
            }
        }
        Ok((self.buffered_rows != 0).then(|| self.take_batch()))
    }

    fn consume_byte(&mut self, byte: u8, batches: &mut Vec<RecordBatch>) -> Result<(), String> {
        match self.state {
            ParseState::FieldStart => match byte {
                byte if byte == self.options.delimiter => self.finish_field(),
                byte if byte == self.options.quote => {
                    self.field_quoted = true;
                    self.state = ParseState::Quoted;
                }
                b'\n' => {
                    self.finish_field();
                    self.finish_record()?;
                }
                b'\r' => {
                    self.finish_field();
                    self.finish_record()?;
                    self.swallow_lf = true;
                }
                byte => {
                    self.current_field.push(byte);
                    self.state = ParseState::Unquoted;
                }
            },
            ParseState::Unquoted => match byte {
                byte if byte == self.options.delimiter => {
                    self.finish_field();
                    self.state = ParseState::FieldStart;
                }
                b'\n' => {
                    self.finish_field();
                    self.finish_record()?;
                }
                b'\r' => {
                    self.finish_field();
                    self.finish_record()?;
                    self.swallow_lf = true;
                }
                byte if byte == self.options.quote => {
                    return Err(format!(
                        "record {} has a quote in an unquoted field",
                        self.record_number + 1
                    ));
                }
                byte => self.current_field.push(byte),
            },
            ParseState::Quoted => {
                if byte == self.options.escape && self.options.escape != self.options.quote {
                    self.state = ParseState::Escaped;
                } else if byte == self.options.quote {
                    self.state = ParseState::AfterQuote;
                } else {
                    self.current_field.push(byte);
                }
            }
            ParseState::Escaped => {
                if byte != self.options.quote && byte != self.options.escape {
                    return Err(format!(
                        "record {} uses the CSV escape before an invalid character",
                        self.record_number + 1
                    ));
                }
                self.current_field.push(byte);
                self.state = ParseState::Quoted;
            }
            ParseState::AfterQuote => {
                if self.options.escape == self.options.quote && byte == self.options.quote {
                    self.current_field.push(byte);
                    self.state = ParseState::Quoted;
                } else if byte == self.options.delimiter {
                    self.finish_field();
                    self.state = ParseState::FieldStart;
                } else if byte == b'\n' {
                    self.finish_field();
                    self.finish_record()?;
                } else if byte == b'\r' {
                    self.finish_field();
                    self.finish_record()?;
                    self.swallow_lf = true;
                } else {
                    return Err(format!(
                        "record {} has data after a closing quote",
                        self.record_number + 1
                    ));
                }
            }
        }

        if self.buffered_rows >= ROWS_PER_BATCH {
            batches.push(self.take_batch());
        }
        Ok(())
    }

    fn finish_field(&mut self) {
        self.current_row.push(RawField {
            bytes: std::mem::take(&mut self.current_field),
            quoted: std::mem::take(&mut self.field_quoted),
        });
    }

    fn finish_record(&mut self) -> Result<(), String> {
        self.state = ParseState::FieldStart;
        self.record_number += 1;

        if self.current_row.len() == 1
            && !self.current_row[0].quoted
            && self.current_row[0].bytes == END_OF_DATA_MARKER
        {
            self.current_row.clear();
            self.saw_end_marker = true;
            return Ok(());
        }

        if self.current_row.len() != self.builders.len() {
            let actual = self.current_row.len();
            self.current_row.clear();
            return Err(format!(
                "record {} has {actual} fields, the statement targets {}",
                self.record_number,
                self.builders.len()
            ));
        }

        let fields = std::mem::take(&mut self.current_row);
        if self.options.header && !self.header_consumed {
            for field in fields {
                std::str::from_utf8(&field.bytes).map_err(|_| {
                    format!(
                        "CSV header in record {} is not valid UTF-8",
                        self.record_number
                    )
                })?;
            }
            self.header_consumed = true;
            return Ok(());
        }
        self.header_consumed = true;

        for (builder, field) in self.builders.iter_mut().zip(fields) {
            let value = String::from_utf8(field.bytes).map_err(|_| {
                format!(
                    "field in CSV record {} is not valid UTF-8",
                    self.record_number
                )
            })?;
            if !field.quoted && value == self.options.null {
                builder.append_null();
            } else {
                builder.append_value(value);
            }
        }
        self.buffered_rows += 1;
        Ok(())
    }

    fn take_batch(&mut self) -> RecordBatch {
        let columns: Vec<ArrayRef> = self
            .builders
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as ArrayRef)
            .collect();
        self.buffered_rows = 0;
        RecordBatch::try_new(self.schema.clone(), columns)
            .expect("CSV columns are appended once per row")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_array::cast::AsArray;

    fn decode_rows(
        column_count: usize,
        options: CopyCsvOptions,
        frames: &[&[u8]],
    ) -> Result<Vec<Vec<Option<String>>>, String> {
        let mut decoder = CsvRowDecoder::new(column_count, options);
        let mut batches = Vec::new();
        for frame in frames {
            batches.extend(decoder.decode(frame)?);
        }
        batches.extend(decoder.finish()?);

        let mut rows = Vec::new();
        for batch in batches {
            let columns: Vec<_> = (0..batch.num_columns())
                .map(|index| batch.column(index).as_string::<i32>())
                .collect();
            for row in 0..batch.num_rows() {
                rows.push(
                    columns
                        .iter()
                        .map(|column| column.is_valid(row).then(|| column.value(row).to_string()))
                        .collect(),
                );
            }
        }
        Ok(rows)
    }

    #[test]
    fn decodes_quotes_newlines_nulls_and_crlf() {
        let rows = decode_rows(
            3,
            CopyCsvOptions::default(),
            &[b"1,\"hello, \"\"csv\"\"\",\r\n2,\"two\nlines\",\"\"\r\n"],
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                vec![Some("1".into()), Some("hello, \"csv\"".into()), None],
                vec![Some("2".into()), Some("two\nlines".into()), Some("".into())],
            ]
        );
    }

    #[test]
    fn carries_every_parser_state_across_frames() {
        let data = b"1,\"alpha,\"\"beta\"\"\"\n2,\"gamma\"\n";
        let frames: Vec<&[u8]> = data.iter().map(std::slice::from_ref).collect();

        let rows = decode_rows(2, CopyCsvOptions::default(), &frames).unwrap();

        assert_eq!(
            rows,
            vec![
                vec![Some("1".into()), Some("alpha,\"beta\"".into())],
                vec![Some("2".into()), Some("gamma".into())],
            ]
        );
    }

    #[test]
    fn applies_header_custom_characters_and_null_string() {
        let options = CopyCsvOptions {
            delimiter: b'|',
            header: true,
            quote: b'\'',
            escape: b'\\',
            null: "NULL".into(),
        };

        let rows = decode_rows(
            2,
            options,
            &[b"id|name\r", b"\n1|'a\\'b'\n2|NULL\n3|'NULL'"],
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                vec![Some("1".into()), Some("a'b".into())],
                vec![Some("2".into()), None],
                vec![Some("3".into()), Some("NULL".into())],
            ]
        );
    }

    #[test]
    fn rejects_bad_width_and_unterminated_quotes() {
        let width = decode_rows(2, CopyCsvOptions::default(), &[b"1,2,3\n"]).unwrap_err();
        assert!(width.contains("3 fields"), "{width}");

        let quote = decode_rows(1, CopyCsvOptions::default(), &[b"\"unfinished"]).unwrap_err();
        assert!(quote.contains("unterminated quoted field"), "{quote}");
    }

    #[test]
    fn accepts_end_marker_but_not_later_data() {
        let rows = decode_rows(1, CopyCsvOptions::default(), &[b"1\n\\.\n"]).unwrap();
        assert_eq!(rows, vec![vec![Some("1".into())]]);

        let error = decode_rows(1, CopyCsvOptions::default(), &[b"\\.\n2\n"]).unwrap_err();
        assert!(error.contains("end-of-copy marker"), "{error}");
    }
}
