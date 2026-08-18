//! Decode the PostgreSQL COPY text format into Arrow batches.
//!
//! The format is line-oriented: one row per newline-terminated line, fields
//! separated by tabs, `\N` for NULL, and backslash escapes for control
//! characters and the separators themselves. Every column decodes to a string;
//! the ingest dataflow's conformance stage casts each one to its table
//! column's physical type, so value parsing (and its errors) live in one
//! place for every COPY format.
//!
//! Frames arrive split at arbitrary byte positions, so a partial trailing
//! line carries over between [`TextRowDecoder::decode`] calls. Escaped
//! newlines arrive as the two bytes `\n`, never as a raw newline, so a raw
//! newline always ends a row.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::builder::StringBuilder;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

/// Rows accumulated before a batch is emitted downstream.
const ROWS_PER_BATCH: usize = 8192;

/// The end-of-data marker line, `\.`, which optionally precedes the
/// protocol's own end of the copy stream.
const END_OF_DATA_MARKER: &[u8] = b"\\.";

pub(crate) struct TextRowDecoder {
    schema: SchemaRef,
    builders: Vec<StringBuilder>,
    buffered_rows: usize,
    /// A partial line carried between frames.
    carry: Vec<u8>,
    /// Set once the `\.` end-of-data marker arrives; further rows are an
    /// error.
    saw_end_marker: bool,
}

impl TextRowDecoder {
    pub(crate) fn new(column_count: usize) -> Self {
        let fields: Vec<Field> = (0..column_count)
            .map(|index| Field::new(format!("field_{index}"), DataType::Utf8, true))
            .collect();
        Self {
            schema: Arc::new(Schema::new(fields)),
            builders: (0..column_count).map(|_| StringBuilder::new()).collect(),
            buffered_rows: 0,
            carry: Vec::new(),
            saw_end_marker: false,
        }
    }

    /// Consume one protocol frame, appending every batch it completes to
    /// `batches`. A partial trailing line waits for the next frame.
    pub(crate) fn decode(
        &mut self,
        bytes: &[u8],
        batches: &mut Vec<RecordBatch>,
    ) -> Result<(), String> {
        let joined: Vec<u8>;
        let data: &[u8] = if self.carry.is_empty() {
            bytes
        } else {
            let mut carried = std::mem::take(&mut self.carry);
            carried.extend_from_slice(bytes);
            joined = carried;
            &joined
        };
        let mut line_start = 0;
        while let Some(offset) = data[line_start..].iter().position(|&byte| byte == b'\n') {
            self.consume_line(&data[line_start..line_start + offset])?;
            line_start += offset + 1;
            if self.buffered_rows >= ROWS_PER_BATCH {
                batches.push(self.take_batch());
            }
        }
        self.carry = data[line_start..].to_vec();
        Ok(())
    }

    /// Report the end of the stream and flush the rows still buffered. The
    /// protocol ends the stream; a final line without its newline is still a
    /// row, matching a server reading to end of input.
    pub(crate) fn finish(&mut self) -> Result<Option<RecordBatch>, String> {
        if !self.carry.is_empty() {
            let line = std::mem::take(&mut self.carry);
            self.consume_line(&line)?;
        }
        if self.buffered_rows == 0 {
            return Ok(None);
        }
        Ok(Some(self.take_batch()))
    }

    fn take_batch(&mut self) -> RecordBatch {
        let columns = self
            .builders
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as arrow_array::ArrayRef)
            .collect();
        self.buffered_rows = 0;
        RecordBatch::try_new(self.schema.clone(), columns)
            .expect("text columns share one row count")
    }

    /// Parse one line into a row (or recognize the end-of-data marker) and
    /// append its fields to the builders.
    fn consume_line(&mut self, line: &[u8]) -> Result<(), String> {
        if self.saw_end_marker {
            return Err("data after the end-of-copy marker".to_string());
        }
        if line == END_OF_DATA_MARKER {
            self.saw_end_marker = true;
            return Ok(());
        }
        let mut fields_appended = 0;
        for raw_field in split_fields(line)? {
            let raw_field = raw_field?;
            if fields_appended == self.builders.len() {
                return Err(format!(
                    "row has more than {} fields, the statement's column count",
                    self.builders.len()
                ));
            }
            let builder = &mut self.builders[fields_appended];
            if raw_field == b"\\N" {
                builder.append_null();
            } else {
                builder.append_value(unescape_field(raw_field)?);
            }
            fields_appended += 1;
        }
        if fields_appended != self.builders.len() {
            return Err(format!(
                "row has {fields_appended} fields, the statement targets {}",
                self.builders.len()
            ));
        }
        self.buffered_rows += 1;
        Ok(())
    }
}

/// Split a line into raw (still escaped) fields at unescaped tabs. A
/// backslash consumes the following byte, so an escaped tab stays inside its
/// field. A raw carriage return is rejected the way a PostgreSQL server
/// rejects it: data must escape it as `\r`.
fn split_fields(line: &[u8]) -> Result<impl Iterator<Item = Result<&[u8], String>>, String> {
    if line.contains(&b'\r') {
        return Err("literal carriage return found in data; escape it as \\r".to_string());
    }
    let mut rest = Some(line);
    Ok(std::iter::from_fn(move || {
        let line = rest?;
        let mut position = 0;
        while position < line.len() {
            match line[position] {
                b'\t' => {
                    rest = Some(&line[position + 1..]);
                    return Some(Ok(&line[..position]));
                }
                b'\\' if position + 1 == line.len() => {
                    rest = None;
                    return Some(Err("field ends with an unescaped backslash".to_string()));
                }
                b'\\' => position += 2,
                _ => position += 1,
            }
        }
        rest = None;
        Some(Ok(line))
    }))
}

/// Resolve a raw field's backslash escapes into the value's bytes and check
/// they form UTF-8. `split_fields` already guaranteed no backslash dangles at
/// the end.
fn unescape_field(raw: &[u8]) -> Result<String, String> {
    let mut value = Vec::with_capacity(raw.len());
    let mut position = 0;
    while position < raw.len() {
        let byte = raw[position];
        if byte != b'\\' {
            value.push(byte);
            position += 1;
            continue;
        }
        position += 1;
        let escaped = raw[position];
        position += 1;
        match escaped {
            b'b' => value.push(0x08),
            b'f' => value.push(0x0C),
            b'n' => value.push(b'\n'),
            b'r' => value.push(b'\r'),
            b't' => value.push(b'\t'),
            b'v' => value.push(0x0B),
            b'0'..=b'7' => {
                let mut octal = u32::from(escaped - b'0');
                for _ in 0..2 {
                    match raw.get(position) {
                        Some(&digit @ b'0'..=b'7') => {
                            octal = octal * 8 + u32::from(digit - b'0');
                            position += 1;
                        }
                        _ => break,
                    }
                }
                value.push(octal as u8);
            }
            b'x' => {
                let mut hex: Option<u32> = None;
                for _ in 0..2 {
                    match raw.get(position).copied().map(hex_digit_value) {
                        Some(Some(digit)) => {
                            hex = Some(hex.unwrap_or(0) * 16 + digit);
                            position += 1;
                        }
                        _ => break,
                    }
                }
                match hex {
                    Some(hex) => value.push(hex as u8),
                    // `\x` with no hex digit is a literal `x`, as a server
                    // reads it.
                    None => value.push(b'x'),
                }
            }
            other => value.push(other),
        }
    }
    String::from_utf8(value).map_err(|_| "field is not valid UTF-8 after unescaping".to_string())
}

fn hex_digit_value(byte: u8) -> Option<u32> {
    char::from(byte).to_digit(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, StringArray};

    /// Decode `data` in one frame plus a finish, returning every row as
    /// optional strings.
    fn decode_rows(column_count: usize, data: &[u8]) -> Result<Vec<Vec<Option<String>>>, String> {
        let mut decoder = TextRowDecoder::new(column_count);
        let mut batches = Vec::new();
        decoder.decode(data, &mut batches)?;
        batches.extend(decoder.finish()?);
        let mut rows = Vec::new();
        for batch in batches {
            let columns: Vec<&StringArray> = (0..batch.num_columns())
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
    fn decodes_rows_nulls_and_escapes() {
        let data = b"1\talice\n2\t\\N\n3\ttab\\there \\\\ \\n\\r\n";

        let rows = decode_rows(2, data).unwrap();

        assert_eq!(
            rows,
            vec![
                vec![Some("1".into()), Some("alice".into())],
                vec![Some("2".into()), None],
                vec![Some("3".into()), Some("tab\there \\ \n\r".into())],
            ]
        );
    }

    #[test]
    fn decodes_octal_and_hex_escapes() {
        let rows = decode_rows(1, b"\\101\\x41\\x4a\n").unwrap();

        assert_eq!(rows, vec![vec![Some("AAJ".into())]]);
    }

    #[test]
    fn carries_a_partial_line_across_frames() {
        let mut decoder = TextRowDecoder::new(2);
        let mut batches = Vec::new();

        decoder.decode(b"1\tali", &mut batches).unwrap();
        decoder.decode(b"ce\n2\tbob\n", &mut batches).unwrap();
        batches.extend(decoder.finish().unwrap());

        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 2);
    }

    #[test]
    fn finishes_a_final_line_without_a_newline() {
        let rows = decode_rows(1, b"last").unwrap();

        assert_eq!(rows, vec![vec![Some("last".into())]]);
    }

    #[test]
    fn end_marker_stops_the_stream() {
        let rows = decode_rows(1, b"1\n\\.\n").unwrap();

        assert_eq!(rows, vec![vec![Some("1".into())]]);
    }

    #[test]
    fn data_after_the_end_marker_is_an_error() {
        let error = decode_rows(1, b"1\n\\.\n2\n").unwrap_err();

        assert!(error.contains("end-of-copy marker"), "{error}");
    }

    #[test]
    fn field_count_mismatches_are_errors() {
        assert!(
            decode_rows(2, b"only-one\n")
                .unwrap_err()
                .contains("1 fields")
        );
        assert!(
            decode_rows(1, b"a\tb\n")
                .unwrap_err()
                .contains("more than 1")
        );
    }

    #[test]
    fn a_literal_carriage_return_is_an_error() {
        let error = decode_rows(1, b"bad\r\n").unwrap_err();

        assert!(error.contains("carriage return"), "{error}");
    }

    #[test]
    fn backslash_n_stays_distinguishable_from_null() {
        let rows = decode_rows(1, b"\\\\N\n\\N\n").unwrap();

        assert_eq!(rows, vec![vec![Some("\\N".into())], vec![None]]);
    }
}
