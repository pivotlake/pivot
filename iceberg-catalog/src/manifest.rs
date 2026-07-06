//! The Avro half of a snapshot: the **manifest list** (one row per manifest
//! file) and the **manifests** themselves (one row per data file). Records are
//! walked field-by-name off [`apache_avro`]'s generic `Value`s - the files are
//! self-describing, and picking named fields tolerates the schema-evolution
//! differences between Iceberg format v1 and v2 (fields we need exist in both;
//! v2-only fields default where v1 omits them).

use apache_avro::Reader;
use apache_avro::types::Value;

use crate::{Error, Result};

/// A manifest-list entry: one manifest file of the snapshot.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ManifestFile {
    pub(crate) path: String,
    /// 0 = data manifest, 1 = delete manifest (v2; v1 has no field and is
    /// always data).
    pub(crate) content: i32,
}

/// A live data file of one manifest: everything a scan needs to read it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ManifestDataFile {
    pub(crate) file_path: String,
    pub(crate) file_size_in_bytes: i64,
}

/// Parse a manifest-list file into its manifest entries.
pub(crate) fn parse_manifest_list(bytes: &[u8]) -> Result<Vec<ManifestFile>> {
    read_records(bytes)?
        .into_iter()
        .map(|record| {
            let path = take_string(&record, "manifest_path")?;
            let content = take_int_or(&record, "content", 0)?;
            Ok(ManifestFile { path, content })
        })
        .collect()
}

/// Parse a (data) manifest into its **live** data files: entries whose status
/// is DELETED (removed by this snapshot) are dropped, EXISTING and ADDED are
/// kept. Errors on a non-Parquet data file or a row-level delete file, rather
/// than scanning something that would return wrong rows.
pub(crate) fn parse_manifest(bytes: &[u8]) -> Result<Vec<ManifestDataFile>> {
    const STATUS_DELETED: i32 = 2;
    const CONTENT_DATA: i32 = 0;

    let mut data_files = Vec::new();
    for record in read_records(bytes)? {
        if take_int_or(&record, "status", 0)? == STATUS_DELETED {
            continue;
        }
        let data_file = take_record(&record, "data_file")?;
        let file_path = take_string(data_file, "file_path")?;
        if take_int_or(data_file, "content", CONTENT_DATA)? != CONTENT_DATA {
            return Err(Error::Unsupported(format!(
                "`{file_path}` is a row-level delete file; iceberg deletes are not supported"
            )));
        }
        let file_format = take_string(data_file, "file_format")?;
        if !file_format.eq_ignore_ascii_case("parquet") {
            return Err(Error::Unsupported(format!(
                "data file `{file_path}` has format {file_format}; only PARQUET is supported"
            )));
        }
        data_files.push(ManifestDataFile {
            file_path,
            file_size_in_bytes: take_long(data_file, "file_size_in_bytes")?,
        });
    }
    Ok(data_files)
}

/// The fields of one top-level record, as `(name, value)` pairs.
type Fields = Vec<(String, Value)>;

/// Read an Avro object-container file into its records' field lists.
fn read_records(bytes: &[u8]) -> Result<Vec<Fields>> {
    Reader::new(bytes)?
        .map(|value| match value? {
            Value::Record(fields) => Ok(fields),
            other => Err(Error::Metadata(format!(
                "avro manifest row is not a record: {other:?}"
            ))),
        })
        .collect()
}

/// Field `name` of `fields`, with a union (nullable) wrapper peeled off.
/// `None` when the field is absent (a format-version difference).
fn find_field<'a>(fields: &'a Fields, name: &str) -> Option<&'a Value> {
    let (_, value) = fields.iter().find(|(field, _)| field == name)?;
    match value {
        Value::Union(_, inner) => Some(inner),
        other => Some(other),
    }
}

/// Field `name` of `fields`, which must be present.
fn require_field<'a>(fields: &'a Fields, name: &str) -> Result<&'a Value> {
    find_field(fields, name)
        .ok_or_else(|| Error::Metadata(format!("avro manifest record lacks field `{name}`")))
}

/// A nested record field (`data_file`), borrowed as its field list.
fn take_record<'a>(fields: &'a Fields, name: &str) -> Result<&'a Fields> {
    match require_field(fields, name)? {
        Value::Record(inner) => Ok(inner),
        other => Err(mistyped(name, "a record", other)),
    }
}

fn take_string(fields: &Fields, name: &str) -> Result<String> {
    match require_field(fields, name)? {
        Value::String(s) => Ok(s.clone()),
        other => Err(mistyped(name, "a string", other)),
    }
}

fn take_long(fields: &Fields, name: &str) -> Result<i64> {
    match require_field(fields, name)? {
        Value::Long(v) => Ok(*v),
        Value::Int(v) => Ok(i64::from(*v)),
        other => Err(mistyped(name, "an integer", other)),
    }
}

/// An integer field that later format versions added: absent means `default`.
fn take_int_or(fields: &Fields, name: &str, default: i32) -> Result<i32> {
    match find_field(fields, name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Int(v)) => Ok(*v),
        Some(Value::Long(v)) => Ok(*v as i32),
        Some(other) => Err(mistyped(name, "an integer", other)),
    }
}

fn mistyped(name: &str, expected: &str, got: &Value) -> Error {
    Error::Metadata(format!(
        "avro manifest field `{name}` is not {expected}: {got:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apache_avro::Schema;
    use apache_avro::Writer;

    /// Serialize `rows` as an Avro object-container file under `schema` - the
    /// fixture side of the parsers under test.
    fn write_avro(schema: &Schema, rows: Vec<Value>) -> Vec<u8> {
        let mut writer = Writer::new(schema, Vec::new());
        for row in rows {
            writer.append(row).unwrap();
        }
        writer.into_inner().unwrap()
    }

    #[test]
    fn parses_manifest_list_entries_with_and_without_content() {
        // The v2 manifest-list shape, trimmed to the fields the parser reads
        // plus one it ignores.
        let schema = Schema::parse_str(
            r#"{"type": "record", "name": "manifest_file", "fields": [
                {"name": "manifest_path", "type": "string"},
                {"name": "manifest_length", "type": "long"},
                {"name": "content", "type": "int"}
            ]}"#,
        )
        .unwrap();
        let row = |path: &str, content: i32| {
            Value::Record(vec![
                ("manifest_path".into(), Value::String(path.into())),
                ("manifest_length".into(), Value::Long(4096)),
                ("content".into(), Value::Int(content)),
            ])
        };
        let bytes = write_avro(
            &schema,
            vec![row("s3://b/m1.avro", 0), row("s3://b/m2.avro", 1)],
        );

        let entries = parse_manifest_list(&bytes).unwrap();

        assert_eq!(
            entries,
            vec![
                ManifestFile {
                    path: "s3://b/m1.avro".into(),
                    content: 0
                },
                ManifestFile {
                    path: "s3://b/m2.avro".into(),
                    content: 1
                },
            ]
        );
    }

    /// The v1 manifest-list shape has no `content` field; it defaults to data.
    #[test]
    fn a_v1_manifest_list_entry_defaults_to_data_content() {
        let schema = Schema::parse_str(
            r#"{"type": "record", "name": "manifest_file", "fields": [
                {"name": "manifest_path", "type": "string"}
            ]}"#,
        )
        .unwrap();
        let bytes = write_avro(
            &schema,
            vec![Value::Record(vec![(
                "manifest_path".into(),
                Value::String("s3://b/m.avro".into()),
            )])],
        );

        assert_eq!(parse_manifest_list(&bytes).unwrap()[0].content, 0);
    }

    fn manifest_entry_schema() -> Schema {
        Schema::parse_str(
            r#"{"type": "record", "name": "manifest_entry", "fields": [
                {"name": "status", "type": "int"},
                {"name": "data_file", "type": {"type": "record", "name": "r2", "fields": [
                    {"name": "content", "type": "int"},
                    {"name": "file_path", "type": "string"},
                    {"name": "file_format", "type": "string"},
                    {"name": "record_count", "type": "long"},
                    {"name": "file_size_in_bytes", "type": "long"}
                ]}}
            ]}"#,
        )
        .unwrap()
    }

    fn manifest_entry(status: i32, content: i32, path: &str, format: &str, size: i64) -> Value {
        Value::Record(vec![
            ("status".into(), Value::Int(status)),
            (
                "data_file".into(),
                Value::Record(vec![
                    ("content".into(), Value::Int(content)),
                    ("file_path".into(), Value::String(path.into())),
                    ("file_format".into(), Value::String(format.into())),
                    ("record_count".into(), Value::Long(3)),
                    ("file_size_in_bytes".into(), Value::Long(size)),
                ]),
            ),
        ])
    }

    #[test]
    fn keeps_live_entries_and_drops_deleted_ones() {
        let bytes = write_avro(
            &manifest_entry_schema(),
            vec![
                manifest_entry(1, 0, "s3://b/added.parquet", "PARQUET", 100),
                manifest_entry(0, 0, "s3://b/existing.parquet", "PARQUET", 200),
                manifest_entry(2, 0, "s3://b/deleted.parquet", "PARQUET", 300),
            ],
        );

        let files = parse_manifest(&bytes).unwrap();

        assert_eq!(
            files,
            vec![
                ManifestDataFile {
                    file_path: "s3://b/added.parquet".into(),
                    file_size_in_bytes: 100
                },
                ManifestDataFile {
                    file_path: "s3://b/existing.parquet".into(),
                    file_size_in_bytes: 200
                },
            ]
        );
    }

    #[test]
    fn rejects_a_delete_file_and_a_non_parquet_file() {
        let delete_file = write_avro(
            &manifest_entry_schema(),
            vec![manifest_entry(
                1,
                1,
                "s3://b/pos-deletes.parquet",
                "PARQUET",
                10,
            )],
        );
        let orc_file = write_avro(
            &manifest_entry_schema(),
            vec![manifest_entry(1, 0, "s3://b/data.orc", "ORC", 10)],
        );

        assert!(
            parse_manifest(&delete_file)
                .unwrap_err()
                .to_string()
                .contains("delete")
        );
        assert!(
            parse_manifest(&orc_file)
                .unwrap_err()
                .to_string()
                .contains("ORC")
        );
    }
}
