//! The bucket listing both remote backends speak: S3's ListObjectsV2 and the
//! Google Cloud Storage XML API answer with the same `ListBucketResult`
//! document, so one parser serves both.

use super::{FileRef, ListedObject, ObjectPath, Result, StoreError, key_name};

/// A `ListBucketResult` document (only the object entries are needed).
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListBucketResult {
    #[serde(default, rename = "Contents")]
    contents: Vec<Contents>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Contents {
    key: String,
    #[serde(default)]
    size: u64,
    /// ISO-8601 `LastModified` (e.g. `2009-10-12T17:50:30.000Z`), parsed by
    /// [`parse_iso8601_millis`] for vacuum's orphan sweep.
    #[serde(default)]
    last_modified: String,
}

/// Parse a listing response body into the objects it names. Each object keeps
/// only its name within the listed prefix, as the store contract's `list`
/// returns relative keys.
pub(super) fn parse_listing(body: &str) -> Result<Vec<ListedObject>> {
    let parsed: ListBucketResult =
        quick_xml::de::from_str(body).map_err(|e| StoreError::Http(format!("LIST parse: {e}")))?;

    parsed
        .contents
        .into_iter()
        .map(|entry| {
            let modified_unix_ms = parse_iso8601_millis(&entry.last_modified).ok_or_else(|| {
                StoreError::Http(format!(
                    "LIST object `{}` has an unparseable LastModified `{}`",
                    entry.key, entry.last_modified
                ))
            })?;
            Ok(ListedObject {
                file: FileRef {
                    path: ObjectPath::new(key_name(&entry.key)),
                    size: entry.size,
                },
                modified_unix_ms,
            })
        })
        .collect()
}

/// Parse a `LastModified` timestamp (RFC 3339, always UTC) into Unix
/// milliseconds. Returns `None` on any malformed field, which the caller turns
/// into a listing error rather than a silently-wrong (too-old) timestamp.
fn parse_iso8601_millis(s: &str) -> Option<u64> {
    let nanos = arrow_cast::parse::string_to_timestamp_nanos(s).ok()?;
    u64::try_from(nanos / 1_000_000).ok()
}

/// Percent-encode an object key for a query-string value per RFC 3986
/// (unreserved chars pass through; `/` is encoded since it's a query value).
pub(super) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_keeps_object_names_sizes_and_modification_times() {
        // Each object carries its byte size, which the remote reader needs to
        // locate a Parquet footer, and its modification time, which vacuum needs
        // to age out orphans. The key is reduced to its name within the prefix.
        let xml = r#"<?xml version="1.0"?>
            <ListBucketResult>
              <Contents>
                <Key>db/events/a.parquet</Key>
                <Size>123</Size>
                <LastModified>2009-10-12T17:50:30.000Z</LastModified>
              </Contents>
              <Contents>
                <Key>db/events/b.parquet</Key>
                <Size>4096</Size>
                <LastModified>2009-10-12T17:50:31.500Z</LastModified>
              </Contents>
            </ListBucketResult>"#;

        let objects = parse_listing(xml).unwrap();

        let described: Vec<_> = objects
            .iter()
            .map(|o| (o.file.path.as_str(), o.file.size, o.modified_unix_ms))
            .collect();
        assert_eq!(
            described,
            vec![
                ("a.parquet", 123, 1_255_369_830_000),
                ("b.parquet", 4096, 1_255_369_831_500),
            ]
        );
    }

    #[test]
    fn an_unparseable_modification_time_fails_the_listing() {
        let xml = r#"<?xml version="1.0"?>
            <ListBucketResult>
              <Contents><Key>a.parquet</Key><Size>1</Size><LastModified>soon</LastModified></Contents>
            </ListBucketResult>"#;

        let error = parse_listing(xml).unwrap_err();

        assert!(error.to_string().contains("unparseable LastModified"));
    }

    #[test]
    fn query_values_are_percent_encoded() {
        assert_eq!(percent_encode("db/events"), "db%2Fevents");
        assert_eq!(percent_encode("a-b_c.d~e"), "a-b_c.d~e");
    }
}
