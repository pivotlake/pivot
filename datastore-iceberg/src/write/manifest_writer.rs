//! Writing a commit's Avro manifest and manifest list.
//!
//! The spec crate's writers want an `OutputFile`; they are given one in an
//! in-memory file IO, and the bytes are read back out and stored through the
//! table's own store, so a manifest lands exactly the way a data file does
//! and through the same credentials.

use iceberg::io::FileIO;
use iceberg::spec::{
    DataFile, FormatVersion, ManifestFile, ManifestListWriter, ManifestWriterBuilder,
    PartitionSpecRef, SchemaRef,
};

use crate::rest_catalog::block_on;
use crate::{Error, Result};

/// Write the data manifest at `uri` adding `files` in snapshot `snapshot_id`,
/// in the table's `format_version`. The entries leave their sequence numbers
/// for the manifest list to assign.
pub(crate) fn write_data_manifest(
    uri: &str,
    format_version: FormatVersion,
    snapshot_id: i64,
    schema: &SchemaRef,
    partition_spec: &PartitionSpecRef,
    files: Vec<DataFile>,
) -> Result<(ManifestFile, Vec<u8>)> {
    let file_io = FileIO::new_with_memory();
    let output = file_io.new_output(uri).map_err(Error::metadata)?;
    let builder = ManifestWriterBuilder::new(
        output,
        Some(snapshot_id),
        schema.clone(),
        partition_spec.as_ref().clone(),
    );
    let mut writer = match format_version {
        FormatVersion::V3 => builder.build_v3_data(),
        _ => builder.build_v2_data(),
    };
    for file in files {
        writer.add_file(file, -1).map_err(Error::metadata)?;
    }
    let manifest = block_on(writer.write_manifest_file()).map_err(Error::metadata)?;
    let bytes = read_back(&file_io, uri)?;
    Ok((manifest, bytes))
}

/// A manifest list a commit wrote, and the row id the table continues from
/// once it is current; `None` for a table without row lineage.
pub(crate) struct WrittenManifestList {
    pub bytes: Vec<u8>,
    pub next_row_id: Option<u64>,
}

/// Write the manifest list at `uri` for snapshot `snapshot_id` naming
/// `manifests` in order, in the table's `format_version`. A manifest this
/// commit wrote gets this commit's sequence number and, with row lineage
/// (`first_row_id` given, format version 3), the next free first row id,
/// advanced by its rows; a carried manifest passes through unchanged.
pub(crate) fn write_manifest_list(
    uri: &str,
    format_version: FormatVersion,
    snapshot_id: i64,
    parent_snapshot_id: Option<i64>,
    sequence_number: i64,
    first_row_id: Option<u64>,
    manifests: Vec<ManifestFile>,
) -> Result<WrittenManifestList> {
    let file_io = FileIO::new_with_memory();
    let output = file_io.new_output(uri).map_err(Error::metadata)?;
    let file_writer = block_on(output.writer()).map_err(Error::metadata)?;
    let mut writer = match format_version {
        FormatVersion::V3 => ManifestListWriter::v3(
            file_writer,
            snapshot_id,
            parent_snapshot_id,
            sequence_number,
            first_row_id,
        ),
        _ => ManifestListWriter::v2(
            file_writer,
            snapshot_id,
            parent_snapshot_id,
            sequence_number,
        ),
    };
    writer
        .add_manifests(manifests.into_iter())
        .map_err(Error::metadata)?;
    let next_row_id = writer.next_row_id();
    block_on(writer.close()).map_err(Error::metadata)?;
    let bytes = read_back(&file_io, uri)?;
    Ok(WrittenManifestList { bytes, next_row_id })
}

/// The bytes a spec writer left at `uri` in the in-memory file IO.
fn read_back(file_io: &FileIO, uri: &str) -> Result<Vec<u8>> {
    let input = file_io.new_input(uri).map_err(Error::metadata)?;
    Ok(block_on(input.read()).map_err(Error::metadata)?.to_vec())
}
