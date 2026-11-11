//! Writing to a catalog table: the catalog as the keeper of the table's
//! current version, an INSERT's target, and the commit that adds its files.
//!
//! The data files are written by the shared table crate's sink, through
//! Pivot's own store with the credentials the catalog vended. The commit is
//! the shared table crate's: the table is brought to the version the catalog
//! has current, the files are recorded against that version, and the
//! snapshot that appends them is built on it and committed through the
//! catalog ([`crate::commit`]), which makes it current only while that
//! version still is. A commit that lost to another writer's is built again
//! on the winner.

use std::collections::HashMap;
use std::sync::Arc;

use catalog::{CommitAction, CommitQueue};
use dispatch::RecordBatchOperatorSpec;
use iceberg::spec::{FormatVersion, Schema};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, ErrorKind, NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::RestCatalog;
use iceberg_tables::commit::COMMIT_RETRIES;
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::io::sink::InsertTarget;
use iceberg_tables::metadata::layout::DATA_PATH_PROPERTY;
use iceberg_tables::types::schema::{
    build_partition_spec, build_schema, build_sort_order, logical_types_property,
};
use iceberg_tables::{LOGICAL_TYPES_PROPERTY, PartitionKeys, Table as LoadedTable, block_on};
use object_storage::FileRef;
use parquet_engine::FileStats;
use parquet_engine::writing::upload::{UploadedFile, UploadedFiles};
use planner::catalog::{Column, Result as CatalogResult, SchemaQualifiedTableName};

use crate::{Error, IcebergDatastore, Result, TableEntry};

/// A catalog table's identifier in the catalog: its namespace is the
/// schema, its name the table.
pub(crate) fn table_ident(name: &SchemaQualifiedTableName) -> TableIdent {
    TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
}

/// Where and how an INSERT writes into `table`, or why it cannot: a partition
/// field whose key cannot be computed, a format version whose manifests this
/// writer does not write. Files go into the table's data directory.
fn writable_target(table: &LoadedTable) -> Result<InsertTarget> {
    let metadata = table.metadata();
    if metadata.format_version() < FormatVersion::V2 {
        return Err(iceberg_tables::Error::UnsupportedFormatVersion {
            table: table.name().to_string(),
            version: metadata.format_version() as u8,
        }
        .into());
    }
    let schema = metadata.current_schema();
    let columns = table.columns().to_vec();
    let partition_keys = PartitionKeys::plan(
        &table.name().to_string(),
        schema,
        metadata.default_partition_spec(),
        &columns,
    )?;
    let data_directory = table.location().data_directory_for(
        metadata
            .properties()
            .get(DATA_PATH_PROPERTY)
            .map(String::as_str),
    )?;
    // The table's columns are its schema's top-level fields, in order.
    let field_ids = schema.as_struct().fields().iter().map(|field| field.id);
    Ok(InsertTarget {
        uploaded_files: UploadedFiles::new(table.location().store().clone(), data_directory),
        columns,
        nullability: table.schema_nullability(),
        field_ids: Some(field_ids.collect()),
        partition_keys,
        sort_by: table.sort_by().to_vec(),
    })
}

/// Build the dataflow that writes `input`'s rows into `table` as Parquet
/// files and emits the inserted-row count, or refuse the INSERT before any
/// file is written when the table cannot be written. Each finished file is
/// pushed onto the target's uploaded files for the statement's transaction
/// to commit.
pub(crate) fn build_insert_spec(
    table: &Arc<LoadedTable>,
    commit_actions: CommitQueue<IcebergDatastore>,
    input: RecordBatchOperatorSpec,
) -> Result<RecordBatchOperatorSpec> {
    let target = writable_target(table)?;
    // The pending insert this statement commits with. Staged before any file
    // can land, so whatever lands is owned by a staged action the transaction
    // can commit or discard.
    let pending_insert: Box<dyn CommitAction<IcebergDatastore>> = Box::new(PendingInsert {
        table: Arc::clone(table),
        uploaded_files: target.uploaded_files.clone(),
    });
    let stage_pending_insert = Box::new(move || commit_actions.push(pending_insert));
    Ok(target.build_insert_spec(stage_pending_insert, input)?)
}

/// The files one INSERT uploaded, waiting for its transaction to append them
/// to the table in one snapshot. Staged by the statement's upload stage as
/// the dataflow starts; the files arrive on `uploaded_files` as it runs, so a
/// sink that wrote no rows appends nothing.
pub(crate) struct PendingInsert {
    /// The table as the query's index had it when the INSERT was bound: its
    /// name in the catalog, which the commit is addressed to, and what the
    /// files were written by, which the table must still have when the
    /// commit is built.
    pub table: Arc<LoadedTable>,
    /// The files as they land, with the store and the directory they were
    /// written into.
    pub uploaded_files: UploadedFiles,
}

impl CommitAction<IcebergDatastore> for PendingInsert {
    /// Commit the files as one append snapshot and publish the table at the
    /// committed version into the datastore's index, so the next query sees the
    /// rows before the background refresh does. A commit that did not succeed
    /// removes the files it wrote, as a discard does: nothing names them. A
    /// commit whose outcome is unknown may have succeeded, so its files stay.
    /// Once the commit has succeeded so has the statement: a version
    /// that cannot be entered now is seen at the next refresh.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> CatalogResult<()> {
        let uploads = self.uploaded_files.take();
        if uploads.is_empty() {
            return Ok(());
        }
        match self.commit_append(&uploads) {
            Ok(committed) => {
                datastore.publish_entry(self.table.name(), TableEntry::loaded(Arc::new(committed)))
            }
            Err(Error::Tables(error @ iceberg_tables::Error::CommittedTableUnloaded { .. })) => {
                tracing::warn!(
                    table = %self.table.name(),
                    error = %error,
                    "the insert was committed, but its version could not be entered; the next refresh enters it"
                )
            }
            Err(error) => {
                if !error.may_have_committed() {
                    self.uploaded_files.delete(&uploads);
                }
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Delete every file the INSERT uploaded: none is in a snapshot, so no
    /// reader can be holding one.
    fn discard(self: Box<Self>) {
        self.uploaded_files.delete_all();
    }
}

impl PendingInsert {
    /// Commit `files`, written by the bound table, as one append snapshot on
    /// top of the version the catalog has current, and hand back the table
    /// at the committed version. A commit that lost to another writer's is
    /// built again on the winner, the files recorded anew against it, up to
    /// [`COMMIT_RETRIES`] times.
    fn commit_append(&self, files: &[UploadedFile]) -> Result<LoadedTable> {
        let mut held = Arc::clone(&self.table);
        for _ in 0..COMMIT_RETRIES {
            let current = self.load_current_version(&held)?;
            let entries = self.build_entries(&current, files)?;
            if let Some(committed) = current.commit(&[], entries)? {
                return Ok(committed);
            }
            held = current;
        }
        Err(iceberg_tables::Error::CommitRacesExhausted {
            attempts: COMMIT_RETRIES,
        }
        .into())
    }

    /// The table at the version the catalog has current, derived from
    /// `held`, an earlier version of it; `held` itself when that is still
    /// current. A table created again under the name is refused before
    /// anything of it is read: its files are elsewhere, and the INSERT's
    /// are not its own. A current schema Pivot cannot read refuses the
    /// load.
    fn load_current_version(&self, held: &Arc<LoadedTable>) -> Result<Arc<LoadedTable>> {
        let current = held.metadata_store().load_latest()?;
        if current.metadata_location == held.metadata_location() {
            return Ok(Arc::clone(held));
        }
        if current.metadata.uuid() != self.table.uuid() {
            return Err(Error::TableRecreated {
                table: self.table.name().to_string(),
            });
        }
        let mut table = LoadedTable::at(
            held.dispatcher(),
            Arc::clone(held.metadata_store()),
            held.location().clone(),
            held.name(),
            current,
            None,
        )?;
        table.carry_over_from(held);
        Ok(Arc::new(table))
    }

    /// `files`, written by the bound table into the directory they were
    /// uploaded to, as `current`, the version the commit is built on,
    /// records them: each under the path the table names it by. The files
    /// are recorded as they were written: by the field ids and the
    /// partition spec of the bound table. A field id outlives a rename and is
    /// retired, never reused, by a drop, so a schema that changed since is no
    /// obstacle: a renamed column's bounds go under the name it has now, a
    /// dropped column's are left out. Nor is a default sort order that
    /// changed: a file's entry names the order the bound table had, by its
    /// id, when the writer reports the rows in it, and the unsorted order
    /// when the bound table's order is one the writer does not produce. A
    /// partition spec that changed is, since the tuple's layout is the
    /// spec's.
    fn build_entries(
        &self,
        current: &LoadedTable,
        files: &[UploadedFile],
    ) -> Result<Vec<DataFileEntry>> {
        let bound: &LoadedTable = &self.table;
        let directory = self.uploaded_files.directory();
        let bound_spec_id = bound.metadata().default_partition_spec_id();
        let current_spec_id = current.metadata().default_partition_spec_id();
        if current_spec_id != bound_spec_id {
            return Err(Error::PartitionSpecChanged {
                table: bound.name().to_string(),
                bound: bound_spec_id,
                current: current_spec_id,
            });
        }
        let bound_schema = bound.metadata().current_schema();
        let current_schema = current.metadata().current_schema();
        files
            .iter()
            .map(|file| {
                Ok(DataFileEntry {
                    file: FileRef {
                        path: bound
                            .location()
                            .file_path_under(directory, &file.file.path)?,
                        size: file.file.size,
                    },
                    partition: file.partition.clone(),
                    stats: Some(Arc::new(file_stats_under_current_names(
                        parquet_engine::aggregate_file_stats(&file.row_groups),
                        bound_schema,
                        current_schema,
                    ))),
                    sort_order_id: Some(
                        bound.find_sort_order_id_of_rows_sorted_by(&file.sorted_by),
                    ),
                    first_row_id: None,
                })
            })
            .collect()
    }
}

/// `stats` of a file whose columns are `bound_schema`'s top-level fields,
/// keyed by their names there, re-keyed by the names the same field ids have
/// in `current_schema`; a column `current_schema` no longer has is left out.
fn file_stats_under_current_names(
    stats: FileStats,
    bound_schema: &Schema,
    current_schema: &Schema,
) -> FileStats {
    let current_names: HashMap<&str, &str> = bound_schema
        .as_struct()
        .fields()
        .iter()
        .filter_map(|field| {
            Some((
                field.name.as_str(),
                current_schema.name_by_field_id(field.id)?,
            ))
        })
        .collect();
    fn rekey<V>(values: HashMap<String, V>, names: &HashMap<&str, &str>) -> HashMap<String, V> {
        values
            .into_iter()
            .filter_map(|(name, value)| Some((names.get(name.as_str())?.to_string(), value)))
            .collect()
    }
    FileStats {
        num_records: stats.num_records,
        min_values: rekey(stats.min_values, &current_names),
        max_values: rekey(stats.max_values, &current_names),
        null_counts: rekey(stats.null_counts, &current_names),
    }
}

/// Create the table `name` in the catalog with the shape asked for: format
/// version 3, every column optional, identity partitioning and sorting by
/// the columns named. The catalog assigns the location and the field ids,
/// so the property naming the Pivot types Iceberg has no spelling for is
/// set afterwards, keyed by the ids it assigned.
pub(crate) fn create_table(
    catalog: &RestCatalog,
    name: &SchemaQualifiedTableName,
    columns: &[Column],
    partition_by: &[String],
    sort_by: &[String],
) -> Result<()> {
    let schema = build_schema(columns)?;
    let partition_spec = build_partition_spec(&schema, partition_by)?;
    let sort_order = build_sort_order(&schema, sort_by)?;
    // The REST request carries the format version as the `format-version`
    // property; the builder's own field is not sent.
    let creation = TableCreation::builder()
        .name(name.table.clone())
        .schema(schema)
        .partition_spec(partition_spec)
        .sort_order(sort_order)
        .properties(HashMap::from([(
            "format-version".to_string(),
            "3".to_string(),
        )]))
        .format_version(FormatVersion::V3)
        .build();
    let namespace = NamespaceIdent::new(name.schema.clone());
    let table = block_on(catalog.create_table(&namespace, creation)).map_err(Box::new)?;
    let Some(logical_types) = logical_types_property(table.metadata().current_schema(), columns)
    else {
        return Ok(());
    };
    let transaction = Transaction::new(&table);
    let transaction = transaction
        .update_table_properties()
        .set(LOGICAL_TYPES_PROPERTY.to_string(), logical_types)
        .apply(transaction)
        .map_err(Box::new)?;
    block_on(transaction.commit(catalog)).map_err(Box::new)?;
    Ok(())
}

/// Drop the table `name` from the catalog, leaving its files where they are,
/// as `DROP TABLE` without `PURGE` does everywhere. With `if_exists`, a
/// table that is already gone is a success.
pub(crate) fn drop_table(
    catalog: &RestCatalog,
    name: &SchemaQualifiedTableName,
    if_exists: bool,
) -> Result<()> {
    match block_on(catalog.drop_table(&table_ident(name))) {
        Ok(()) => Ok(()),
        Err(error) if if_exists && error.kind() == ErrorKind::TableNotFound => Ok(()),
        Err(error) => Err(Error::Catalog(Box::new(error))),
    }
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Type};

    use super::*;

    /// A schema of string columns with the ids and names given.
    fn schema_of(fields: &[(i32, &str)]) -> Schema {
        let fields = fields.iter().map(|(id, name)| {
            NestedField::optional(*id, *name, Type::Primitive(PrimitiveType::String)).into()
        });
        Schema::builder().with_fields(fields).build().unwrap()
    }

    /// Statistics recording a NULL count of 1 for each of `columns`.
    fn stats_of(columns: &[&str]) -> FileStats {
        FileStats {
            num_records: Some(1),
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            null_counts: columns
                .iter()
                .map(|column| (column.to_string(), 1))
                .collect(),
        }
    }

    #[test]
    fn statistics_of_a_renamed_column_are_filed_under_its_current_name() {
        let bound_schema = schema_of(&[(1, "id"), (2, "customer")]);
        let current_schema = schema_of(&[(1, "id"), (2, "buyer")]);

        let stats = file_stats_under_current_names(
            stats_of(&["id", "customer"]),
            &bound_schema,
            &current_schema,
        );

        let mut columns: Vec<&str> = stats.null_counts.keys().map(String::as_str).collect();
        columns.sort_unstable();
        assert_eq!(columns, ["buyer", "id"]);
    }

    #[test]
    fn statistics_of_a_dropped_column_are_left_out() {
        let bound_schema = schema_of(&[(1, "id"), (2, "customer")]);
        let current_schema = schema_of(&[(1, "id")]);

        let stats = file_stats_under_current_names(
            stats_of(&["id", "customer"]),
            &bound_schema,
            &current_schema,
        );

        let columns: Vec<&str> = stats.null_counts.keys().map(String::as_str).collect();
        assert_eq!(columns, ["id"]);
    }
}
