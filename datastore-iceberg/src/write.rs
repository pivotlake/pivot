//! Writing to a catalog table: the catalog as the keeper of the table's
//! current version, an INSERT's target, and the commit that lands its files.
//!
//! The data files are written by the shared table crate's sink, through
//! Pivot's own store with the credentials the catalog vended. The commit is
//! the iceberg crate's own fast append, run through the datastore's catalog
//! client, whose file IO writes through Pivot's stores: the crate builds the
//! snapshot's manifest and manifest list on the table as the catalog has it
//! now, then one `updateTable` adds the snapshot and moves `main` onto it,
//! under the requirement that `main` still names the snapshot the commit was
//! built on. A catalog that refuses the requirement answers a conflict, and
//! the crate builds the commit again on whatever landed in between.

use std::collections::HashMap;
use std::sync::Arc;

use catalog::{CommitAction, CommitQueue};
use dispatch::RecordBatchOperatorSpec;
use iceberg::spec::{FormatVersion, TableMetadata};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, ErrorKind, NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::RestCatalog;
use iceberg_tables::data_files::{DataFileEntry, EntryContext};
use iceberg_tables::layout::{DATA_PATH_PROPERTY, METADATA_DIR, METADATA_PATH_PROPERTY};
use iceberg_tables::schema::{
    build_partition_spec, build_schema, build_sort_order, logical_types_property,
};
use iceberg_tables::sink::InsertTarget;
use iceberg_tables::{
    CurrentMetadata, LOGICAL_TYPES_PROPERTY, PartitionKeys, Table as LoadedTable, TableCatalog,
    TableLocation, TableStore, block_on, read_logical_types, table_columns,
};
use parquet_engine::FileStats;
use parquet_engine::writing::upload::{UploadedFile, UploadedFiles, delete_uploaded_files};
use planner::catalog::{Column, Result as CatalogResult, SchemaQualifiedTableName};

use crate::table::{current_metadata, fetch_manifest_list_size};
use crate::{Error, IcebergDatastore, Result, TableEntry};

/// A catalog table's identifier in the catalog: its namespace is the
/// schema, its name the table.
pub(crate) fn table_ident(name: &SchemaQualifiedTableName) -> TableIdent {
    TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
}

/// One table's current version as its REST catalog keeps it: what the shared
/// crate's table loads and refreshes from. Versions are not published
/// through it: a catalog table is written through the catalog's own commit.
pub(crate) struct RestTableCatalog {
    catalog: Arc<RestCatalog>,
    name: SchemaQualifiedTableName,
}

impl RestTableCatalog {
    pub(crate) fn new(catalog: Arc<RestCatalog>, name: &SchemaQualifiedTableName) -> Self {
        Self {
            catalog,
            name: name.clone(),
        }
    }
}

impl TableCatalog for RestTableCatalog {
    fn load(&self) -> iceberg_tables::Result<CurrentMetadata> {
        let table = block_on(self.catalog.load_table(&table_ident(&self.name)))
            .map_err(|error| iceberg_tables::Error::Catalog(Box::new(error)))?;
        current_metadata(&self.name, &table)
    }

    /// One load-table call; the catalog answers the same location for an
    /// unchanged table.
    fn refresh(&self, held: &CurrentMetadata) -> iceberg_tables::Result<Option<CurrentMetadata>> {
        let current = self.load()?;
        Ok((current.metadata_location != held.metadata_location).then_some(current))
    }

    fn commit_catalog(&self, _base: Table) -> iceberg_tables::Result<Arc<dyn Catalog>> {
        Err(iceberg_tables::Error::CatalogCommitsItsOwnTables {
            table: self.name.to_string(),
        })
    }
}

/// Where `metadata`'s table keeps its files under `store`: its directory,
/// and its metadata objects under `write.metadata.path` when the table sets
/// it, else under the directory's `metadata/`; its URIs spelled as the
/// table's own metadata location is.
pub(crate) fn table_location(
    metadata: &TableMetadata,
    store: &TableStore,
) -> iceberg_tables::Result<TableLocation> {
    let directory = store.key_of(metadata.location())?;
    let metadata_dir = match metadata.properties().get(METADATA_PATH_PROPERTY) {
        Some(path) => store.key_of(path.trim_end_matches('/'))?,
        None => directory.join(METADATA_DIR),
    };
    TableLocation::with_metadata_dir(
        store.store().clone(),
        store.root().to_string(),
        directory,
        metadata_dir,
    )
}

/// Where and how an INSERT writes into `table`, or why it cannot: a partition
/// field whose key cannot be computed, a format version whose manifests this
/// writer does not write. Files go under the table's data path, as the table
/// names it.
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
        table.name(),
        schema,
        metadata.default_partition_spec(),
        &columns,
    )?;
    let location = table.location();
    let file_prefix = location.data_prefix_for(
        metadata
            .properties()
            .get(DATA_PATH_PROPERTY)
            .map(String::as_str),
    )?;
    // The table's columns are its schema's top-level fields, in order.
    let field_ids = schema.as_struct().fields().iter().map(|field| field.id);
    Ok(InsertTarget {
        store: table.store().store().clone(),
        directory: location.directory().clone(),
        file_prefix,
        columns,
        nullability: table.declared_nullability(),
        field_ids: Some(field_ids.collect()),
        partition_keys,
        sort_by: table.sort_by().to_vec(),
    })
}

/// Build the dataflow that writes `input`'s rows into `table` as Parquet
/// files and emits the inserted-row count, or refuse the INSERT before any
/// file is written when the table cannot be written. Each finished file is
/// pushed onto `uploaded_files` for the statement's transaction to commit
/// to the table `name`.
pub(crate) fn build_insert_spec(
    name: SchemaQualifiedTableName,
    table: &Arc<LoadedTable>,
    commit_actions: CommitQueue<IcebergDatastore>,
    input: RecordBatchOperatorSpec,
) -> Result<RecordBatchOperatorSpec> {
    let target = writable_target(table)?;
    let uploaded_files = UploadedFiles::new();
    // The pending insert this statement commits with. Staged before any file
    // can land, so whatever lands is owned by a staged action the transaction
    // can commit or discard.
    let pending_insert: Box<dyn CommitAction<IcebergDatastore>> = Box::new(PendingInsert {
        name,
        table: Arc::clone(table),
        uploaded_files: uploaded_files.clone(),
    });
    let stage_pending_insert = Box::new(move || commit_actions.push(pending_insert));
    Ok(iceberg_tables::sink::build_insert_spec(
        &target,
        uploaded_files.queue(),
        stage_pending_insert,
        input,
    )?)
}

/// The files one INSERT uploaded, waiting for its transaction to append them
/// to the table in one snapshot. Staged by the statement's upload stage as
/// the dataflow starts; the files arrive on `uploaded_files` as it runs, so a
/// sink that wrote no rows appends nothing.
pub(crate) struct PendingInsert {
    /// The table's name in the catalog, which the commit is addressed to.
    pub name: SchemaQualifiedTableName,
    /// The table as the query's index had it when the INSERT was bound: the
    /// store its files went through, the directory their paths are relative
    /// to, and what the files were written by, which the table must still
    /// have when the commit is built.
    pub table: Arc<LoadedTable>,
    pub uploaded_files: UploadedFiles,
}

impl CommitAction<IcebergDatastore> for PendingInsert {
    /// Commit the files as one append snapshot and publish the table at the
    /// landed version into the datastore's index, so the next query sees the
    /// rows before the background refresh does. A commit that fails removes
    /// the files it did not land, as a discard does: nothing names them.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> CatalogResult<()> {
        let uploads = self.uploaded_files.take();
        if uploads.is_empty() {
            return Ok(());
        }
        let store = self.table.store();
        let committed = match self.commit_append(&datastore.catalog, &uploads) {
            Ok(committed) => committed,
            Err(error) => {
                delete_uploaded_files(
                    store.store().as_ref(),
                    self.table.location().directory(),
                    &uploads,
                );
                return Err(error.into());
            }
        };
        let manifest_list_size = fetch_manifest_list_size(&self.name, &committed, store)?
            .expect("an append leaves the table with a current snapshot");
        datastore.publish_entry(
            &self.name,
            TableEntry {
                current: current_metadata(&self.name, &committed).map_err(Error::from)?,
                store: store.clone(),
                manifest_list_size: Some(manifest_list_size),
            },
        );
        Ok(())
    }

    /// Delete every file the INSERT uploaded: none is in a snapshot, so no
    /// reader can be holding one.
    fn discard(self: Box<Self>) {
        self.uploaded_files.delete_from(
            self.table.store().store().as_ref(),
            self.table.location().directory(),
        );
    }
}

impl PendingInsert {
    /// Commit `files`, written by the bound table, as one append snapshot of
    /// the table as `catalog` has it now, through the crate's transaction,
    /// which builds the commit again on the winner each time another writer
    /// lands first. The table returned is at the landed version.
    fn commit_append(&self, catalog: &RestCatalog, files: &[UploadedFile]) -> Result<Table> {
        let current = block_on(catalog.load_table(&table_ident(&self.name))).map_err(Box::new)?;
        let data_files = data_files_of_bound_files(&self.table, &current, files)?;
        let transaction = Transaction::new(&current);
        // The files were just uploaded under fresh names, so none is in the
        // table already; the check for one would read every manifest.
        let append = transaction
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(data_files);
        let transaction = append.apply(transaction).map_err(Box::new)?;
        Ok(block_on(transaction.commit(catalog)).map_err(Box::new)?)
    }
}

/// `files`, written by the table `bound`, as the manifest of `current`, the
/// version the commit is built on, records them. The files are recorded as
/// they were written: by the field ids and the partition spec of the bound
/// table. A field id outlives a rename and is retired, never reused, by a
/// drop, so a schema that changed since is no obstacle: a renamed column's
/// bounds go under the name it has now, a dropped column's are left out. A
/// partition spec that changed is, since the tuple's layout is the spec's,
/// and so is a table created again under the name, whose files these are
/// not.
fn data_files_of_bound_files(
    bound: &LoadedTable,
    current: &Table,
    files: &[UploadedFile],
) -> Result<Vec<iceberg::spec::DataFile>> {
    let table_name = bound.name();
    let metadata = current.metadata();
    if metadata.uuid() != bound.uuid() {
        return Err(Error::TableRecreated {
            table: table_name.to_string(),
        });
    }
    let bound_spec_id = bound.metadata().default_partition_spec_id();
    let current_spec_id = metadata.default_partition_spec_id();
    if current_spec_id != bound_spec_id {
        return Err(Error::PartitionSpecChanged {
            table: table_name.to_string(),
            bound: bound_spec_id,
            current: current_spec_id,
        });
    }
    let schema = metadata.current_schema();
    // The files' columns are the bound schema's top-level fields.
    let bound_fields = bound.metadata().current_schema().as_struct().fields();
    let current_names: HashMap<&str, &str> = bound_fields
        .iter()
        .filter_map(|field| Some((field.name.as_str(), schema.name_by_field_id(field.id)?)))
        .collect();
    let columns: Vec<Column> = table_columns(
        table_name,
        schema,
        &read_logical_types(metadata.properties())?,
    )?
    .columns()
    .to_vec();
    let ctx = EntryContext {
        schema,
        partition_spec: metadata.default_partition_spec(),
        columns: &columns,
        location: bound.location(),
        format_version: metadata.format_version(),
    };
    files
        .iter()
        .map(|file| {
            let entry = DataFileEntry {
                file: file.file.clone(),
                partition: file.partition.clone(),
                stats: Some(Arc::new(stats_under_current_names(
                    parquet_engine::aggregate_file_stats(&file.row_groups),
                    &current_names,
                ))),
                origin: None,
            };
            Ok(entry.create_data_file(&ctx)?)
        })
        .collect()
}

/// `stats`, keyed by the columns' bound-time names, re-keyed by
/// `current_names`; a column with no current name is left out.
fn stats_under_current_names(stats: FileStats, current_names: &HashMap<&str, &str>) -> FileStats {
    fn rekey<V>(values: HashMap<String, V>, names: &HashMap<&str, &str>) -> HashMap<String, V> {
        values
            .into_iter()
            .filter_map(|(name, value)| Some((names.get(name.as_str())?.to_string(), value)))
            .collect()
    }
    FileStats {
        num_records: stats.num_records,
        min_values: rekey(stats.min_values, current_names),
        max_values: rekey(stats.max_values, current_names),
        null_counts: rekey(stats.null_counts, current_names),
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
