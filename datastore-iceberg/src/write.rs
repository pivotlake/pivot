//! Writing to a catalog table: the catalog as the keeper of the table's
//! current version, an INSERT's target, and the commit that lands its files.
//!
//! The data files, manifests and manifest list of a commit are written by
//! the shared table crate, through Pivot's own store with the credentials
//! the catalog vended. Only the last step is the catalog's: an `updateTable`
//! that adds the snapshot and moves `main` onto it, under the requirement
//! that `main` still names the snapshot the commit was built on. A catalog
//! that refuses the requirement answers a conflict, and the commit is built
//! again on whatever landed in between.

use std::collections::HashMap;
use std::sync::Arc;

use crossbeam_deque::Injector;
use dispatch::DataFlowDispatcher;
use iceberg::spec::{FormatVersion, SnapshotReference, SnapshotRetention, TableMetadata};
use iceberg::table::Table;
use iceberg::{
    Catalog, ErrorKind, NamespaceIdent, TableCommit, TableCreation, TableIdent, TableRequirement,
    TableUpdate,
};
use iceberg_catalog_rest::RestCatalog;
use iceberg_tables::commit::{COMMIT_RETRIES, now_unix_ms};
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::layout::{DATA_PATH_PROPERTY, METADATA_DIR, METADATA_PATH_PROPERTY};
use iceberg_tables::schema::{
    build_partition_spec, build_schema, build_sort_order, logical_types_property,
};
use iceberg_tables::sink::InsertTarget;
use iceberg_tables::{
    CommitChanges, CurrentMetadata, LOGICAL_TYPES_PROPERTY, PartitionKeys, Table as LoadedTable,
    TableCatalog, TableLocation, TableStore, block_on, identity_columns, read_logical_types,
    table_columns,
};
use parquet_engine::writing::upload::UploadedFile;
use planner::catalog::{Column, SchemaQualifiedTableName};

use crate::{CommitAction, Error, IcebergDatastore, Result, TableEntry, drain_injector};

/// The branch every commit of this datastore moves.
const MAIN_BRANCH: &str = "main";

/// A catalog table's identifier in the catalog: its namespace is the
/// schema, its name the table.
pub(crate) fn table_ident(name: &SchemaQualifiedTableName) -> TableIdent {
    TableIdent::new(NamespaceIdent::new(name.schema.clone()), name.table.clone())
}

/// The current metadata `table` carries, as the shared crate holds it. A
/// staged table has no metadata location yet and cannot be written.
fn current_metadata(name: &str, table: &Table) -> iceberg_tables::Result<CurrentMetadata> {
    let metadata_location =
        table
            .metadata_location()
            .ok_or_else(|| iceberg_tables::Error::TableMetadataMissing {
                table: name.to_string(),
                location: table.identifier().to_string(),
            })?;
    Ok(CurrentMetadata {
        metadata_location: metadata_location.to_string(),
        metadata: table.metadata_ref(),
    })
}

/// One table's versions as its REST catalog keeps them.
pub(crate) struct RestTableCatalog {
    catalog: Arc<RestCatalog>,
    ident: TableIdent,
    name: String,
}

impl RestTableCatalog {
    pub(crate) fn new(catalog: Arc<RestCatalog>, name: &SchemaQualifiedTableName) -> Self {
        Self {
            catalog,
            ident: table_ident(name),
            name: name.to_string(),
        }
    }

    /// The table as the catalog has it now.
    pub(crate) fn load_table(&self) -> iceberg_tables::Result<Table> {
        block_on(self.catalog.load_table(&self.ident))
            .map_err(|error| iceberg_tables::Error::Catalog(Box::new(error)))
    }
}

impl TableCatalog for RestTableCatalog {
    fn load(&self) -> iceberg_tables::Result<CurrentMetadata> {
        current_metadata(&self.name, &self.load_table()?)
    }

    /// One load-table call; the catalog answers the same location for an
    /// unchanged table.
    fn refresh(&self, held: &CurrentMetadata) -> iceberg_tables::Result<Option<CurrentMetadata>> {
        let current = self.load()?;
        Ok((current.metadata_location != held.metadata_location).then_some(current))
    }

    /// Add the snapshot and move `main` onto it, provided the table is still
    /// the one the commit was built on (same uuid, `main` at the same
    /// snapshot). The catalog's conflict answer is a lost race.
    fn publish(
        &self,
        base: Option<&CurrentMetadata>,
        _metadata: Arc<TableMetadata>,
        changes: &CommitChanges,
    ) -> iceberg_tables::Result<Option<CurrentMetadata>> {
        let base = base.ok_or_else(|| iceberg_tables::Error::CatalogCreatesTables {
            table: self.name.clone(),
        })?;
        let requirements = vec![
            TableRequirement::UuidMatch {
                uuid: base.metadata.uuid(),
            },
            TableRequirement::RefSnapshotIdMatch {
                r#ref: MAIN_BRANCH.to_string(),
                snapshot_id: base.metadata.current_snapshot_id(),
            },
        ];
        let mut updates = Vec::new();
        if let Some(snapshot) = &changes.snapshot {
            updates.push(TableUpdate::AddSnapshot {
                snapshot: snapshot.clone(),
            });
            updates.push(TableUpdate::SetSnapshotRef {
                ref_name: MAIN_BRANCH.to_string(),
                reference: SnapshotReference::new(
                    snapshot.snapshot_id(),
                    SnapshotRetention::branch(None, None, None),
                ),
            });
        }
        if !changes.expired.is_empty() {
            updates.push(TableUpdate::RemoveSnapshots {
                snapshot_ids: changes.expired.clone(),
            });
        }
        let commit = TableCommit::new(self.ident.clone(), requirements, updates);
        match block_on(self.catalog.update_table(commit)) {
            Ok(table) => current_metadata(&self.name, &table).map(Some),
            Err(error) if error.kind() == ErrorKind::CatalogCommitConflicts => Ok(None),
            Err(error) => Err(iceberg_tables::Error::Catalog(Box::new(error))),
        }
    }

    fn expires_snapshots(&self) -> bool {
        false
    }
}

/// A table a transaction will write into, as the query's index had it when
/// the INSERT was bound: the store its reads go through, whose vended keys
/// the refresh renews, and the catalog client the commit lands through.
pub(crate) struct WriteTarget {
    pub name: SchemaQualifiedTableName,
    pub catalog: Arc<RestTableCatalog>,
    pub store: TableStore,
    pub location: TableLocation,
}

/// Where `metadata`'s table keeps its files under `store`: its directory,
/// and its metadata objects under `write.metadata.path` when the table sets
/// it, else under the directory's `metadata/`.
pub(crate) fn table_location(
    metadata: &TableMetadata,
    store: &TableStore,
) -> iceberg_tables::Result<TableLocation> {
    let directory = store.key_of(metadata.location())?;
    let metadata_dir = match metadata.properties().get(METADATA_PATH_PROPERTY) {
        Some(path) => store.key_of(path.trim_end_matches('/'))?,
        None => directory.join(METADATA_DIR),
    };
    TableLocation::with_metadata_dir(store.store().clone(), directory, metadata_dir)
}

/// Where and how an INSERT writes into the table `metadata` describes, or why
/// it cannot: a partition field whose key cannot be computed, a format version
/// whose manifests this writer does not write, a column whose type Pivot
/// cannot represent. Files go under the table's data path, manifests under
/// its metadata path, both as the table names them.
pub(crate) fn writable_target(
    name: &SchemaQualifiedTableName,
    metadata: &TableMetadata,
    store: &TableStore,
) -> Result<(InsertTarget, TableLocation)> {
    let table_name = name.to_string();
    if metadata.format_version() < FormatVersion::V2 {
        return Err(iceberg_tables::Error::UnsupportedFormatVersion {
            table: table_name,
            version: metadata.format_version() as u8,
        }
        .into());
    }
    let schema = metadata.current_schema();
    let logical_types = read_logical_types(metadata.properties())?;
    let columns: Vec<Column> = table_columns(&table_name, schema, &logical_types)?
        .columns()
        .to_vec();
    let mut field_ids = Vec::with_capacity(columns.len());
    let mut nullability = Vec::with_capacity(columns.len());
    for column in &columns {
        let field = schema.field_by_name(&column.name).ok_or_else(|| {
            iceberg_tables::Error::UnknownSpecColumn {
                option: "schema".to_string(),
                column: column.name.clone(),
            }
        })?;
        field_ids.push(field.id);
        nullability.push(!field.required);
    }
    let partition_keys = PartitionKeys::plan(
        &table_name,
        schema,
        metadata.default_partition_spec(),
        &columns,
    )?;
    let sort_by = identity_columns(
        schema,
        metadata
            .default_sort_order()
            .fields
            .iter()
            .map(|field| (field.source_id, &field.transform)),
    );

    let location = table_location(metadata, store)?;
    let directory = location.directory().clone();
    let file_prefix = location.data_prefix_for(
        metadata
            .properties()
            .get(DATA_PATH_PROPERTY)
            .map(String::as_str),
    )?;
    let target = InsertTarget {
        store: store.store().clone(),
        directory,
        file_prefix,
        columns,
        nullability,
        field_ids: Some(field_ids),
        partition_keys,
        sort_by,
    };
    Ok((target, location))
}

/// The files one INSERT uploaded, waiting for its transaction to append them
/// to the table in one snapshot. Staged when the statement's sink is built;
/// the files arrive on `uploaded_files` as the dataflow runs, so a sink that
/// wrote no rows appends nothing.
pub(crate) struct PendingInsert {
    pub target: Arc<WriteTarget>,
    pub uploaded_files: Arc<Injector<UploadedFile>>,
}

impl CommitAction for PendingInsert {
    /// Commit the files as one append snapshot and publish the table at the
    /// landed version into the datastore's index, so the next query sees the
    /// rows before the background refresh does.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> Result<()> {
        let uploads = drain_injector(&self.uploaded_files);
        if uploads.is_empty() {
            return Ok(());
        }
        let committed = commit_uploads(&datastore.dispatcher, &self.target, &uploads)?;
        datastore.publish_entry(
            &self.target.name,
            TableEntry {
                metadata: Arc::clone(committed.metadata()),
                metadata_location: committed.metadata_location().to_string(),
                store: self.target.store.clone(),
                manifest_list_size: committed.manifest_list_size(),
            },
        );
        Ok(())
    }
}

/// Commit `files`, uploaded under `target`, as one append snapshot of the
/// table, building the commit again on the winner each time another writer
/// lands first. The table returned is at the landed version.
pub(crate) fn commit_uploads(
    dispatcher: &DataFlowDispatcher,
    target: &WriteTarget,
    files: &[UploadedFile],
) -> Result<LoadedTable> {
    let table_name = target.name.to_string();
    let load = || {
        LoadedTable::load(
            dispatcher,
            Arc::clone(&target.catalog) as Arc<dyn TableCatalog>,
            target.location.clone(),
            target.store.clone(),
            &table_name,
        )
    };
    let added: Vec<DataFileEntry> = files
        .iter()
        .map(|file| DataFileEntry {
            file: file.file.clone(),
            partition: file.partition.clone(),
            stats: Some(Arc::new(parquet_engine::aggregate_file_stats(
                &file.row_groups,
            ))),
            origin: None,
        })
        .collect();
    let mut table = load()?;
    for _ in 0..COMMIT_RETRIES {
        match table.commit(&[], added.clone(), true, now_unix_ms())? {
            Some(committed) => return Ok(committed),
            None => table = load()?,
        }
    }
    Err(iceberg_tables::Error::CommitRacesExhausted {
        attempts: COMMIT_RETRIES,
    }
    .into())
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
) -> Result<Table> {
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
        return Ok(table);
    };
    let commit = TableCommit::new(
        table_ident(name),
        vec![TableRequirement::UuidMatch {
            uuid: table.metadata().uuid(),
        }],
        vec![TableUpdate::SetProperties {
            updates: HashMap::from([(LOGICAL_TYPES_PROPERTY.to_string(), logical_types)]),
        }],
    );
    Ok(block_on(catalog.update_table(commit)).map_err(Box::new)?)
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
