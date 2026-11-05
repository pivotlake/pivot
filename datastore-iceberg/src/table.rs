//! A pinned Iceberg snapshot: the table as the catalog handed it over, planned
//! by the iceberg crate and read by Pivot's parquet engine. Pivot evaluates
//! the crate's plan before reading surviving files' footers.

use planner::expression::Expression;
use std::sync::Arc;

use arrow_schema::{Field, Schema};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{SchemaRef, TableMetadataRef, Transform};
use iceberg::table::Table;
use object_storage::ExternalStoreFactory;
use parquet_engine::{FileRowGroups, ParquetTable, TableColumns, load_file_row_groups};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::types::physical_arrow_type;

use crate::columns::table_columns;
use crate::deletes::count_deleted_rows;
use crate::planning::{FileDescriptor, count_rows, estimate_rows, live_delete_manifests};
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::{Error, Result};

pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    /// The table's identity across every metadata version.
    pub uuid: String,
    /// The metadata file this table was loaded from. Every commit writes a new
    /// one, so it is the table's version: a cached plan is reused only while
    /// the table is still at the location the plan was bound at.
    pub metadata_location: String,
    pub(super) metadata: TableMetadataRef,
    columns: TableColumns,
    pub(super) partition_by: Vec<String>,
    pub(super) sort_by: Vec<String>,
    /// The table as the crate plans it: reading its manifests through the
    /// datastore's shared cache and the table's store.
    pub(super) table: Table,
    pub(super) store: TableStore,
    pub(super) dispatcher: DataFlowDispatcher,
}

impl LoadedTable {
    /// Load the schema, validating unsupported table features before a plan
    /// can hide them. Manifests and data footers remain lazy.
    pub(crate) fn load(
        name: &SchemaQualifiedTableName,
        table: &Table,
        store: TableStore,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let metadata = table.metadata_ref();
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: table_name.clone(),
            })?
            .to_string();
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable { table: table_name });
        }
        let schema = metadata.current_schema();
        let columns = table_columns(&table_name, schema)?;
        let partition_by = identity_columns(
            schema,
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            metadata,
            columns,
            partition_by,
            sort_by,
            table: table.clone(),
            store,
            dispatcher: dispatcher.clone(),
        })
    }

    /// Plan the files the filters keep, then fetch their row groups, with the
    /// rows their deletion vectors delete marked on them. Path order, then
    /// file-local row-group order, makes the same filters yield the same
    /// row-group positions for scanning and materialization.
    pub(crate) fn load_parquet(&self, filters: &[Expression]) -> Result<ParquetTable> {
        let planned_files = self.plan_files(filters)?;
        tracing::debug!(table = %self.name, files = planned_files.len(), "selected Iceberg files");
        let deletion_vectors = self.read_deletion_vectors(&planned_files)?;
        let mut footers =
            self.read_footers(planned_files.iter().map(|planned| &planned.data_file))?;
        for (footer, deleted) in footers.iter_mut().zip(deletion_vectors) {
            if let Some(deleted) = deleted {
                footer.mark_deleted_rows(deleted);
            }
        }
        let row_groups: Vec<_> = footers
            .into_iter()
            .flat_map(|footer| footer.row_groups)
            .collect();
        Ok(if row_groups.is_empty() {
            ParquetTable::empty(self.declared_schema())
        } else {
            ParquetTable::new(row_groups)
        })
    }

    /// Read file metadata in parallel and return it in request order.
    pub(super) fn read_footers<'a>(
        &self,
        files: impl IntoIterator<Item = &'a FileDescriptor>,
    ) -> Result<Vec<FileRowGroups>> {
        let data_files = files
            .into_iter()
            .map(|file| self.store.data_file(&file.path, file.length))
            .collect::<Result<Vec<_>>>()?;
        Ok(load_file_row_groups(
            &self.dispatcher,
            &data_files,
            self.columns.clone(),
        )?)
    }

    pub(crate) fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    pub(crate) fn nullability(&self) -> Vec<bool> {
        self.metadata
            .current_schema()
            .as_struct()
            .fields()
            .iter()
            .map(|field| !field.required)
            .collect()
    }

    fn declared_schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(
            self.columns()
                .iter()
                .zip(self.nullability())
                .map(|(column, nullable)| {
                    Field::new(
                        &column.name,
                        physical_arrow_type(&column.col_type),
                        nullable,
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub(crate) fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid.clone(),
            version: self.metadata_location.clone(),
        }
    }

    /// The rows of the data files less the rows their deletion vectors
    /// delete, read from the manifests alone. Older writers may omit counts,
    /// and a delete manifest may fail to read or hold a delete file that is
    /// not applied; planning then has no statistic and COUNT uses the
    /// ordinary scan, which reports what is wrong.
    pub(crate) fn row_count(&self) -> Option<i64> {
        let manifest_list = self.load_manifest_list().ok()??;
        let rows = count_rows(manifest_list.entries())?;
        let delete_manifests = self
            .load_manifests(live_delete_manifests(manifest_list.entries()))
            .ok()?;
        let deleted = count_deleted_rows(delete_manifests.iter().map(Arc::as_ref)).ok()?;
        rows.checked_sub(i64::try_from(deleted).ok()?)
    }

    /// The row count as the manifest list alone has it, with no manifest
    /// read: exact when every delete file is a deletion vector, an estimate
    /// otherwise. A table without a snapshot has no rows.
    pub(crate) fn estimated_row_count(&self) -> Option<i64> {
        match self.load_manifest_list().ok()? {
            Some(manifest_list) => estimate_rows(manifest_list.entries()),
            None => Some(0),
        }
    }
}

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source field id and transform) take by identity, in field order. A
/// transformed field (a bucket, a day) is not a column value, so it is left out.
fn identity_columns<'a>(
    schema: &SchemaRef,
    fields: impl Iterator<Item = (i32, &'a Transform)>,
) -> Vec<String> {
    fields
        .filter(|(_, transform)| **transform == Transform::Identity)
        .filter_map(|(source_id, _)| schema.name_by_field_id(source_id))
        .map(str::to_string)
        .collect()
}

/// The store `table`'s files are read through: opened with the credentials
/// the catalog vended for the table, or through `store_factory` when it
/// vended none. Opened when the table is indexed rather than when a query loads
/// it, since opening a store may ask the bucket for its region; and opened
/// again at every refresh, so vended credentials never outlive their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::StagedTable {
            table: table_name.clone(),
        })?;
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    TableStore::open(&table_name, metadata_location, store_factory, vended)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};

    use super::*;

    #[test]
    fn only_identity_fields_are_columns_in_field_order() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::required(3, "day", Type::Primitive(PrimitiveType::Date)).into(),
                ])
                .build()
                .unwrap(),
        );
        let fields = [
            (2, Transform::Identity),
            (1, Transform::Bucket(16)),
            (3, Transform::Day),
            (1, Transform::Identity),
        ];

        let columns = identity_columns(
            &schema,
            fields
                .iter()
                .map(|(source_id, transform)| (*source_id, transform)),
        );

        assert_eq!(columns, ["region", "id"]);
    }
}
