//! Creating and dropping a catalog table: what a `CREATE TABLE` or a
//! `DROP TABLE` resolved to, staged on the transaction's queue when its
//! dataflow runs and applied through the catalog at commit.
//!
//! A catalog table is shared with other engines, so a created table
//! declares only types Iceberg has: a column declared with a narrow or
//! unsigned integer is created as the Iceberg integer that holds every
//! value of it ([`widen_to_native_columns`]).

use std::collections::HashMap;

use catalog::{CommitAction, CommitQueue};
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, RecordBatchOperatorSpec};
use iceberg::spec::FormatVersion;
use iceberg::{Catalog, ErrorKind, NamespaceIdent};
use iceberg_catalog_rest::RestCatalog;
use iceberg_tables::types::schema::{build_partition_spec, build_schema, build_sort_order};
use iceberg_tables::{block_on, build_table_ident};
use planner::catalog::{
    Column, Result as CatalogResult, SchemaQualifiedTableName, TableCreation, TableDrop,
};
use planner::types::{Type, sql_type_name};

use crate::{Error, IcebergDatastore, Result};

/// `columns`, the declared columns of the new table `table`, each with the
/// type Iceberg stores it as: a signed integer narrower than 32 bits and an
/// unsigned one of 8 or 16 bits as `Int32`, an unsigned one of 32 bits as
/// `Int64`, every other type as declared. An unsigned 64-bit integer is
/// refused, since Iceberg's `long` does not hold its upper half.
pub(crate) fn widen_to_native_columns(table: &str, columns: Vec<Column>) -> Result<Vec<Column>> {
    columns
        .into_iter()
        .map(|column| {
            let col_type = match column.col_type {
                Type::Int8 | Type::Int16 | Type::UInt8 | Type::UInt16 => Type::Int32,
                Type::UInt32 => Type::Int64,
                Type::UInt64 => {
                    return Err(Error::UnsignedBigintColumn {
                        table: table.to_string(),
                        column: column.name,
                        declared: sql_type_name(&Type::UInt64),
                    });
                }
                declared => declared,
            };
            Ok(Column {
                name: column.name,
                col_type,
            })
        })
        .collect()
}

/// One table creation resolved by a transaction and waiting for it to commit.
#[derive(Clone)]
pub(crate) struct PendingTableCreation {
    pub(crate) name: SchemaQualifiedTableName,
    /// The columns as the table declares them: of types Iceberg has.
    pub(crate) columns: Vec<Column>,
    /// The terms the table is partitioned by, in order, each a column or a
    /// transform of one.
    pub(crate) partition_by: Vec<String>,
    pub(crate) sort_by: Vec<String>,
    /// Whether the statement used `IF NOT EXISTS`, so the commit treats a
    /// table that appeared since resolution as a success rather than an error.
    pub(crate) if_not_exists: bool,
}

impl PendingTableCreation {
    /// Create the table in `catalog` with the shape asked for, in one
    /// call: format version 3, every column optional, partitioned by the
    /// terms `partition_by` (each a column or a transform of one) and sorted
    /// by the columns `sort_by`. The catalog assigns the location and the
    /// field ids.
    fn create_in(&self, catalog: &RestCatalog) -> Result<()> {
        let schema = build_schema(&self.columns)?;
        let partition_spec = build_partition_spec(&schema, &self.partition_by)?;
        let sort_order = build_sort_order(&schema, &self.sort_by)?;
        // The REST request carries the format version as the `format-version`
        // property; the builder's own field is not sent.
        let creation = iceberg::TableCreation::builder()
            .name(self.name.table.clone())
            .schema(schema)
            .partition_spec(partition_spec)
            .sort_order(sort_order)
            .properties(HashMap::from([(
                "format-version".to_string(),
                "3".to_string(),
            )]))
            .format_version(FormatVersion::V3)
            .build();
        let namespace = NamespaceIdent::new(self.name.schema.clone());
        block_on(catalog.create_table(&namespace, creation)).map_err(Box::new)?;
        Ok(())
    }
}

impl CommitAction<IcebergDatastore> for PendingTableCreation {
    /// Register the table through the catalog, then refresh the index, which
    /// is shaped by what the catalog lists.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> CatalogResult<()> {
        match self.create_in(&datastore.catalog) {
            Ok(()) => {}
            Err(Error::Catalog(error))
                if self.if_not_exists && error.kind() == ErrorKind::TableAlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        Ok(datastore.refresh()?)
    }

    /// A table creation writes nothing before its commit.
    fn discard(self: Box<Self>) {}
}

/// One table drop resolved by a transaction and waiting for it to commit.
#[derive(Clone)]
pub(crate) struct PendingTableDrop {
    pub(crate) name: SchemaQualifiedTableName,
    /// Whether the statement used `IF EXISTS`, so the commit treats a table
    /// that vanished since resolution as a success rather than an error.
    pub(crate) if_exists: bool,
}

impl PendingTableDrop {
    /// Drop the table from `catalog`, leaving its files where they are,
    /// as `DROP TABLE` without `PURGE` does everywhere. With `IF EXISTS`, a
    /// table that is already gone is a success.
    fn drop_from(&self, catalog: &RestCatalog) -> Result<()> {
        match block_on(catalog.drop_table(&build_table_ident(&self.name))) {
            Ok(()) => Ok(()),
            Err(error) if self.if_exists && error.kind() == ErrorKind::TableNotFound => Ok(()),
            Err(error) => Err(Error::Catalog(Box::new(error))),
        }
    }
}

impl CommitAction<IcebergDatastore> for PendingTableDrop {
    /// Unregister the table through the catalog, leaving its files, then
    /// refresh the index, which is shaped by what the catalog lists.
    fn apply(self: Box<Self>, datastore: &IcebergDatastore) -> CatalogResult<()> {
        self.drop_from(&datastore.catalog)?;
        Ok(datastore.refresh()?)
    }

    /// A table drop writes nothing before its commit.
    fn discard(self: Box<Self>) {}
}

/// A dataflow that stages `action` on the transaction's queue when it runs
/// and emits no rows: one nullary per worker, of which only the first
/// carries the action, so the catalog changes when the statement runs rather
/// than when it is planned.
fn staging_spec<T: CommitAction<IcebergDatastore> + Clone + 'static>(
    dispatcher: &DataFlowDispatcher,
    commit_actions: &CommitQueue<IcebergDatastore>,
    action: T,
) -> RecordBatchOperatorSpec {
    let mut carried = Some(action);
    let factories: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| {
            let carried = carried.take();
            let commit_actions = commit_actions.clone();
            OneShotNullaryFactory::new(move || {
                if let Some(action) = carried {
                    commit_actions.push(Box::new(action));
                }
                None
            })
        })
        .collect();
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// A resolved `CREATE TABLE` for a catalog table, staged when its dataflow
/// runs.
pub(crate) struct IcebergTableCreation {
    pub(crate) commit_actions: CommitQueue<IcebergDatastore>,
    pub(crate) creation: PendingTableCreation,
}

impl TableCreation for IcebergTableCreation {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(staging_spec(
            dispatcher,
            &self.commit_actions,
            self.creation.clone(),
        ))
    }
}

/// A resolved `DROP TABLE` for a catalog table, staged when its dataflow
/// runs.
pub(crate) struct IcebergTableDrop {
    pub(crate) commit_actions: CommitQueue<IcebergDatastore>,
    pub(crate) drop: PendingTableDrop,
}

impl TableDrop for IcebergTableDrop {
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(staging_spec(
            dispatcher,
            &self.commit_actions,
            self.drop.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    #[test]
    fn narrow_and_unsigned_integers_widen_to_the_integer_iceberg_stores_them_as() {
        let declared = vec![
            column("tiny", Type::Int8),
            column("small", Type::Int16),
            column("unsigned_tiny", Type::UInt8),
            column("unsigned_small", Type::UInt16),
            column("unsigned", Type::UInt32),
        ];

        let widened = widen_to_native_columns("main.t", declared).unwrap();

        let types: Vec<Type> = widened.into_iter().map(|column| column.col_type).collect();
        assert_eq!(
            types,
            [
                Type::Int32,
                Type::Int32,
                Type::Int32,
                Type::Int32,
                Type::Int64
            ]
        );
    }

    #[test]
    fn a_type_iceberg_has_is_kept_as_declared() {
        let declared = vec![
            column("id", Type::Int64),
            column("count", Type::Int32),
            column("name", Type::Utf8),
            column("day", Type::Date),
        ];

        let widened = widen_to_native_columns("main.t", declared.clone()).unwrap();

        assert_eq!(widened, declared);
    }

    #[test]
    fn an_unsigned_bigint_is_refused_naming_the_column_and_what_to_declare() {
        let declared = vec![column("id", Type::Int64), column("total", Type::UInt64)];

        let error = widen_to_native_columns("main.t", declared)
            .unwrap_err()
            .to_string();

        assert!(error.contains("`total`"), "{error}");
        assert!(error.contains("UBIGINT"), "{error}");
        assert!(error.contains("DECIMAL(20, 0)"), "{error}");
    }
}
