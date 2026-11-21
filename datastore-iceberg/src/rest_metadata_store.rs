//! The versions of a catalog's tables as it keeps them, and one commit to a
//! table: the version the commit is built on, and the one `updateTable` that
//! asks the catalog to make its successor current.
//!
//! The shared table crate derives the committed table from the version the
//! commit was built on, so a commit must succeed only as that version's
//! successor:
//!
//! - The transaction is answered the held version as the table's current
//!   one, so it never builds the commit on any other. No error it is handed
//!   is retryable, so it asks once.
//! - The update is the one call a commit makes to the catalog. It carries
//!   the requirements the transaction built from the held version: the
//!   table's UUID, and `main` still naming the held snapshot. The catalog
//!   checks them as it applies the update, so another writer's snapshot
//!   ends the commit as a lost race.
//! - A change of another kind by another engine since the held version, to
//!   the schema, the properties, the partition spec or the sort order,
//!   leaves a legal table: the appended files are recorded by field id
//!   under the spec and the order they were written with. The committed
//!   table is built from the metadata the catalog answers, and reuses what
//!   the held version learned only when the schema, spec and sort order ids
//!   are the held ones.
//!
//! A catalog that refuses the update did not apply it. One that answers
//! without saying whether it applied it (a server error, a gateway timeout, a
//! connection that broke) may have, and the commit's error says so.

use std::sync::Arc;

use async_trait::async_trait;
use iceberg::table::Table as SpecTable;
use iceberg::{Catalog, ErrorKind, TableCommit};
use iceberg_tables::{MetadataFile, MetadataStore, block_on, build_table_ident};
use planner::catalog::SchemaQualifiedTableName;

/// The versions of every table of a catalog as the catalog keeps them: what
/// the shared crate's tables load and refresh from, and commit their next
/// versions through. One store serves all the tables of its catalog.
pub(crate) struct RestMetadataStore {
    catalog: Arc<dyn Catalog>,
}

impl RestMetadataStore {
    pub(crate) fn new(catalog: Arc<dyn Catalog>) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl MetadataStore for RestMetadataStore {
    /// One load-table call; the catalog answers the same location for an
    /// unchanged table.
    fn load_latest(&self, name: &SchemaQualifiedTableName) -> iceberg_tables::Result<MetadataFile> {
        let table = block_on(self.catalog.load_table(&build_table_ident(name)))
            .map_err(|error| iceberg_tables::Error::Catalog(Box::new(error)))?;
        MetadataFile::try_from(&table)
    }

    /// Ask the catalog to apply the commit as the successor of the base, in
    /// one call. A commit conflict means another writer moved `main` off
    /// the base's snapshot. A table that is gone is a refusal, which is no
    /// commit conflict: the transaction leaves the manifests and the
    /// manifest list of the refused commit where it wrote them. Any other
    /// error of the update leaves the outcome unknown, which covers every
    /// answer the catalog gives without saying whether it applied the
    /// commit and every failure after it did: the shared crate's error
    /// rides along as the source, which is how the table tells an unknown
    /// outcome from a failure. The commit names the table and carries the
    /// requirements built from the base, so the base is not read. The
    /// catalog writes the table's metadata files and keeps their log
    /// itself, so the expired entries are not read either.
    async fn commit(
        &self,
        _base: &SpecTable,
        commit: TableCommit,
        _expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable> {
        let error = match self.catalog.update_table(commit).await {
            Ok(committed) => return Ok(committed),
            Err(error) => error,
        };
        match error.kind() {
            ErrorKind::CatalogCommitConflicts => Err(iceberg::Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("the catalog refused the commit's requirements: {error}"),
            )),
            ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound => {
                Err(error.with_retryable(false))
            }
            _ => Err(iceberg::Error::new(
                ErrorKind::Unexpected,
                format!("the catalog did not say whether it applied the commit: {error}"),
            )
            .with_source(iceberg_tables::Error::CommitOutcomeUnknown {
                source: Box::new(iceberg_tables::Error::Catalog(Box::new(error))),
            })),
        }
    }
}
