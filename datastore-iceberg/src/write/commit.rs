//! Landing an INSERT's files as one append snapshot of a catalog table.
//!
//! The commit is the iceberg crate's own fast append, run through a catalog
//! client whose file IO writes through the table's store. The crate builds
//! the commit on the table as the catalog has it: a manifest of the added
//! files, a manifest list naming it ahead of the current snapshot's
//! manifests, and the snapshot; then one `updateTable` adds the snapshot and
//! moves `main` onto it, under the requirement that `main` still names the
//! snapshot the commit was built on. A catalog that refuses the requirement
//! answers a conflict: another writer landed first, and the crate loads the
//! table again and builds the commit on the winner. The objects of a commit
//! that did not land are orphans the catalog's own maintenance reclaims.

use iceberg::Catalog;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg_catalog_rest::RestCatalog;

use crate::rest_catalog::block_on;
use crate::write::data_files::{AddedFile, EntryContext};
use crate::write::{WriteTarget, table_ident};
use crate::{Error, Result};

/// Commit `files`, written for `target`, as one append snapshot of the
/// table, and return the table at the version the commit landed.
pub(crate) fn commit_append(
    catalog: &RestCatalog,
    target: &WriteTarget,
    files: &[AddedFile],
) -> Result<Table> {
    let table = block_on(catalog.load_table(&table_ident(&target.name))).map_err(Box::new)?;
    let metadata = table.metadata();
    // The files carry the partition tuples of the spec the insert was bound
    // under; a spec that changed since would record them under the wrong
    // fields.
    if metadata.default_partition_spec_id() != target.partition_spec_id {
        return Err(Error::PartitionSpecChanged {
            table: target.name.to_string(),
        });
    }
    let ctx = EntryContext {
        schema: metadata.current_schema(),
        partition_spec: metadata.default_partition_spec(),
        partition_value_types: &target.partition_keys.value_types,
    };
    let data_files = files
        .iter()
        .map(|file| file.create_data_file(&ctx))
        .collect::<Result<Vec<_>>>()?;
    let transaction = Transaction::new(&table);
    // The files were just uploaded under fresh names, so none is in the
    // table already; the check for one would read every manifest.
    let append = transaction
        .fast_append()
        .with_check_duplicate(false)
        .add_data_files(data_files);
    let transaction = append.apply(transaction).map_err(Error::metadata)?;
    Ok(block_on(transaction.commit(catalog)).map_err(Box::new)?)
}
