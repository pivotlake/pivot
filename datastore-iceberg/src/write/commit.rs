//! Landing an INSERT's files as one append snapshot of a catalog table.
//!
//! The commit is the iceberg crate's own fast append, run through the
//! datastore's catalog client, whose file IO writes through Pivot's stores.
//! The crate loads the table as the catalog has it now, builds the commit on
//! that: a manifest of the added files, a manifest list naming it ahead of
//! the current snapshot's manifests, and the snapshot; then one `updateTable`
//! adds the snapshot and moves `main` onto it, under the requirement that
//! `main` still names the snapshot the commit was built on. A catalog that
//! refuses the requirement answers a conflict: another writer landed first,
//! and the crate loads the table again and builds the commit on the winner.
//! The objects of a commit that did not land are orphans the catalog's own
//! maintenance reclaims.
//!
//! The files are recorded as they were written: by the field ids and the
//! partition spec of the table the INSERT was bound to. A field id outlives
//! a rename and is retired, never reused, by a drop, so a schema that changed
//! since is no obstacle; a partition spec that changed is, since the tuple's
//! layout is the spec's, and the crate refuses the files.

use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg_catalog_rest::RestCatalog;

use crate::rest_catalog::block_on;
use crate::write::PendingInsert;
use crate::write::data_files::AddedFile;
use crate::{Error, Result};

impl PendingInsert {
    /// Commit `files`, written into the bound table under its partition keys,
    /// as one append snapshot of `current`, the same table as the index holds
    /// it now, and return the table at the version the commit landed.
    ///
    /// A table created again under the name since the INSERT was bound is
    /// another table, whose own files these are not; one created again since
    /// the last refresh is not told apart, as the reference implementation
    /// does not tell it apart either.
    pub(super) fn commit_append(
        &self,
        catalog: &RestCatalog,
        current: &Table,
        files: &[AddedFile],
    ) -> Result<Table> {
        let bound = &self.table;
        if current.metadata().uuid().to_string() != bound.uuid {
            return Err(Error::TableRecreated {
                table: bound.name.to_string(),
            });
        }
        let data_files = files
            .iter()
            .map(|file| self.manifest_entry_of(file))
            .collect::<Result<Vec<_>>>()?;
        let transaction = Transaction::new(current);
        // The files were just uploaded under fresh names, so none is in the
        // table already; the check for one would read every manifest.
        let append = transaction
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(data_files);
        let transaction = append.apply(transaction).map_err(Error::metadata)?;
        Ok(block_on(transaction.commit(catalog)).map_err(Box::new)?)
    }
}
