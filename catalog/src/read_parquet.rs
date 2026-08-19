//! The `read_parquet(location)` table function: reading Parquet files a query
//! names, rather than tables the catalog holds.
//!
//! It lives here, above the datastores, because a location belongs to none of
//! them: the store it sits in is opened through the **metastore**, so the same
//! scoped secret that authenticates a datastore's own storage authenticates
//! these files. Everything after that is a datastore's work, on the pool the
//! datastores read through - the listing, the footers, and the scan they
//! compile to.
//!
//! Binding is where the work happens: the location is listed and every matched
//! file's footer is read, which is what lets the call report the columns the
//! files actually hold. Nothing is kept between queries, so a query over a
//! location always reads the files that are there when it is planned (and the
//! binding it produces refuses to be plan-cached for exactly that reason).

use std::sync::Arc;

use datastore_delta::store::LocationPattern;
use datastore_delta::{bind_external_parquet, list_external_parquet_files};
use dispatch::DataFlowDispatcher;
use metastore::Metastore;
use planner::catalog::{Error as CatalogError, Result as CatalogResult};
use planner::types::Type;
use planner::{BoundTableFunction, ScalarValue, TableFunction};

/// The spellings this function is invoked as. `parquet_scan` is the other name
/// for the same read, as DuckDB spells it.
const NAMES: [&str; 2] = ["read_parquet", "parquet_scan"];

/// A read of a location, before any call is bound: it holds only what reaching
/// storage takes.
pub(crate) struct ReadParquet {
    /// The name this call was made under, so an error quotes what was written.
    name: &'static str,
    /// Where the credentials for an arbitrary location come from.
    metastore: Arc<dyn Metastore>,
    /// The pool the footer reads and the scan run on.
    dispatcher: DataFlowDispatcher,
}

impl ReadParquet {
    /// This function under `name`, or `None` if that is not one of its
    /// spellings.
    pub(crate) fn under_name(
        name: &str,
        metastore: Arc<dyn Metastore>,
        dispatcher: DataFlowDispatcher,
    ) -> Option<Self> {
        let name = NAMES.into_iter().find(|spelling| *spelling == name)?;
        Some(Self {
            name,
            metastore,
            dispatcher,
        })
    }
}

/// A call that never reached the files it named.
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("{function} takes one argument, the location to read, but was given {count}")]
    Arity {
        function: &'static str,
        count: usize,
    },
    #[error("{function}: the location must be a string, not `{argument}`")]
    NotALocation {
        function: &'static str,
        argument: String,
    },
    #[error("cannot open the storage holding `{location}`: {source}")]
    OpenStore {
        location: String,
        #[source]
        source: metastore::Error,
    },
}

impl From<Error> for CatalogError {
    fn from(error: Error) -> Self {
        CatalogError::Other(Box::new(error))
    }
}

impl TableFunction for ReadParquet {
    fn name(&self) -> &str {
        self.name
    }

    fn argument_types(&self) -> Vec<Type> {
        vec![Type::Utf8]
    }

    fn bind(&self, arguments: &[ScalarValue]) -> CatalogResult<BoundTableFunction> {
        let location = self.location_argument(arguments)?;
        let pattern = LocationPattern::parse(&location)
            .map_err(|error| CatalogError::Other(Box::new(error)))?;
        let store = self
            .metastore
            .open_store(&pattern.directory_uri)
            .map_err(|source| Error::OpenStore {
                location: location.clone(),
                source,
            })?;
        let files = list_external_parquet_files(store.as_ref(), &pattern.name_pattern, &location)?;
        let binding = bind_external_parquet(&self.dispatcher, location, files)?;
        Ok(BoundTableFunction::Table(Box::new(binding)))
    }
}

impl ReadParquet {
    /// The one location a call names. DuckDB binds the argument as VARCHAR, so
    /// anything else here is a call this function cannot serve rather than a
    /// location it failed to read.
    fn location_argument(&self, arguments: &[ScalarValue]) -> Result<String, Error> {
        let [argument] = arguments else {
            return Err(Error::Arity {
                function: self.name,
                count: arguments.len(),
            });
        };
        match argument {
            ScalarValue::Utf8(location) => Ok(location.clone()),
            other => Err(Error::NotALocation {
                function: self.name,
                argument: other.to_string(),
            }),
        }
    }
}
