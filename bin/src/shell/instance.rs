//! Lifecycle for one embedded, in-process Pivot instance.

use std::collections::HashMap;
use std::sync::Arc;

use catalog::metastore::{Metastore, UserAuth};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_delta::DeltaDatastore;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use object_storage::AmbientExternalStoreFactory;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

#[derive(Debug)]
struct EphemeralMetastore;

impl Metastore for EphemeralMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DEFAULT_DATASTORE_NAME
    }

    fn user_auth(&self, _username: &str) -> Option<UserAuth> {
        None
    }
}

struct DispatchOwner(Option<Dispatch>);

impl DispatchOwner {
    fn new(dispatch: Dispatch) -> Self {
        Self(Some(dispatch))
    }

    fn dispatcher(&self) -> &DataFlowDispatcher {
        self.0.as_ref().unwrap().dispatcher()
    }

    fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(dispatch) = self.0.take() {
            dispatch.exit();
        }
    }
}

impl Drop for DispatchOwner {
    fn drop(&mut self) {
        self.stop();
    }
}

struct ShellState {
    executor: crate::execution::Executor,
    catalog: Arc<PivotCatalog>,
    dispatch: DispatchOwner,
    /// Keeps the datastore lock held until every dispatch worker has joined.
    datastore: Arc<DeltaDatastore>,
}

impl ShellState {
    fn shutdown(self) {
        let Self {
            executor,
            catalog,
            dispatch,
            datastore,
        } = self;

        catalog.abort();
        drop(executor);
        drop(catalog);
        dispatch.shutdown();
        drop(datastore);
    }
}

/// An embedded Pivot executor over one persistent Delta datastore.
pub struct ShellInstance {
    state: Option<ShellState>,
    location: String,
}

impl ShellInstance {
    /// Open the production CLI instance on all available workers with half of
    /// physical memory assigned to dispatch. `location` is a local directory, or
    /// an object-store URI (`s3://bucket/prefix`, `gs://bucket/prefix`) whose
    /// credentials come from the environment.
    pub fn open(location: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let workers = dispatch::default_worker_count();
        let memory_bytes = total_memory_bytes() / 2;
        let buffers = (memory_bytes / BUFFER_SIZE).max(1);
        Self::open_with_resources(location, workers, buffers)
    }

    /// Open with an explicit dispatch shape. This is useful for embedding and
    /// for small black-box test instances.
    pub fn open_with_resources(
        location: &str,
        workers: usize,
        buffers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dispatch = DispatchOwner::new(Dispatch::spin_up(workers, buffers, None));
        let datastore = DeltaDatastore::open(location, dispatch.dispatcher())?;
        let metastore: Arc<dyn Metastore> = Arc::new(EphemeralMetastore);
        let catalog = Arc::new(
            PivotCatalog::new(
                HashMap::from([(
                    DEFAULT_DATASTORE_NAME.to_string(),
                    datastore.clone() as Arc<dyn Datastore>,
                )]),
                DEFAULT_DATASTORE_NAME.to_string(),
                metastore,
            )?
            .with_external_parquet_read_context(
                dispatch.dispatcher(),
                Arc::new(AmbientExternalStoreFactory),
            ),
        );
        let executor =
            crate::execution::Executor::new(catalog.clone(), dispatch.dispatcher().clone());
        Ok(Self {
            state: Some(ShellState {
                executor,
                catalog,
                dispatch,
                datastore,
            }),
            location: location.to_string(),
        })
    }

    pub fn executor(&self) -> &crate::execution::Executor {
        &self
            .state
            .as_ref()
            .expect("shell instance has already shut down")
            .executor
    }

    /// The location this instance was opened on, as it was spelled: a local
    /// directory or an object-store URI.
    pub fn location(&self) -> &str {
        &self.location
    }

    fn shutdown(&mut self) {
        if let Some(state) = self.state.take() {
            state.shutdown();
        }
    }
}

impl Drop for ShellInstance {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn total_memory_bytes() -> usize {
    let bytes = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    )
    .total_memory();
    usize::try_from(bytes).unwrap_or(usize::MAX)
}
