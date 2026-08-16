//! Lifecycle for one embedded, in-process Pivot instance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_delta::DeltaDatastore;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use metastore::{Metastore, UserAuth};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

#[derive(Debug)]
struct EphemeralMetastore;

impl Metastore for EphemeralMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DEFAULT_DATASTORE_NAME
    }

    fn user_auth(&self, _username: &str) -> Option<UserAuth> {
        None
    }

    fn user_names(&self) -> Vec<String> {
        Vec::new()
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
    engine: engine::Engine,
    catalog: Arc<PivotCatalog>,
    dispatch: DispatchOwner,
    /// Keeps the datastore lock held until every dispatch worker has joined.
    datastore: Arc<DeltaDatastore>,
}

impl ShellState {
    fn shutdown(self) {
        let Self {
            engine,
            catalog,
            dispatch,
            datastore,
        } = self;

        catalog.abort();
        drop(engine);
        drop(catalog);
        dispatch.shutdown();
        drop(datastore);
    }
}

/// An embedded Pivot engine over one persistent Delta datastore.
pub struct ShellInstance {
    state: Option<ShellState>,
    data_path: PathBuf,
}

impl ShellInstance {
    /// Open the production CLI instance on all available workers with half of
    /// physical memory assigned to dispatch.
    pub fn open(data_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let workers = dispatch::default_worker_count();
        let memory_bytes = total_memory_bytes() / 2;
        let buffers = (memory_bytes / BUFFER_SIZE).max(1);
        Self::open_with_resources(data_path, workers, buffers)
    }

    /// Open with an explicit dispatch shape. This is useful for embedding and
    /// for small black-box test instances.
    pub fn open_with_resources(
        data_path: impl AsRef<Path>,
        workers: usize,
        buffers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let data_path = data_path.as_ref().to_path_buf();
        let dispatch = DispatchOwner::new(Dispatch::spin_up(workers, buffers, None));
        let datastore = DeltaDatastore::open(&data_path.to_string_lossy(), dispatch.dispatcher())?;
        let metastore: Arc<dyn Metastore> = Arc::new(EphemeralMetastore);
        let catalog = Arc::new(PivotCatalog::new(
            HashMap::from([(
                DEFAULT_DATASTORE_NAME.to_string(),
                datastore.clone() as Arc<dyn Datastore>,
            )]),
            DEFAULT_DATASTORE_NAME.to_string(),
            metastore,
        )?);
        let engine = engine::Engine::new(catalog.clone(), dispatch.dispatcher().clone());
        Ok(Self {
            state: Some(ShellState {
                engine,
                catalog,
                dispatch,
                datastore,
            }),
            data_path,
        })
    }

    pub fn engine(&self) -> &engine::Engine {
        &self
            .state
            .as_ref()
            .expect("shell instance has already shut down")
            .engine
    }

    pub fn data_path(&self) -> &Path {
        &self.data_path
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
