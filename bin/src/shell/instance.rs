//! Lifecycle for one embedded, in-process Pivot instance.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use catalog::metastore::{Metastore, UserAuth};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_delta::DeltaDatastore;
use dispatch::io::DiskCache;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use object_storage::AmbientExternalStoreFactory;

use crate::resources::{DEFAULT_DISK_CACHE_MAX_OBJECTS, DEFAULT_DISK_CACHE_SIZE};

/// Resource budgets for [`ShellInstance::open`]. Every unset field falls back
/// to a machine-derived default, so `OpenOptions::default()` opens the instance
/// a plain `pivot open` does.
#[derive(Debug, Default)]
pub struct OpenOptions {
    /// Buffer pool budget in bytes. Unset assigns half of physical memory. The
    /// budget is checked against the memory actually available, since every
    /// pool slot is faulted in while the workers start.
    pub memory_bytes: Option<u64>,
    /// Number of dispatch worker threads. Unset uses every available core.
    pub workers: Option<usize>,
    /// On-disk cache for remote (object store) reads. Unset disables it; local
    /// datastores are read from the filesystem directly and never cached.
    pub disk_cache: Option<DiskCacheOptions>,
}

/// The disk cache named by [`OpenOptions::disk_cache`].
#[derive(Debug)]
pub struct DiskCacheOptions {
    /// Directory the cached byte ranges are stored in. The contents persist
    /// across sessions.
    pub dir: PathBuf,
    /// Size budget for the cached bytes. Unset uses the server's default.
    pub size_bytes: Option<u64>,
}

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
    /// Open the production CLI instance with the given resource budgets.
    /// `location` is a local directory, or an object-store URI
    /// (`s3://bucket/prefix`, `gs://bucket/prefix`) whose credentials come from
    /// the environment.
    pub fn open(location: &str, options: OpenOptions) -> Result<Self, Box<dyn std::error::Error>> {
        let workers = options
            .workers
            .unwrap_or_else(dispatch::default_worker_count);
        let memory_bytes = match options.memory_bytes {
            Some(bytes) => usize::try_from(bytes).unwrap_or(usize::MAX),
            None => crate::resources::total_memory_bytes() / 2,
        };
        crate::resources::check_pool_fits(memory_bytes)?;
        let buffers = (memory_bytes / BUFFER_SIZE).max(1);
        let disk_cache = match options.disk_cache {
            Some(cache) => Some(crate::resources::open_disk_cache(
                cache.dir,
                cache
                    .size_bytes
                    .unwrap_or(DEFAULT_DISK_CACHE_SIZE.as_bytes()),
                DEFAULT_DISK_CACHE_MAX_OBJECTS,
            )?),
            None => None,
        };
        Self::open_with_dispatch(location, workers, buffers, disk_cache)
    }

    /// Open with an explicit dispatch shape. This is useful for embedding and
    /// for small black-box test instances.
    pub fn open_with_resources(
        location: &str,
        workers: usize,
        buffers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::open_with_dispatch(location, workers, buffers, None)
    }

    fn open_with_dispatch(
        location: &str,
        workers: usize,
        buffers: usize,
        disk_cache: Option<Arc<DiskCache>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dispatch = DispatchOwner::new(Dispatch::spin_up(workers, buffers, disk_cache));
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
