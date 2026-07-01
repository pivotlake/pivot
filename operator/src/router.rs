//! The connection router: the data plane that sits behind an endpoint's Service
//! and turns each psql session into the right pivot pod.
//!
//! It is a transparent TCP proxy. The pivot server does no TLS and no auth, so
//! the router never parses the postgres protocol; it pumps bytes between the
//! client and a backend pod (the client's SSLRequest / startup just flow through
//! to the backend, which answers them). What differs per mode is how a backend
//! is chosen:
//!
//! * `perConnection`: a fresh pivot pod is created for the session and deleted
//!   when it ends. With `reuseIdlePods`, the pod instead returns to an idle pool
//!   and is reaped only after `idleTimeoutSeconds`.
//! * `shared`: the router keeps `sharedReplicas` long-lived pods ready and round
//!   robins sessions across them; pods outlive sessions.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context as _, anyhow, bail};
use clap::Parser;
use k8s_openapi::api::core::v1::Pod;
use kube::api::{Api, DeleteParams, ListParams, PostParams};
use kube::{Client, ResourceExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

use crate::crd::{EndpointMode, PivotEndpoint};
use crate::resources;
use crate::{LABEL_ENDPOINT, LABEL_MODE, LABEL_ROLE};

/// CLI configuration for the router process.
#[derive(Clone, Debug, Parser)]
#[command(name = "pivot-router")]
pub struct RouterConfig {
    /// Name of the PivotEndpoint this router serves.
    #[arg(long)]
    pub endpoint: String,

    /// Namespace the PivotEndpoint and backend pods live in.
    #[arg(long)]
    pub namespace: String,

    /// Address to accept psql sessions on.
    #[arg(long, default_value = "0.0.0.0:5432")]
    pub listen: String,

    /// Address to serve the health endpoint on (must differ from `--listen`).
    #[arg(long, default_value = "0.0.0.0:8080")]
    pub health: String,
}

/// A backend a session is proxied to. `pod_name` is set for per-connection pods
/// the router owns (so it can release them); shared pods are not released.
#[derive(Clone, Debug)]
struct Backend {
    ip: String,
    port: i32,
    pod_name: Option<String>,
}

struct IdlePod {
    ip: String,
    pod_name: String,
    idle_since: Instant,
}

/// Shared router state.
struct Router {
    pods: Api<Pod>,
    endpoint: String,
    /// The endpoint spec, refreshed in the background so spec edits take effect.
    spec: Arc<RwLock<PivotEndpoint>>,
    /// per-connection: pods waiting to be reused.
    idle: Mutex<VecDeque<IdlePod>>,
    /// per-connection: pods currently created (checked out + idle), for max_pods.
    live_pods: AtomicI64,
    /// per-connection: monotonic suffix source for pod names.
    spawn_counter: AtomicU64,
    /// shared: IPs of ready shared pods, refreshed by the reconcile loop.
    shared_ready: RwLock<Vec<String>>,
    /// shared: round-robin cursor.
    shared_cursor: AtomicUsize,
}

/// Entry point: run the router until the process is stopped.
pub async fn run(config: RouterConfig) -> anyhow::Result<()> {
    let client = Client::try_default().await?;
    let pods: Api<Pod> = Api::namespaced(client.clone(), &config.namespace);
    let endpoints: Api<PivotEndpoint> = Api::namespaced(client.clone(), &config.namespace);

    let ep = endpoints
        .get(&config.endpoint)
        .await
        .with_context(|| format!("reading PivotEndpoint {}", config.endpoint))?;
    let mode = ep.spec.mode.clone();
    info!(endpoint = %config.endpoint, ?mode, "router starting");

    let router = Arc::new(Router {
        pods,
        endpoint: config.endpoint.clone(),
        spec: Arc::new(RwLock::new(ep)),
        idle: Mutex::new(VecDeque::new()),
        live_pods: AtomicI64::new(0),
        spawn_counter: AtomicU64::new(0),
        shared_ready: RwLock::new(Vec::new()),
        shared_cursor: AtomicUsize::new(0),
    });

    // Keep the cached spec fresh so `kubectl edit` takes effect without a
    // restart.
    {
        let router = router.clone();
        let endpoints = endpoints.clone();
        let name = config.endpoint.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                if let Ok(ep) = endpoints.get(&name).await {
                    *router.spec.write().await = ep;
                }
            }
        });
    }

    match mode {
        EndpointMode::PerConnection => router.cleanup_stale_per_connection_pods().await,
        EndpointMode::Shared => router.spawn_shared_reconciler(),
    }
    router.spawn_idle_reaper();

    serve_health(&config.health);

    let listener = TcpListener::bind(&config.listen)
        .await
        .with_context(|| format!("binding psql listener {}", config.listen))?;
    info!(listen = %config.listen, "accepting psql sessions");

    loop {
        let (client_sock, peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                warn!("accept failed: {e}");
                continue;
            }
        };
        let router = router.clone();
        tokio::spawn(async move {
            if let Err(e) = router.handle_session(client_sock).await {
                warn!(%peer, "session ended with error: {e}");
            }
        });
    }
}

impl Router {
    /// Current spec snapshot.
    async fn spec(&self) -> PivotEndpoint {
        self.spec.read().await.clone()
    }

    /// Handle one psql session: pick a backend, proxy bytes, then release.
    async fn handle_session(self: Arc<Self>, mut client_sock: TcpStream) -> anyhow::Result<()> {
        let ep = self.spec().await;
        let backend = match ep.spec.mode {
            EndpointMode::PerConnection => self.acquire_per_connection(&ep).await?,
            EndpointMode::Shared => self.acquire_shared(&ep).await?,
        };

        let addr = format!("{}:{}", backend.ip, backend.port);
        let mut backend_sock = match TcpStream::connect(&addr).await {
            Ok(s) => s,
            Err(e) => {
                // A per-connection pod we just made is unusable; drop it.
                self.release(&ep, backend, false).await;
                return Err(anyhow!("connecting to backend {addr}: {e}"));
            }
        };
        info!(backend = %addr, "session connected to backend");

        let pump = tokio::io::copy_bidirectional(&mut client_sock, &mut backend_sock).await;
        // Reuse the pod only on a clean close, so a crashed backend is not pooled.
        let keep = pump.is_ok();
        if let Err(e) = pump {
            warn!(backend = %addr, "proxy copy error: {e}");
        }
        self.release(&ep, backend, keep).await;
        Ok(())
    }

    // ---- per-connection mode -------------------------------------------------

    async fn acquire_per_connection(&self, ep: &PivotEndpoint) -> anyhow::Result<Backend> {
        if ep.spec.reuse_idle()
            && let Some(idle) = self.idle.lock().await.pop_front()
        {
            info!(pod = %idle.pod_name, "reusing idle pod");
            return Ok(Backend {
                ip: idle.ip,
                port: ep.spec.backend_port(),
                pod_name: Some(idle.pod_name),
            });
        }

        if let Some(max) = ep.spec.max_pods
            && self.live_pods.load(Ordering::SeqCst) >= max as i64
        {
            bail!("max_pods ({max}) reached; rejecting session");
        }

        self.live_pods.fetch_add(1, Ordering::SeqCst);
        match self.spawn_pod(ep).await {
            Ok(backend) => Ok(backend),
            Err(e) => {
                self.live_pods.fetch_sub(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    /// Create a fresh pivot pod and wait until it accepts connections.
    async fn spawn_pod(&self, ep: &PivotEndpoint) -> anyhow::Result<Backend> {
        let suffix = self.spawn_counter.fetch_add(1, Ordering::SeqCst);
        let rnd: u32 = rand::random();
        let pod_name = format!("{}-c-{:x}-{:x}", self.endpoint, suffix, rnd);
        let pod = resources::build_backend_pod(ep, &pod_name);

        self.pods
            .create(&PostParams::default(), &pod)
            .await
            .with_context(|| format!("creating pod {pod_name}"))?;
        info!(pod = %pod_name, "spawned per-connection pod");

        let port = ep.spec.backend_port();
        let deadline = Instant::now() + Duration::from_secs(180);
        match self.wait_pod_ready(&pod_name, port, deadline).await {
            Ok(ip) => Ok(Backend {
                ip,
                port,
                pod_name: Some(pod_name),
            }),
            Err(e) => {
                let _ = self.pods.delete(&pod_name, &DeleteParams::default()).await;
                Err(e)
            }
        }
    }

    /// Release a backend after its session ends. `keep` requests reuse; it only
    /// applies to per-connection pods when `reuseIdlePods` is on.
    async fn release(&self, ep: &PivotEndpoint, backend: Backend, keep: bool) {
        let Some(pod_name) = backend.pod_name else {
            return; // shared pod: nothing to do.
        };
        if keep && ep.spec.reuse_idle() {
            self.idle.lock().await.push_back(IdlePod {
                ip: backend.ip,
                pod_name,
                idle_since: Instant::now(),
            });
            return;
        }
        self.delete_pod(&pod_name).await;
        self.live_pods.fetch_sub(1, Ordering::SeqCst);
    }

    async fn delete_pod(&self, pod_name: &str) {
        match self.pods.delete(pod_name, &DeleteParams::default()).await {
            Ok(_) => info!(pod = %pod_name, "deleted pod"),
            Err(e) => warn!(pod = %pod_name, "failed to delete pod: {e}"),
        }
    }

    /// Reap idle pods past their idle timeout. No-op unless reuse is on and a
    /// timeout is set.
    fn spawn_idle_reaper(self: &Arc<Self>) {
        let router = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let ep = router.spec().await;
                let Some(timeout) = ep.spec.idle_timeout_seconds else {
                    continue;
                };
                if !ep.spec.reuse_idle() {
                    continue;
                }
                let timeout = Duration::from_secs(timeout.max(0) as u64);
                let mut to_delete = Vec::new();
                {
                    let mut idle = router.idle.lock().await;
                    let mut kept = VecDeque::new();
                    while let Some(pod) = idle.pop_front() {
                        if pod.idle_since.elapsed() >= timeout {
                            to_delete.push(pod.pod_name);
                        } else {
                            kept.push_back(pod);
                        }
                    }
                    *idle = kept;
                }
                for name in to_delete {
                    router.delete_pod(&name).await;
                    router.live_pods.fetch_sub(1, Ordering::SeqCst);
                }
            }
        });
    }

    /// On startup, delete leftover per-connection pods from a previous router
    /// incarnation: they are in-memory and have no live client.
    async fn cleanup_stale_per_connection_pods(&self) {
        let selector = format!(
            "{LABEL_ENDPOINT}={},{LABEL_ROLE}=backend,{LABEL_MODE}={}",
            self.endpoint,
            resources::mode_label(&EndpointMode::PerConnection),
        );
        let lp = ListParams::default().labels(&selector);
        match self.pods.list(&lp).await {
            Ok(list) => {
                for pod in list {
                    self.delete_pod(&pod.name_any()).await;
                }
            }
            Err(e) => warn!("listing stale pods failed: {e}"),
        }
    }

    // ---- shared mode ---------------------------------------------------------

    async fn acquire_shared(&self, ep: &PivotEndpoint) -> anyhow::Result<Backend> {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            {
                let ready = self.shared_ready.read().await;
                if !ready.is_empty() {
                    let idx = self.shared_cursor.fetch_add(1, Ordering::SeqCst) % ready.len();
                    return Ok(Backend {
                        ip: ready[idx].clone(),
                        port: ep.spec.backend_port(),
                        pod_name: None,
                    });
                }
            }
            if Instant::now() > deadline {
                bail!("no ready shared pods after timeout");
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }

    /// Keep `sharedReplicas` shared pods alive and publish their ready IPs.
    fn spawn_shared_reconciler(self: &Arc<Self>) {
        let router = self.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = router.reconcile_shared().await {
                    warn!("shared reconcile failed: {e}");
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
    }

    async fn reconcile_shared(&self) -> anyhow::Result<()> {
        let ep = self.spec().await;
        let replicas = ep.spec.shared_replicas_or_default();
        let mut ready = Vec::new();
        for i in 0..replicas {
            let pod_name = format!("{}-shared-{i}", self.endpoint);
            match self.pods.get_opt(&pod_name).await? {
                Some(pod) => {
                    if let Some(ip) = ready_pod_ip(&pod) {
                        ready.push(ip);
                    }
                }
                None => {
                    let pod = resources::build_backend_pod(&ep, &pod_name);
                    match self.pods.create(&PostParams::default(), &pod).await {
                        Ok(_) => info!(pod = %pod_name, "created shared pod"),
                        Err(e) => warn!(pod = %pod_name, "creating shared pod failed: {e}"),
                    }
                }
            }
        }
        *self.shared_ready.write().await = ready;
        Ok(())
    }

    // ---- readiness -----------------------------------------------------------

    /// Poll a pod until it has an IP and accepts a TCP connection on `port`.
    async fn wait_pod_ready(
        &self,
        pod_name: &str,
        port: i32,
        deadline: Instant,
    ) -> anyhow::Result<String> {
        loop {
            if let Ok(pod) = self.pods.get(pod_name).await
                && let Some(ip) = pod.status.as_ref().and_then(|s| s.pod_ip.clone())
                && TcpStream::connect(format!("{ip}:{port}")).await.is_ok()
            {
                return Ok(ip);
            }
            if Instant::now() > deadline {
                bail!("pod {pod_name} not ready before deadline");
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
}

/// IP of a pod that is Running with all containers ready, else `None`.
fn ready_pod_ip(pod: &Pod) -> Option<String> {
    let status = pod.status.as_ref()?;
    let ip = status.pod_ip.clone()?;
    let all_ready = status
        .container_statuses
        .as_ref()
        .map(|cs| !cs.is_empty() && cs.iter().all(|c| c.ready))
        .unwrap_or(false);
    all_ready.then_some(ip)
}

/// A tiny always-200 health endpoint on its own port, so Kubernetes probes never
/// touch the psql port (which would spawn a backend pod).
fn serve_health(addr: &str) {
    let addr = addr.to_string();
    tokio::spawn(async move {
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                error!("health listener bind {addr} failed: {e}");
                return;
            }
        };
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let body = "ok";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
}
