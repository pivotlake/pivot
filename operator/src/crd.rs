//! The `PivotEndpoint` custom resource: the user-facing knob for declaring a
//! psql endpoint and how its sessions map onto pivot pods.
//!
//! A `PivotEndpoint` reconciles into a load-balancer Service (the address psql
//! clients connect to) fronting a single connection-router pod. The router then
//! either spawns one pivot pod per psql session (`perConnection`) or routes
//! every session to a shared, long-lived pivot pod set (`shared`).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use kube::CustomResource;

/// How psql sessions arriving at an endpoint map onto pivot pods.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum EndpointMode {
    /// Spawn a dedicated pivot pod for each psql session and tear it down when
    /// the session ends (or after an idle timeout, if pods are reused).
    PerConnection,
    /// Route every psql session to a shared, long-lived set of pivot pods.
    Shared,
}

/// Pod sizing for the spawned pivot pods. Each field maps onto the pivot
/// container's Kubernetes resource requests/limits. Limits default to the
/// matching request when omitted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSpec {
    /// CPU request, e.g. `"500m"` or `"2"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    /// Memory request, e.g. `"512Mi"` or `"4Gi"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
    /// CPU limit. Defaults to `cpu` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_limit: Option<String>,
    /// Memory limit. Defaults to `memory` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<String>,
}

/// A single environment variable injected into the pivot container.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EnvSpec {
    pub name: String,
    pub value: String,
}

#[derive(CustomResource, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[kube(
    group = "pivot.epsio.io",
    version = "v1alpha1",
    kind = "PivotEndpoint",
    plural = "pivotendpoints",
    shortname = "pep",
    namespaced,
    status = "PivotEndpointStatus",
    printcolumn = r#"{"name":"Mode","type":"string","jsonPath":".spec.mode"}"#,
    printcolumn = r#"{"name":"Address","type":"string","jsonPath":".status.address"}"#,
    printcolumn = r#"{"name":"Phase","type":"string","jsonPath":".status.phase"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct PivotEndpointSpec {
    /// Whether each psql session gets its own pivot pod, or all sessions share
    /// a long-lived pod set.
    pub mode: EndpointMode,

    /// Container image for the pivot backend pods (the postgres-wire server).
    pub image: String,

    /// Port the pivot server listens on inside its container. Defaults to 5432.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<i32>,

    /// Override the container entrypoint. Leave unset to use the image's own
    /// entrypoint (the pivot image runs `pivotdb-server`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,

    /// Inject `--bind 0.0.0.0:<port>` so the pivot server accepts connections
    /// from the router over the pod network. Defaults to true. Set false for a
    /// backend image that already listens on all interfaces (e.g. stock
    /// postgres) or takes a different bind flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_bind: Option<bool>,

    /// Kubernetes Service type for the endpoint's address. Defaults to
    /// `LoadBalancer`. Use `NodePort` or `ClusterIP` where a cloud LB is not
    /// available (e.g. a local kind cluster).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_type: Option<String>,

    /// Sizing for each spawned pivot pod.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceSpec>,

    /// Extra command-line arguments appended to the pivot server invocation,
    /// e.g. `["--path", "gs://bucket/db"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,

    /// Extra environment variables for the pivot container.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvSpec>,

    /// (`perConnection` only) Seconds an idle reused pod is kept before it is
    /// torn down. Requires `reuseIdlePods`. Without reuse, pods are deleted
    /// immediately on disconnect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_timeout_seconds: Option<i64>,

    /// (`perConnection` only) Keep a disconnected pod around and hand it to the
    /// next session instead of deleting it, reaping it only after
    /// `idleTimeoutSeconds`. Defaults to false (delete on disconnect).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reuse_idle_pods: Option<bool>,

    /// (`perConnection` only) Cap on concurrently live pivot pods. Sessions
    /// arriving past the cap are rejected. Unset means no cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pods: Option<i32>,

    /// (`shared` only) Number of shared pivot pods to keep ready. Defaults to 1
    /// (all sessions share one pod).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_replicas: Option<i32>,
}

/// Observed state, written back by the operator.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PivotEndpointStatus {
    /// Human-readable lifecycle phase, e.g. `Pending`, `Ready`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// The resolved address psql clients connect to (LB ingress, or
    /// `<host>:<nodePort>` for NodePort), once known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    /// The `.metadata.generation` this status reflects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}

impl PivotEndpointSpec {
    /// Port the pivot server listens on (default 5432).
    pub fn backend_port(&self) -> i32 {
        self.port.unwrap_or(5432)
    }

    /// Service type for the endpoint address (default `LoadBalancer`).
    pub fn service_type_or_default(&self) -> String {
        self.service_type
            .clone()
            .unwrap_or_else(|| "LoadBalancer".to_string())
    }

    /// Number of shared pods to keep ready (default 1). Only meaningful in
    /// `shared` mode.
    pub fn shared_replicas_or_default(&self) -> i32 {
        self.shared_replicas.unwrap_or(1).max(1)
    }

    /// Whether disconnected per-connection pods are reused (default false).
    pub fn reuse_idle(&self) -> bool {
        self.reuse_idle_pods.unwrap_or(false)
    }

    /// Whether to inject `--bind 0.0.0.0:<port>` into the backend args (default
    /// true).
    pub fn auto_bind(&self) -> bool {
        self.auto_bind.unwrap_or(true)
    }

    /// The full argument list passed to the backend container: an optional
    /// injected `--bind`, then the user's `args`. The server sizes its buffer
    /// pool from its cgroup memory limit, so no memory flag is needed.
    pub fn backend_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if self.auto_bind() {
            args.push("--bind".to_string());
            args.push(format!("0.0.0.0:{}", self.backend_port()));
        }
        args.extend(self.args.iter().cloned());
        args
    }
}
