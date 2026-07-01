//! Operator + connection-router for running the pivotdb postgres-wire server on
//! Kubernetes.
//!
//! Architecture: a [`crd::PivotEndpoint`] reconciles (see [`controller`]) into a
//! load-balancer Service fronting a single [`router`] pod. The router accepts
//! psql sessions and, depending on the endpoint mode, either spawns one pivot
//! backend pod per session or routes every session to a shared pod set. The
//! shared pod-building logic lives in [`resources`] so the operator and the
//! router agree on exactly how pods are shaped.

pub mod controller;
pub mod crd;
pub mod resources;
pub mod router;

/// CRD API group.
pub const GROUP: &str = "pivot.epsio.io";

/// Value of the `app.kubernetes.io/managed-by` label on everything we create.
pub const MANAGED_BY: &str = "pivot-operator";

/// Label whose value is the owning endpoint's name.
pub const LABEL_ENDPOINT: &str = "pivot.epsio.io/endpoint";
/// Label distinguishing a `router` pod from a `backend` (pivot) pod.
pub const LABEL_ROLE: &str = "pivot.epsio.io/role";
/// Label recording the endpoint mode on backend pods.
pub const LABEL_MODE: &str = "pivot.epsio.io/mode";
/// Standard managed-by label key.
pub const LABEL_MANAGED_BY: &str = "app.kubernetes.io/managed-by";

/// Port the router listens for psql sessions on, inside its pod.
pub const ROUTER_LISTEN_PORT: i32 = 5432;
/// Port the router serves health/readiness on (must differ from the psql port
/// so probes never trigger a backend spawn).
pub const ROUTER_HEALTH_PORT: i32 = 8080;

/// Env var (set on the operator Deployment) naming the image the operator should
/// use for the router pods it creates. The router binary lives in this same
/// image, so it is normally the operator's own image.
pub const ENV_ROUTER_IMAGE: &str = "PIVOT_ROUTER_IMAGE";

/// Suffix for the per-endpoint router Deployment, ServiceAccount, Role, and
/// RoleBinding. The Service itself takes the bare endpoint name (it is the
/// public address).
pub const ROUTER_SUFFIX: &str = "-router";

/// Name of the router Deployment / ServiceAccount / RBAC objects for an
/// endpoint.
pub fn router_name(endpoint: &str) -> String {
    format!("{endpoint}{ROUTER_SUFFIX}")
}

/// Initialise `tracing` from `RUST_LOG` (defaulting to `info`). Shared by the
/// operator and router binaries.
pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
