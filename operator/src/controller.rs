//! The operator control loop. Watches [`PivotEndpoint`] objects and reconciles
//! each into a ServiceAccount + Role + RoleBinding (so the router may manage
//! pods), a one-replica router Deployment, and a Service that is the endpoint's
//! public psql address. Everything is created with server-side apply and owned
//! by the endpoint, so updates are idempotent and deletion garbage-collects.

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::{Pod, Service};
use k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition;
use kube::api::{Api, Patch, PatchParams};
use kube::core::NamespaceResourceScope;
use kube::runtime::controller::Action;
use kube::runtime::{Controller, watcher};
use kube::{Client, CustomResourceExt, Resource, ResourceExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tracing::{error, info, warn};

use crate::crd::{PivotEndpoint, PivotEndpointStatus};
use crate::{ENV_ROUTER_IMAGE, resources};

const FIELD_MANAGER: &str = "pivot-operator";

/// Server-side-apply a namespaced object and return the applied result. Used for
/// every child object so the apply shape lives in one place.
async fn apply<K>(client: &Client, namespace: &str, obj: &K) -> Result<K, kube::Error>
where
    K: Resource<Scope = NamespaceResourceScope, DynamicType = ()>
        + Serialize
        + DeserializeOwned
        + Clone
        + Debug,
{
    let pp = PatchParams::apply(FIELD_MANAGER).force();
    Api::<K>::namespaced(client.clone(), namespace)
        .patch(&obj.name_any(), &pp, &Patch::Apply(obj))
        .await
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("kube api error: {0}")]
    Kube(#[from] kube::Error),
    #[error("endpoint has no namespace")]
    NoNamespace,
}

struct Context {
    client: Client,
    /// Image carrying the `pivot-router` binary, stamped onto router pods.
    router_image: String,
}

/// Apply the `PivotEndpoint` CRD to the cluster (idempotent) so a fresh cluster
/// can accept endpoints without a separate `kubectl apply`.
pub async fn ensure_crd(client: Client) -> Result<(), kube::Error> {
    let crd = PivotEndpoint::crd();
    let api: Api<CustomResourceDefinition> = Api::all(client);
    api.patch(
        &crd.name_any(),
        &PatchParams::apply(FIELD_MANAGER).force(),
        &Patch::Apply(&crd),
    )
    .await?;
    info!("PivotEndpoint CRD applied");
    Ok(())
}

/// Resolve the address psql clients use to reach the endpoint, from the live
/// Service. `None` while a LoadBalancer is still provisioning.
fn service_address(svc: &Service, service_type: &str) -> Option<String> {
    let spec = svc.spec.as_ref()?;
    let port = spec.ports.as_ref()?.first()?.port;
    match service_type {
        "LoadBalancer" => {
            let ingress = svc
                .status
                .as_ref()?
                .load_balancer
                .as_ref()?
                .ingress
                .as_ref()?;
            let first = ingress.first()?;
            let host = first.ip.clone().or_else(|| first.hostname.clone())?;
            Some(format!("{host}:{port}"))
        }
        "NodePort" => {
            let node_port = spec.ports.as_ref()?.first()?.node_port?;
            Some(format!("<node-ip>:{node_port}"))
        }
        _ => {
            let cluster_ip = spec.cluster_ip.clone()?;
            Some(format!("{cluster_ip}:{port}"))
        }
    }
}

async fn reconcile(ep: Arc<PivotEndpoint>, ctx: Arc<Context>) -> Result<Action, Error> {
    let ns = ep.namespace().ok_or(Error::NoNamespace)?;
    let name = ep.name_any();
    let client = &ctx.client;

    info!(endpoint = %name, namespace = %ns, mode = ?ep.spec.mode, "reconciling");

    // RBAC so the router can create/delete backend pods and read its endpoint.
    apply(client, &ns, &resources::build_service_account(&ep)).await?;
    apply(client, &ns, &resources::build_role(&ep)).await?;
    apply(client, &ns, &resources::build_role_binding(&ep)).await?;

    // The data-plane router and its public Service. The Service apply returns the
    // server-populated object, so we read the address from it directly.
    apply(
        client,
        &ns,
        &resources::build_router_deployment(&ep, &ctx.router_image),
    )
    .await?;
    let live_svc = apply(client, &ns, &resources::build_service(&ep)).await?;

    // Reflect the resolved address back into status.
    let address = service_address(&live_svc, &ep.spec.service_type_or_default());
    let phase = if address.is_some() {
        "Ready"
    } else {
        "Pending"
    };
    let status = PivotEndpointStatus {
        phase: Some(phase.to_string()),
        address,
        observed_generation: ep.meta().generation,
    };
    Api::<PivotEndpoint>::namespaced(client.clone(), &ns)
        .patch_status(
            &name,
            &PatchParams::default(),
            &Patch::Merge(json!({ "status": status })),
        )
        .await?;

    // Requeue periodically so a late-arriving LB address eventually lands in
    // status even without a Service event.
    Ok(Action::requeue(Duration::from_secs(30)))
}

fn error_policy(ep: Arc<PivotEndpoint>, err: &Error, _ctx: Arc<Context>) -> Action {
    warn!(endpoint = %ep.name_any(), "reconcile failed: {err}");
    Action::requeue(Duration::from_secs(5))
}

/// The image the operator itself runs, discovered from its own pod via the
/// downward-API `POD_NAME`/`POD_NAMESPACE` env vars. Used as the default router
/// image so an install only has to set one image (the operator Deployment's).
async fn detect_own_image(client: &Client) -> Option<String> {
    let name = std::env::var("POD_NAME").ok()?;
    let namespace = std::env::var("POD_NAMESPACE").ok()?;
    let pod = Api::<Pod>::namespaced(client.clone(), &namespace)
        .get(&name)
        .await
        .ok()?;
    let containers = pod.spec?.containers;
    let container = containers
        .iter()
        .find(|c| c.name == "operator")
        .or_else(|| containers.first())?;
    container.image.clone()
}

/// Start the controller and run it until the process is stopped.
pub async fn run(client: Client) -> anyhow::Result<()> {
    // The router runs the operator's own image. Prefer an explicit override, else
    // discover it from our own pod so a normal install sets only one image.
    let router_image = match resources::router_image_from_env() {
        Some(image) => image,
        None => detect_own_image(&client).await.ok_or_else(|| {
            anyhow::anyhow!(
                "could not determine router image: set {ENV_ROUTER_IMAGE}, or run as a pod \
                 with POD_NAME/POD_NAMESPACE (downward API) so it can be auto-detected"
            )
        })?,
    };
    info!(%router_image, "starting pivot-operator controller");

    let endpoints = Api::<PivotEndpoint>::all(client.clone());
    let ctx = Arc::new(Context {
        client: client.clone(),
        router_image,
    });

    Controller::new(endpoints, watcher::Config::default())
        .owns(
            Api::<Deployment>::all(client.clone()),
            watcher::Config::default(),
        )
        .owns(
            Api::<Service>::all(client.clone()),
            watcher::Config::default(),
        )
        .run(reconcile, error_policy, ctx)
        .for_each(|res| async move {
            match res {
                Ok((obj, _)) => info!(endpoint = %obj.name, "reconciled"),
                Err(e) => error!("controller error: {e}"),
            }
        })
        .await;
    Ok(())
}
