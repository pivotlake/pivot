//! Builders that turn a [`PivotEndpoint`] into the
//! Kubernetes objects that implement it. The operator uses the Service / RBAC /
//! Deployment builders; the router uses [`build_backend_pod`]. Keeping them in
//! one module means the two processes agree on names, labels, and pod shape.

use std::collections::BTreeMap;

use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, EnvVar, HTTPGetAction, Pod, PodSpec, PodTemplateSpec, Probe,
    ResourceRequirements, Service, ServiceAccount, ServicePort, ServiceSpec, TCPSocketAction,
};
use k8s_openapi::api::rbac::v1::{PolicyRule, Role, RoleBinding, RoleRef, Subject};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, OwnerReference};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::api::ObjectMeta;
use kube::{Resource, ResourceExt};

use crate::crd::{EndpointMode, PivotEndpoint};
use crate::{
    ENV_ROUTER_IMAGE, LABEL_ENDPOINT, LABEL_MANAGED_BY, LABEL_MODE, LABEL_ROLE, MANAGED_BY,
    ROUTER_HEALTH_PORT, ROUTER_LISTEN_PORT, router_name,
};

/// String form of the mode, used in labels and CLI args.
pub fn mode_label(mode: &EndpointMode) -> &'static str {
    match mode {
        EndpointMode::PerConnection => "per-connection",
        EndpointMode::Shared => "shared",
    }
}

/// The endpoint's name. Reconcile never runs on an object without a name, so the
/// `name_any` fallback only guards builder unit tests.
fn endpoint_name(ep: &PivotEndpoint) -> String {
    ep.name_any()
}

fn namespace(ep: &PivotEndpoint) -> String {
    ep.namespace().unwrap_or_else(|| "default".to_string())
}

/// `&[&str]` to `Vec<String>`, for the small RBAC string lists.
fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// OwnerReference pointing at the endpoint, so deleting the endpoint garbage
/// collects everything we create for it. `controller` is set on the operator's
/// own children (Deployment/Service/RBAC) so the controller's owned-object watch
/// picks them up; backend pods set it false to avoid a controller conflict with
/// the router that actually manages them. The apiVersion/kind come from the
/// derived `Resource` impl so they can't drift from the CRD definition.
pub fn owner_reference(ep: &PivotEndpoint, controller: bool) -> OwnerReference {
    OwnerReference {
        api_version: PivotEndpoint::api_version(&()).into_owned(),
        kind: PivotEndpoint::kind(&()).into_owned(),
        name: ep.name_any(),
        uid: ep.uid().unwrap_or_default(),
        controller: Some(controller),
        block_owner_deletion: Some(false),
    }
}

/// Labels shared by every object belonging to an endpoint.
fn base_labels(endpoint: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (LABEL_MANAGED_BY.to_string(), MANAGED_BY.to_string()),
        (LABEL_ENDPOINT.to_string(), endpoint.to_string()),
    ])
}

/// Labels (and selector) for the router pods of an endpoint.
pub fn router_labels(endpoint: &str) -> BTreeMap<String, String> {
    let mut labels = base_labels(endpoint);
    labels.insert(LABEL_ROLE.to_string(), "router".to_string());
    labels
}

/// Labels for the backend (pivot) pods of an endpoint.
pub fn backend_labels(endpoint: &str, mode: &EndpointMode) -> BTreeMap<String, String> {
    let mut labels = base_labels(endpoint);
    labels.insert(LABEL_ROLE.to_string(), "backend".to_string());
    labels.insert(LABEL_MODE.to_string(), mode_label(mode).to_string());
    labels
}

/// Translate the endpoint's [`ResourceSpec`](crate::crd::ResourceSpec) into a
/// Kubernetes `ResourceRequirements`. Limits default to the matching request.
fn resource_requirements(ep: &PivotEndpoint) -> Option<ResourceRequirements> {
    let spec = ep.spec.resources.as_ref()?;
    let mut requests = BTreeMap::new();
    let mut limits = BTreeMap::new();
    if let Some(cpu) = &spec.cpu {
        requests.insert("cpu".to_string(), Quantity(cpu.clone()));
    }
    if let Some(mem) = &spec.memory {
        requests.insert("memory".to_string(), Quantity(mem.clone()));
    }
    let cpu_limit = spec.cpu_limit.as_ref().or(spec.cpu.as_ref());
    if let Some(cpu) = cpu_limit {
        limits.insert("cpu".to_string(), Quantity(cpu.clone()));
    }
    let mem_limit = spec.memory_limit.as_ref().or(spec.memory.as_ref());
    if let Some(mem) = mem_limit {
        limits.insert("memory".to_string(), Quantity(mem.clone()));
    }
    if requests.is_empty() && limits.is_empty() {
        return None;
    }
    Some(ResourceRequirements {
        requests: (!requests.is_empty()).then_some(requests),
        limits: (!limits.is_empty()).then_some(limits),
        ..Default::default()
    })
}

/// Build a single backend (pivot) pod for an endpoint. `pod_name` must be a
/// valid DNS-1123 label and unique within the namespace. The router calls this
/// for both per-connection and shared pods.
pub fn build_backend_pod(ep: &PivotEndpoint, pod_name: &str) -> Pod {
    let endpoint = endpoint_name(ep);
    let port = ep.spec.backend_port();

    let env: Vec<EnvVar> = ep
        .spec
        .env
        .iter()
        .map(|e| EnvVar {
            name: e.name.clone(),
            value: Some(e.value.clone()),
            ..Default::default()
        })
        .collect();

    let container = Container {
        name: "pivot".to_string(),
        image: Some(ep.spec.image.clone()),
        image_pull_policy: Some("IfNotPresent".to_string()),
        command: (!ep.spec.command.is_empty()).then(|| ep.spec.command.clone()),
        args: Some(ep.spec.backend_args()),
        env: (!env.is_empty()).then_some(env),
        ports: Some(vec![ContainerPort {
            container_port: port,
            name: Some("pgwire".to_string()),
            ..Default::default()
        }]),
        resources: resource_requirements(ep),
        readiness_probe: Some(Probe {
            tcp_socket: Some(TCPSocketAction {
                port: IntOrString::Int(port),
                ..Default::default()
            }),
            period_seconds: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    };

    Pod {
        metadata: ObjectMeta {
            name: Some(pod_name.to_string()),
            namespace: Some(namespace(ep)),
            labels: Some(backend_labels(&endpoint, &ep.spec.mode)),
            owner_references: Some(vec![owner_reference(ep, false)]),
            ..Default::default()
        },
        spec: Some(PodSpec {
            containers: vec![container],
            // Per-connection pods are ephemeral and in-memory; a tight grace
            // period frees them quickly. The server handles SIGINT, not SIGTERM,
            // so a clean drain would need a preStop hook; for ephemeral pods the
            // default SIGTERM-then-SIGKILL is acceptable.
            termination_grace_period_seconds: Some(10),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The ServiceAccount the router pod runs as.
pub fn build_service_account(ep: &PivotEndpoint) -> ServiceAccount {
    let endpoint = endpoint_name(ep);
    ServiceAccount {
        metadata: ObjectMeta {
            name: Some(router_name(&endpoint)),
            namespace: Some(namespace(ep)),
            labels: Some(router_labels(&endpoint)),
            owner_references: Some(vec![owner_reference(ep, true)]),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Role granting the router exactly what it needs: manage backend pods and read
/// its own PivotEndpoint.
pub fn build_role(ep: &PivotEndpoint) -> Role {
    let endpoint = endpoint_name(ep);
    Role {
        metadata: ObjectMeta {
            name: Some(router_name(&endpoint)),
            namespace: Some(namespace(ep)),
            labels: Some(router_labels(&endpoint)),
            owner_references: Some(vec![owner_reference(ep, true)]),
            ..Default::default()
        },
        rules: Some(vec![
            PolicyRule {
                api_groups: Some(strings(&[""])),
                resources: Some(strings(&["pods"])),
                verbs: strings(&["get", "list", "watch", "create", "delete"]),
                ..Default::default()
            },
            PolicyRule {
                api_groups: Some(strings(&[crate::GROUP])),
                resources: Some(strings(&["pivotendpoints"])),
                verbs: strings(&["get", "list", "watch"]),
                ..Default::default()
            },
        ]),
    }
}

/// Bind the router Role to the router ServiceAccount.
pub fn build_role_binding(ep: &PivotEndpoint) -> RoleBinding {
    let endpoint = endpoint_name(ep);
    let name = router_name(&endpoint);
    RoleBinding {
        metadata: ObjectMeta {
            name: Some(name.clone()),
            namespace: Some(namespace(ep)),
            labels: Some(router_labels(&endpoint)),
            owner_references: Some(vec![owner_reference(ep, true)]),
            ..Default::default()
        },
        role_ref: RoleRef {
            api_group: "rbac.authorization.k8s.io".to_string(),
            kind: "Role".to_string(),
            name: name.clone(),
        },
        subjects: Some(vec![Subject {
            kind: "ServiceAccount".to_string(),
            name: name.clone(),
            namespace: Some(namespace(ep)),
            ..Default::default()
        }]),
    }
}

/// The router Deployment (one replica). `router_image` is the image carrying the
/// `pivot-router` binary (normally the operator's own image).
pub fn build_router_deployment(ep: &PivotEndpoint, router_image: &str) -> Deployment {
    let endpoint = endpoint_name(ep);
    let name = router_name(&endpoint);
    let labels = router_labels(&endpoint);

    let container = Container {
        name: "router".to_string(),
        image: Some(router_image.to_string()),
        image_pull_policy: Some("IfNotPresent".to_string()),
        command: Some(vec!["pivot-router".to_string()]),
        args: Some(vec![
            "--endpoint".to_string(),
            endpoint.clone(),
            "--namespace".to_string(),
            namespace(ep),
            "--listen".to_string(),
            format!("0.0.0.0:{ROUTER_LISTEN_PORT}"),
            "--health".to_string(),
            format!("0.0.0.0:{ROUTER_HEALTH_PORT}"),
        ]),
        ports: Some(vec![
            ContainerPort {
                container_port: ROUTER_LISTEN_PORT,
                name: Some("pgwire".to_string()),
                ..Default::default()
            },
            ContainerPort {
                container_port: ROUTER_HEALTH_PORT,
                name: Some("health".to_string()),
                ..Default::default()
            },
        ]),
        readiness_probe: Some(Probe {
            http_get: Some(HTTPGetAction {
                path: Some("/healthz".to_string()),
                port: IntOrString::Int(ROUTER_HEALTH_PORT),
                ..Default::default()
            }),
            period_seconds: Some(2),
            ..Default::default()
        }),
        ..Default::default()
    };

    Deployment {
        metadata: ObjectMeta {
            name: Some(name.clone()),
            namespace: Some(namespace(ep)),
            labels: Some(labels.clone()),
            owner_references: Some(vec![owner_reference(ep, true)]),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(1),
            selector: LabelSelector {
                match_labels: Some(labels.clone()),
                ..Default::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels.clone()),
                    ..Default::default()
                }),
                spec: Some(PodSpec {
                    service_account_name: Some(name.clone()),
                    containers: vec![container],
                    ..Default::default()
                }),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The endpoint's public Service: the address psql clients connect to. Selects
/// the router pods and forwards the psql port to them.
pub fn build_service(ep: &PivotEndpoint) -> Service {
    let endpoint = endpoint_name(ep);
    // The selector must match the router pod labels exactly; build once and share
    // so that invariant is explicit.
    let labels = router_labels(&endpoint);
    Service {
        metadata: ObjectMeta {
            name: Some(endpoint.clone()),
            namespace: Some(namespace(ep)),
            labels: Some(labels.clone()),
            owner_references: Some(vec![owner_reference(ep, true)]),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            type_: Some(ep.spec.service_type_or_default()),
            selector: Some(labels),
            ports: Some(vec![ServicePort {
                name: Some("pgwire".to_string()),
                port: ROUTER_LISTEN_PORT,
                target_port: Some(IntOrString::Int(ROUTER_LISTEN_PORT)),
                ..Default::default()
            }]),
            ..Default::default()
        }),
        status: None,
    }
}

/// Read the router image the operator should stamp onto router Deployments, from
/// its own environment ([`ENV_ROUTER_IMAGE`]).
pub fn router_image_from_env() -> Option<String> {
    std::env::var(ENV_ROUTER_IMAGE)
        .ok()
        .filter(|s| !s.is_empty())
}
