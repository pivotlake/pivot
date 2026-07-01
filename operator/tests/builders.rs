//! Pure unit tests for the resource builders: no cluster required.

use operator::crd::{EndpointMode, PivotEndpoint, PivotEndpointSpec, ResourceSpec};
use operator::resources;

fn endpoint(mode: EndpointMode) -> PivotEndpoint {
    let spec = PivotEndpointSpec {
        mode,
        image: "pivot:test".to_string(),
        port: None,
        command: vec![],
        auto_bind: None,
        service_type: None,
        resources: Some(ResourceSpec {
            cpu: Some("500m".to_string()),
            memory: Some("1Gi".to_string()),
            cpu_limit: None,
            memory_limit: None,
        }),
        args: vec!["--path".to_string(), "gs://bucket/db".to_string()],
        env: vec![],
        idle_timeout_seconds: None,
        reuse_idle_pods: None,
        max_pods: None,
        shared_replicas: None,
    };
    let mut ep = PivotEndpoint::new("demo", spec);
    ep.metadata.namespace = Some("default".to_string());
    ep.metadata.uid = Some("uid-123".to_string());
    ep
}

#[test]
fn backend_args_inject_bind_before_user_args() {
    let ep = endpoint(EndpointMode::PerConnection);

    let args = ep.spec.backend_args();

    assert_eq!(
        args,
        vec!["--bind", "0.0.0.0:5432", "--path", "gs://bucket/db"]
    );
}

#[test]
fn auto_bind_disabled_omits_bind() {
    let mut ep = endpoint(EndpointMode::Shared);
    ep.spec.auto_bind = Some(false);

    let args = ep.spec.backend_args();

    assert!(!args.iter().any(|a| a == "--bind"));
    assert_eq!(args, vec!["--path", "gs://bucket/db"]);
}

#[test]
fn backend_pod_carries_image_labels_owner_and_resources() {
    let ep = endpoint(EndpointMode::PerConnection);

    let pod = resources::build_backend_pod(&ep, "demo-c-1");

    let container = &pod.spec.as_ref().unwrap().containers[0];
    assert_eq!(container.image.as_deref(), Some("pivot:test"));
    let labels = pod.metadata.labels.unwrap();
    assert_eq!(labels.get("pivot.epsio.io/endpoint").unwrap(), "demo");
    assert_eq!(labels.get("pivot.epsio.io/role").unwrap(), "backend");
    assert_eq!(labels.get("pivot.epsio.io/mode").unwrap(), "per-connection");
    let owner = &pod.metadata.owner_references.unwrap()[0];
    assert_eq!(owner.kind, "PivotEndpoint");
    assert_eq!(owner.uid, "uid-123");
    let requests = container
        .resources
        .as_ref()
        .unwrap()
        .requests
        .as_ref()
        .unwrap();
    assert_eq!(requests.get("memory").unwrap().0, "1Gi");
}

#[test]
fn service_defaults_to_load_balancer_and_selects_router_pods() {
    let ep = endpoint(EndpointMode::Shared);

    let svc = resources::build_service(&ep);

    let spec = svc.spec.unwrap();
    assert_eq!(spec.type_.as_deref(), Some("LoadBalancer"));
    let selector = spec.selector.unwrap();
    assert_eq!(selector.get("pivot.epsio.io/role").unwrap(), "router");
    assert_eq!(svc.metadata.name.as_deref(), Some("demo"));
}

#[test]
fn router_deployment_runs_router_for_its_endpoint() {
    let ep = endpoint(EndpointMode::PerConnection);

    let deploy = resources::build_router_deployment(&ep, "pivot-operator:dev");

    let container = &deploy.spec.unwrap().template.spec.unwrap().containers[0];
    assert_eq!(container.command.as_ref().unwrap()[0], "pivot-router");
    let args = container.args.as_ref().unwrap();
    assert!(args.windows(2).any(|w| w == ["--endpoint", "demo"]));
}
