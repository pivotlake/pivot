//! End-to-end test that stands up a real Kubernetes cluster (via `kind`),
//! deploys the operator, and drives both endpoint modes through real psql
//! connections.
//!
//! It is heavy (builds an image, creates a cluster) so it only runs when
//! `PIVOT_E2E=1` is set:
//!
//!   PIVOT_E2E=1 cargo test --test e2e -- --nocapture
//!
//! Requirements on PATH: `docker` (running), `kind`, `kubectl`, plus the
//! `arrow-rs` / `duckdb` submodules checked out (the image build compiles the
//! pivot server). The cluster is deleted at the end unless `PIVOT_E2E_KEEP=1`.
//!
//! The backend pods run the real pivot server image (`pivotdb-server`); the
//! operator and router run from the separate, lighter operator image. Sessions
//! exercise the full postgres-wire path into pivot (`SELECT 1` over the simple
//! query protocol pivot implements).

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CLUSTER: &str = "pivot-e2e";
const KCTX: &str = "kind-pivot-e2e";
// The operator and router share the lighter operator image; the backend pods run
// the separate pivot server image.
const OPERATOR_IMAGE: &str = "pivot-operator:e2e";
const PIVOT_IMAGE: &str = "pivot:e2e";

/// Run a command, returning stdout on success and panicking with full output on
/// failure.
fn run(program: &str, args: &[&str]) -> String {
    let out = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn `{program} {}`: {e}", args.join(" ")));
    if !out.status.success() {
        panic!(
            "`{program} {}` failed ({}):\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run a command, returning whether it succeeded (no panic on failure).
fn run_ok(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn kubectl(args: &[&str]) -> String {
    let mut full = vec!["--context", KCTX];
    full.extend_from_slice(args);
    run("kubectl", &full)
}

fn kubectl_ok(args: &[&str]) -> bool {
    let mut full = vec!["--context", KCTX];
    full.extend_from_slice(args);
    run_ok("kubectl", &full)
}

/// Apply a YAML document from a string.
fn kubectl_apply(yaml: &str) {
    use std::io::Write;
    let mut child = Command::new("kubectl")
        .args(["--context", KCTX, "apply", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn kubectl apply");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(yaml.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "kubectl apply failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Poll `predicate` until it returns true or the timeout elapses.
fn wait_until(what: &str, timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    loop {
        if predicate() {
            return;
        }
        if Instant::now() > deadline {
            panic!("timed out after {timeout:?} waiting for: {what}");
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Count backend pods for an endpoint.
fn backend_pod_count(endpoint: &str) -> usize {
    let selector = format!("pivot.epsio.io/endpoint={endpoint},pivot.epsio.io/role=backend");
    let out = kubectl(&["get", "pods", "-l", &selector, "-o", "name"]);
    out.lines().filter(|l| !l.trim().is_empty()).count()
}

/// A `kubectl port-forward` child that is killed on drop.
struct PortForward {
    child: Child,
    local_port: u16,
}

impl PortForward {
    fn to_service(endpoint: &str, local_port: u16) -> Self {
        let child = Command::new("kubectl")
            .args([
                "--context",
                KCTX,
                "port-forward",
                &format!("svc/{endpoint}"),
                &format!("{local_port}:5432"),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn kubectl port-forward");
        // Give the forward a moment to establish.
        std::thread::sleep(Duration::from_secs(3));
        PortForward { child, local_port }
    }
}

impl Drop for PortForward {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// An open psql session through the router. Dropping it closes the connection.
struct Session {
    client: tokio_postgres::Client,
    _conn_task: tokio::task::JoinHandle<()>,
}

impl Session {
    /// Open a session to a local forwarded port, retrying the connect since a
    /// per-connection backend may take a while to spawn.
    async fn open(local_port: u16) -> Session {
        let conn_str = format!(
            "host=127.0.0.1 port={local_port} user=postgres dbname=postgres connect_timeout=180"
        );
        let deadline = Instant::now() + Duration::from_secs(200);
        loop {
            match tokio_postgres::connect(&conn_str, tokio_postgres::NoTls).await {
                Ok((client, connection)) => {
                    let conn_task = tokio::spawn(async move {
                        let _ = connection.await;
                    });
                    return Session {
                        client,
                        _conn_task: conn_task,
                    };
                }
                Err(e) => {
                    if Instant::now() > deadline {
                        panic!("could not connect within timeout: {e}");
                    }
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
        }
    }

    /// Run `SELECT 1` over the simple query protocol (the only protocol pivot
    /// implements) and return column 0 as text.
    async fn select_one(&self) -> String {
        use tokio_postgres::SimpleQueryMessage;
        let msgs = self
            .client
            .simple_query("SELECT 1")
            .await
            .expect("SELECT 1 query");
        for msg in msgs {
            if let SimpleQueryMessage::Row(row) = msg {
                return row.get(0).expect("column 0").to_string();
            }
        }
        panic!("no row returned for SELECT 1");
    }
}

/// PivotEndpoint YAML for the test. ClusterIP + port-forward, since kind has no
/// LoadBalancer. The backend is the real pivot server: `autoBind` (default)
/// injects `--bind 0.0.0.0:5432`, and the server sizes its buffer pool from the
/// pod's cgroup memory limit (the `resources.memory` limit below).
fn endpoint_yaml(name: &str, extra_spec: &str) -> String {
    format!(
        "apiVersion: pivot.epsio.io/v1alpha1
kind: PivotEndpoint
metadata:
  name: {name}
  namespace: default
spec:
  image: {PIVOT_IMAGE}
  serviceType: ClusterIP
  resources:
    memory: 1Gi
{extra_spec}
"
    )
}

fn ensure_cluster() {
    let clusters = run("kind", &["get", "clusters"]);
    if !clusters.lines().any(|l| l.trim() == CLUSTER) {
        eprintln!("creating kind cluster {CLUSTER}");
        run("kind", &["create", "cluster", "--name", CLUSTER]);
    } else {
        eprintln!("reusing existing kind cluster {CLUSTER}");
    }
}

fn build_and_load_images() {
    // The operator/router image is self-contained (no submodules); build it from
    // this crate directory.
    eprintln!("building operator image");
    run(
        "docker",
        &["build", "-t", OPERATOR_IMAGE, "-f", "Dockerfile", "."],
    );
    // The pivot server image builds from server/Dockerfile with the workspace
    // root as context (one level up), which holds the server crate and its
    // arrow-rs / duckdb submodules; this compiles DuckDB and is slow first time.
    eprintln!("building pivot server image; compiles DuckDB, slow first time");
    run(
        "docker",
        &[
            "build",
            "-t",
            PIVOT_IMAGE,
            "-f",
            "../server/Dockerfile",
            "..",
        ],
    );
    eprintln!("loading images into kind");
    for image in [OPERATOR_IMAGE, PIVOT_IMAGE] {
        run("kind", &["load", "docker-image", image, "--name", CLUSTER]);
    }
}

fn deploy_operator() {
    kubectl(&["apply", "-f", "deploy/crd.yaml"]);
    kubectl(&["apply", "-f", "deploy/operator.yaml"]);
    // Point the operator at the image we built and loaded, overriding the
    // manifest's placeholder tag. The router pods inherit it via the operator's
    // own-image auto-detection (no PIVOT_ROUTER_IMAGE needed).
    kubectl(&[
        "-n",
        "pivot-system",
        "set",
        "image",
        "deployment/pivot-operator",
        &format!("operator={OPERATOR_IMAGE}"),
    ]);
    eprintln!("waiting for operator rollout");
    run(
        "kubectl",
        &[
            "--context",
            KCTX,
            "-n",
            "pivot-system",
            "rollout",
            "status",
            "deploy/pivot-operator",
            "--timeout=180s",
        ],
    );
}

/// Wait for the operator to reconcile an endpoint's router Deployment to ready.
fn wait_router_ready(endpoint: &str) {
    let router = format!("{endpoint}-router");
    wait_until(
        &format!("router deployment {router} exists"),
        Duration::from_secs(60),
        || kubectl_ok(&["get", "deploy", &router]),
    );
    run(
        "kubectl",
        &[
            "--context",
            KCTX,
            "rollout",
            "status",
            &format!("deploy/{router}"),
            "--timeout=180s",
        ],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_e2e() {
    if std::env::var("PIVOT_E2E").as_deref() != Ok("1") {
        eprintln!("skipping e2e: set PIVOT_E2E=1 to run (needs docker + kind + kubectl)");
        return;
    }

    ensure_cluster();
    build_and_load_images();
    deploy_operator();

    per_connection_scenario().await;
    shared_scenario().await;

    if std::env::var("PIVOT_E2E_KEEP").as_deref() != Ok("1") {
        eprintln!("deleting kind cluster {CLUSTER}");
        let _ = run_ok("kind", &["delete", "cluster", "--name", CLUSTER]);
    }
}

/// A session spawns its own backend pod, which is torn down on disconnect.
async fn per_connection_scenario() {
    let name = "e2e-perconn";
    eprintln!("== per-connection scenario ==");
    kubectl_apply(&endpoint_yaml(name, "  mode: perConnection\n  maxPods: 5"));
    wait_router_ready(name);

    assert_eq!(
        backend_pod_count(name),
        0,
        "no backend pods before any session"
    );

    let pf = PortForward::to_service(name, 15500);
    let session = Session::open(pf.local_port).await;
    assert_eq!(
        session.select_one().await,
        "1",
        "SELECT 1 routed through a spawned backend"
    );

    // While the session is open, exactly one backend pod must exist for it.
    assert_eq!(
        backend_pod_count(name),
        1,
        "a dedicated backend pod exists during the session"
    );

    // Closing the session tears the pod down.
    drop(session);
    drop(pf);
    wait_until(
        "per-connection backend torn down",
        Duration::from_secs(60),
        || backend_pod_count(name) == 0,
    );
    eprintln!("per-connection: backend spawned and torn down OK");
}

/// All sessions share one long-lived backend pod that outlives them.
async fn shared_scenario() {
    let name = "e2e-shared";
    eprintln!("== shared scenario ==");
    kubectl_apply(&endpoint_yaml(name, "  mode: shared\n  sharedReplicas: 1"));
    wait_router_ready(name);

    // The shared pod is created by the router proactively.
    wait_until("shared backend pod ready", Duration::from_secs(120), || {
        backend_pod_count(name) == 1
    });

    let pf = PortForward::to_service(name, 15501);
    let first = Session::open(pf.local_port).await;
    assert_eq!(first.select_one().await, "1", "first shared session");
    drop(first);
    let second = Session::open(pf.local_port).await;
    assert_eq!(second.select_one().await, "1", "second shared session");
    drop(second);

    // The shared pod persists across both sessions and after they close.
    assert_eq!(
        backend_pod_count(name),
        1,
        "exactly one shared pod serves all sessions"
    );
    drop(pf);
    std::thread::sleep(Duration::from_secs(5));
    assert_eq!(
        backend_pod_count(name),
        1,
        "shared pod outlives its sessions"
    );
    eprintln!("shared: single pod served all sessions OK");
}
