# operator

A Kubernetes operator for running the pivotdb postgres-wire [`server`](../server)
on Kubernetes, with per-session or shared pod routing.

## Architecture

A `PivotEndpoint` custom resource declares a psql endpoint and how its sessions
map onto pivot pods. The operator reconciles each endpoint into:

```
            psql clients
                 │
        ┌────────▼─────────┐   Service (LoadBalancer / NodePort / ClusterIP)
        │  <endpoint> svc  │   the address clients connect to
        └────────┬─────────┘
                 │ selects
        ┌────────▼─────────┐   Deployment <endpoint>-router (1 replica)
        │   pivot-router   │   transparent TCP proxy
        └────────┬─────────┘
                 │ creates / routes to (using its own RBAC)
   ┌─────────────┼──────────────┐
   ▼             ▼              ▼
 pivot pod   pivot pod   ...  pivot pod      backend pods (the server)
```

* **`perConnection`**: the router spawns a fresh pivot pod for each psql session
  and deletes it when the session ends. With `reuseIdlePods`, a disconnected pod
  returns to an idle pool and is reaped only after `idleTimeoutSeconds`.
* **`shared`**: the router keeps `sharedReplicas` long-lived pivot pods ready and
  round-robins sessions across them; pods outlive sessions. `sharedReplicas: 1`
  means every session shares one pod.

The router is a transparent TCP proxy: the pivot server does no TLS and no auth,
so the client's startup/SSLRequest just flow through to the chosen backend. The
operator never bakes pivot specifics into the proxy, so `spec.image` can be any
postgres-wire server.

There are two images. The **operator image** ([`operator/Dockerfile`](Dockerfile))
carries the operator's three binaries: `pivot-operator` (the controller),
`pivot-router` (the data plane), and `pivot-crdgen` (prints the CRD). It is
self-contained (no submodules). The **pivot server image**
([`server/Dockerfile`](../server/Dockerfile)) carries `pivotdb-server` and is
what the backend pods run; it needs the `arrow-rs` / `duckdb` submodules checked
out. CI publishes both to Docker Hub (`giladkl/pivot`, `giladkl/pivot-operator`).

## Install on a new cluster

CI publishes both images to Docker Hub (`giladkl/pivot-operator` and
`giladkl/pivot`), and the manifests default to them, so a plain install is one
command:

```sh
kubectl apply -k operator/deploy   # CRD + RBAC + operator Deployment
```

Then create endpoints (the samples already point `spec.image` at `giladkl/pivot`):

```sh
kubectl apply -f operator/deploy/samples/per-connection.yaml
kubectl apply -f operator/deploy/samples/shared.yaml
kubectl get pivotendpoints
```

To use your own registry instead of the published images, build and push:

```sh
# Operator/router image (no submodules needed):
docker build -t <registry>/pivot-operator:<tag> operator/ && docker push <registry>/pivot-operator:<tag>
# Pivot server image (needs the arrow-rs / duckdb submodules):
git submodule update --init --recursive
docker build -t <registry>/pivot:<tag> -f server/Dockerfile . && docker push <registry>/pivot:<tag>
```

then override the operator image (the router pods reuse it automatically, so it
is the only one to set) and point endpoints at your pivot image:

```sh
cd operator/deploy
kustomize edit set image giladkl/pivot-operator:latest=<registry>/pivot-operator:<tag>
kubectl apply -k .
```

Notes:
- The operator discovers its own image (via the downward-API `POD_NAME`/
  `POD_NAMESPACE` env on its Deployment) and stamps it onto the router pods. To
  use a different router image, set `PIVOT_ROUTER_IMAGE` on the operator
  Deployment.
- For a local kind cluster there is no registry: `kind load docker-image` the
  images and keep `imagePullPolicy: IfNotPresent` (the default). The e2e test
  does exactly this.

## PivotEndpoint spec

| Field                | Mode        | Meaning                                                        |
| -------------------- | ----------- | ------------------------------------------------------------- |
| `mode`               | both        | `perConnection` or `shared`.                                  |
| `image`              | both        | Backend (pivot) container image.                              |
| `port`               | both        | Port the backend listens on (default 5432).                   |
| `serviceType`        | both        | `LoadBalancer` (default), `NodePort`, or `ClusterIP`.         |
| `resources`          | both        | Pod sizing: `cpu`, `memory`, `cpuLimit`, `memoryLimit`.       |
| `args` / `command`   | both        | Extra args / entrypoint override for the backend.             |
| `autoBind`           | both        | Inject `--bind 0.0.0.0:<port>` (default true).                |
| `env`                | both        | Extra env vars for the backend container.                     |
| `maxPods`            | perConn     | Cap on concurrent live pods (extra sessions rejected).        |
| `reuseIdlePods`      | perConn     | Pool disconnected pods instead of deleting them.              |
| `idleTimeoutSeconds` | perConn     | Reap a pooled idle pod after this long.                       |
| `sharedReplicas`     | shared      | Number of shared pods to keep ready (default 1).              |

### Memory sizing

The pivot server sizes its internal buffer pool from its cgroup memory limit
(falling back to the machine's total RAM when unlimited), so it fits the pod
automatically. Just set a `resources.memory` limit; no memory flag is needed.

## Tests

Pure builder unit tests (no cluster):

```sh
cargo test --test builders
```

Full end-to-end test against a real cluster (creates a `kind` cluster, builds and
loads the image, deploys the operator, drives both modes through real psql
connections). Requires `docker` (running), `kind`, and `kubectl`:

```sh
PIVOT_E2E=1 cargo test --test e2e -- --nocapture
# keep the cluster afterwards for inspection:
PIVOT_E2E=1 PIVOT_E2E_KEEP=1 cargo test --test e2e -- --nocapture
```
