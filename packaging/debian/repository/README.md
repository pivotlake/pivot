# PivotLake APT repository

Google Artifact Registry stores the versioned Debian packages and generates and
signs the repository metadata served at `https://packages.pivotlake.io/deb`.

Two public APT repositories in `pivot-packages/us` are the release lines:

```text
stable   final releases
testing  release candidates and final releases
```

Artifact Registry signs `InRelease` with Google's repository signer. The
public key fingerprint is:

```text
35BA A0B3 3E9E B396 F59C  A838 C0BA 5CE6 DC63 15A3
```

The load balancer serves Google's current public key at the stable PivotLake
URL `https://packages.pivotlake.io/keys/pivotlake-archive-key.asc`. PivotLake
does not hold an APT signing private key.

## Publication model

The Deploy Binaries workflow receives the already tested `amd64` and `arm64`
package artifacts and uploads them with `gcloud artifacts apt upload`.

- A release candidate updates `testing` only.
- A final release updates `testing` first and then `stable`.
- Artifact Registry retains older package versions, so an indexed version can
  be selected with `apt-get install pivot=<version>`.
- The deploy workflow's emergency `override` option deletes the existing
  version from each target repository before uploading both architectures
  again. Clients that already installed that version will not receive the new
  package from `apt upgrade`.
- Artifact Registry generates and signs the APT metadata.
- The workflow tests the exact public URL in a fresh Debian container without
  installing Google's `apt-transport-artifact-registry` helper.

Releases remain serialized so two jobs cannot interleave channel updates and
release-candidate timestamps remain strictly increasing.

## Repository configuration

GitHub repository variables:

```text
PIVOT_APT_PROJECT=pivot-packages
PIVOT_APT_LOCATION=us
PIVOT_APT_URL=https://packages.pivotlake.io/deb
```

The existing `GCP_SA_KEY` service account needs Artifact Registry Writer on
the `stable` and `testing` repositories. Overrides additionally require the
`artifactregistry.versions.delete` permission. Both repositories grant
Artifact Registry Reader to `allUsers`.

Publishing no longer uses a repository-signing secret or private reprepro
state. The previous GCS objects are retained temporarily as rollback data.

## HTTPS endpoint

Artifact Registry's native repository root is
`https://us-apt.pkg.dev/projects/pivot-packages`. A global external HTTPS load
balancer exposes the conventional public layout:

```text
packages.pivotlake.io/deb/dists/stable/...
                    ↓
us-apt.pkg.dev/projects/pivot-packages/dists/stable/...
```

The same load balancer rewrites the public-key URL to Artifact Registry's
current signer key. Other paths continue to use `pivot-releases` as the default
backend, which also leaves the previous static repository available for
rollback. The `pivotlake.io` zone is hosted by Cloudflare; its `packages` A
record points directly to the load balancer with the Cloudflare proxy disabled.
