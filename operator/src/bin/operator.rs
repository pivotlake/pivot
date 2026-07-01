//! Entry point for the pivot operator: applies the CRD, then runs the
//! controller reconciling `PivotEndpoint` resources.

use kube::Client;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    operator::init_tracing();
    let client = Client::try_default().await?;
    operator::controller::ensure_crd(client.clone()).await?;
    info!("starting controller");
    operator::controller::run(client).await
}
