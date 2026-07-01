//! Entry point for the connection router: proxies psql sessions to pivot pods
//! per the named `PivotEndpoint`. All configuration is on [`RouterConfig`],
//! which derives the CLI directly.

use clap::Parser;
use operator::router::{RouterConfig, run};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    operator::init_tracing();
    run(RouterConfig::parse()).await
}
