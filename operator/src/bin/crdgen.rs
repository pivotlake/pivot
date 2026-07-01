//! Prints the `PivotEndpoint` CustomResourceDefinition as YAML, for committing
//! to `deploy/` or piping into `kubectl apply`.

use kube::CustomResourceExt;
use operator::crd::PivotEndpoint;

fn main() {
    let crd = PivotEndpoint::crd();
    print!("{}", serde_yaml::to_string(&crd).expect("serialize CRD"));
}
