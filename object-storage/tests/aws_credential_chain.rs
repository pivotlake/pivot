//! The AWS credential chain, through the stores it authenticates: which
//! source's keys a store signs with, how keys close to expiring are read
//! again, and what a store without any says when S3 denies it.
//!
//! Every source is pointed at a service of the test's own, through the same
//! environment variables the AWS SDKs read, so nothing here depends on the
//! machine's own AWS identity.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use object_storage::{
    AwsCredentialChain, DataFileLocation, ObjectPath, ObjectStore, S3Auth, S3Credentials, S3Store,
};

/// The request line and the headers of one request a [`FakeService`] received.
#[derive(Clone)]
struct ReceivedRequest {
    head: Vec<String>,
}

impl ReceivedRequest {
    fn request_line(&self) -> &str {
        &self.head[0]
    }

    fn has_header(&self, name: &str, value: &str) -> bool {
        self.head[1..].iter().any(|line| {
            line.split_once(':').is_some_and(|(header, held)| {
                header.eq_ignore_ascii_case(name) && held.trim() == value
            })
        })
    }
}

/// An HTTP service answering every request with what `respond` makes of it:
/// a status line and a body.
struct FakeService {
    endpoint: String,
    received: Arc<Mutex<Vec<ReceivedRequest>>>,
}

impl FakeService {
    fn spawn(
        respond: impl Fn(&ReceivedRequest) -> (&'static str, String) + Send + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let received = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&received);

        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let head: Vec<String> = BufReader::new(&stream)
                    .lines()
                    .map(Result::unwrap)
                    .take_while(|line| !line.is_empty())
                    .collect();
                let request = ReceivedRequest { head };

                let (status, response_body) = respond(&request);
                recorded.lock().unwrap().push(request);
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                )
                .unwrap();
            }
        });

        Self { endpoint, received }
    }

    fn received(&self) -> Vec<ReceivedRequest> {
        self.received.lock().unwrap().clone()
    }
}

/// An instance metadata service whose instance carries `role`, answering each
/// request for the role's keys with the next document `issue_keys` writes.
fn spawn_instance_metadata(
    role: &'static str,
    issue_keys: impl Fn() -> String + Send + 'static,
) -> FakeService {
    FakeService::spawn(move |request| {
        let line = request.request_line();
        if line.starts_with("PUT /latest/api/token") {
            ("200 OK", "session-token".to_string())
        } else if !request.has_header("X-aws-ec2-metadata-token", "session-token") {
            ("401 Unauthorized", String::new())
        } else if line.starts_with("GET /latest/meta-data/iam/security-credentials/ ") {
            ("200 OK", role.to_string())
        } else if line.starts_with(&format!(
            "GET /latest/meta-data/iam/security-credentials/{role} "
        )) {
            ("200 OK", issue_keys())
        } else {
            ("404 Not Found", String::new())
        }
    })
}

/// The document a credentials endpoint issues a session's keys in.
fn session_keys_document(access_key: &str, token: &str, lifetime: Duration) -> String {
    let expiration: chrono::DateTime<chrono::Utc> = (SystemTime::now() + lifetime).into();
    format!(
        r#"{{"Code": "Success", "AccessKeyId": "{access_key}", "SecretAccessKey": "secret",
            "Token": "{token}", "Expiration": "{}"}}"#,
        expiration.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    )
}

/// The chain over exactly `variables`. The instance metadata service is
/// switched off unless a test points the chain at one of its own.
fn chain(variables: &[(&str, &str)]) -> Arc<AwsCredentialChain> {
    let names_metadata_endpoint = variables
        .iter()
        .any(|(name, _)| *name == "AWS_EC2_METADATA_SERVICE_ENDPOINT");
    let metadata_switch =
        (!names_metadata_endpoint).then_some(("AWS_EC2_METADATA_DISABLED", "true"));
    AwsCredentialChain::new(
        variables
            .iter()
            .copied()
            .chain(metadata_switch)
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
    )
}

/// Open a store authenticated by `chain`, at an endpoint nothing is sent to
/// unless the test names one.
fn open_store(chain: Arc<AwsCredentialChain>, endpoint: &str) -> object_storage::Result<S3Store> {
    S3Store::with_credentials(
        "s3://lake/warehouse",
        S3Credentials {
            region: Some("us-east-1".to_string()),
            auth: S3Auth::Chain(chain),
            endpoint: Some(endpoint.to_string()),
        },
    )
}

const UNUSED_ENDPOINT: &str = "http://objects.example";

/// What authenticates the ring's reads of one of `store`'s files.
fn open_file(store: &S3Store) -> Arc<dyn dispatch::io::RequestAuth> {
    let DataFileLocation::Remote { auth, .. } =
        store.source(&ObjectPath::new("part.parquet")).unwrap()
    else {
        panic!("an S3 source is remote");
    };
    auth.expect("a store with keys signs the ring's requests")
}

/// The headers a read of `file` is sent with now.
fn sign_read(file: &dyn dispatch::io::RequestAuth) -> Vec<(String, String)> {
    file.authenticate("GET", "/lake/warehouse/part.parquet")
        .unwrap()
}

/// The access key a request with `headers` is signed with.
fn signing_access_key(headers: &[(String, String)]) -> String {
    let authorization = find_header(headers, "authorization").expect("a signed request");
    let credential = authorization.split("Credential=").nth(1).unwrap();
    credential.split('/').next().unwrap().to_string()
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[test]
fn keys_in_the_environment_sign_requests_with_their_session_token() {
    let chain = chain(&[
        ("AWS_ACCESS_KEY_ID", "ENVIRONMENTKEY"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
        ("AWS_SESSION_TOKEN", "environment-token"),
    ]);

    let store = open_store(chain, UNUSED_ENDPOINT).unwrap();
    let headers = sign_read(open_file(&store).as_ref());

    assert_eq!(signing_access_key(&headers), "ENVIRONMENTKEY");
    assert_eq!(
        find_header(&headers, "x-amz-security-token"),
        Some("environment-token")
    );
}

#[test]
fn an_access_key_without_its_secret_is_an_error() {
    let chain = chain(&[("AWS_ACCESS_KEY_ID", "ENVIRONMENTKEY")]);

    let error = open_store(chain, UNUSED_ENDPOINT).unwrap_err();

    assert!(
        error.to_string().contains("must be set together"),
        "{error}"
    );
}

#[test]
fn the_instance_role_signs_requests() {
    let metadata = spawn_instance_metadata("lake-reader", || {
        session_keys_document("ROLEKEY", "role-token", Duration::from_secs(6 * 3600))
    });
    let chain = chain(&[("AWS_EC2_METADATA_SERVICE_ENDPOINT", &metadata.endpoint)]);

    let store = open_store(chain, UNUSED_ENDPOINT).unwrap();
    let headers = sign_read(open_file(&store).as_ref());

    assert_eq!(signing_access_key(&headers), "ROLEKEY");
    assert_eq!(
        find_header(&headers, "x-amz-security-token"),
        Some("role-token")
    );
}

#[test]
fn a_file_opened_before_its_keys_expire_is_read_with_the_keys_read_next() {
    // The first keys are three seconds away from the point where the chain
    // reads them again, five minutes before they expire.
    let issued = Arc::new(Mutex::new(0));
    let metadata = spawn_instance_metadata("lake-reader", move || {
        let mut issued = issued.lock().unwrap();
        *issued += 1;
        match *issued {
            1 => session_keys_document("FIRSTKEY", "first-token", Duration::from_secs(303)),
            _ => session_keys_document("SECONDKEY", "second-token", Duration::from_secs(6 * 3600)),
        }
    });
    let chain = chain(&[("AWS_EC2_METADATA_SERVICE_ENDPOINT", &metadata.endpoint)]);
    let store = open_store(chain, UNUSED_ENDPOINT).unwrap();
    let file = open_file(&store);
    let first_key = signing_access_key(&sign_read(file.as_ref()));

    // The store reads its keys again when it hands a file out, as each later
    // query asks it to.
    let deadline = Instant::now() + Duration::from_secs(20);
    while signing_access_key(&sign_read(file.as_ref())) == first_key && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
        open_file(&store);
    }

    let headers = sign_read(file.as_ref());
    assert_eq!(first_key, "FIRSTKEY");
    assert_eq!(signing_access_key(&headers), "SECONDKEY");
    assert_eq!(
        find_header(&headers, "x-amz-security-token"),
        Some("second-token")
    );
}

#[test]
fn an_instance_without_a_role_leaves_the_store_anonymous() {
    let metadata = FakeService::spawn(|request| {
        if request.request_line().starts_with("PUT /latest/api/token") {
            ("200 OK", "session-token".to_string())
        } else {
            ("404 Not Found", String::new())
        }
    });
    let chain = chain(&[("AWS_EC2_METADATA_SERVICE_ENDPOINT", &metadata.endpoint)]);

    let store = open_store(chain, UNUSED_ENDPOINT).unwrap();

    assert!(store.is_anonymous());
}

#[test]
fn a_container_credentials_endpoint_signs_requests() {
    let endpoint = FakeService::spawn(|request| {
        if request.has_header("Authorization", "task-token") {
            (
                "200 OK",
                session_keys_document("TASKKEY", "task-session", Duration::from_secs(3600)),
            )
        } else {
            ("401 Unauthorized", String::new())
        }
    });
    let chain = chain(&[
        (
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            &format!("{}/v2/credentials", endpoint.endpoint),
        ),
        ("AWS_CONTAINER_AUTHORIZATION_TOKEN", "task-token"),
    ]);

    let store = open_store(chain, UNUSED_ENDPOINT).unwrap();
    let headers = sign_read(open_file(&store).as_ref());

    assert_eq!(signing_access_key(&headers), "TASKKEY");
    assert!(
        endpoint.received()[0]
            .request_line()
            .starts_with("GET /v2/credentials ")
    );
}

#[test]
fn a_denied_request_without_credentials_says_none_were_found() {
    let s3 = FakeService::spawn(|_| {
        (
            "403 Forbidden",
            "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>".to_string(),
        )
    });
    let store = open_store(chain(&[]), &s3.endpoint).unwrap();

    let error = store
        .get(&ObjectPath::new("_pivot_manifest.json"))
        .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("sent unsigned because no AWS credentials were found"),
        "{message}"
    );
    assert!(
        !s3.received()[0]
            .head
            .iter()
            .any(|line| { line.to_ascii_lowercase().starts_with("authorization:") })
    );
}
