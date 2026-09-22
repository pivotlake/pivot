//! A catalog's credentials from the process's environment, for a process with
//! no config file to name a secret in (`pivot open`): the way a bucket's keys
//! come from `AWS_*` when no storage secret covers it.

use crate::IcebergCatalogAuth;

/// A bearer token the catalog is authenticated to with.
pub const TOKEN_VAR: &str = "PIVOT_ICEBERG_TOKEN";
/// An OAuth2 client credential (`client_id:client_secret`) exchanged for a
/// token at the catalog's token endpoint.
pub const CREDENTIAL_VAR: &str = "PIVOT_ICEBERG_CREDENTIAL";
/// The OAuth2 token endpoint, when it is not the catalog's own.
pub const OAUTH2_SERVER_URI_VAR: &str = "PIVOT_ICEBERG_OAUTH2_SERVER_URI";
/// The OAuth2 scope requested with the credential.
pub const OAUTH2_SCOPE_VAR: &str = "PIVOT_ICEBERG_OAUTH2_SCOPE";

#[derive(Debug, thiserror::Error)]
pub enum EnvCredentialError {
    #[error("set exactly one of {TOKEN_VAR} and {CREDENTIAL_VAR}, not both")]
    TokenAndCredential,
    #[error("{variable} applies only together with {CREDENTIAL_VAR}")]
    OAuth2WithoutCredential { variable: &'static str },
    #[error("{variable} is set but empty")]
    Empty { variable: &'static str },
    #[error("{variable} is not valid UTF-8")]
    NotUtf8 { variable: &'static str },
}

impl IcebergCatalogAuth {
    /// What the catalog is authenticated to with, from the environment:
    /// [`TOKEN_VAR`] for a bearer token, or [`CREDENTIAL_VAR`] for an OAuth2
    /// client credential with the optional [`OAUTH2_SERVER_URI_VAR`] and
    /// [`OAUTH2_SCOPE_VAR`]. `None` when neither is set, for a catalog that
    /// requires nothing.
    pub fn from_env() -> Result<Option<Self>, EnvCredentialError> {
        parse_auth(
            read_variable(TOKEN_VAR)?,
            read_variable(CREDENTIAL_VAR)?,
            read_variable(OAUTH2_SERVER_URI_VAR)?,
            read_variable(OAUTH2_SCOPE_VAR)?,
        )
    }
}

/// The value of `variable`, or `None` when it is not set. An empty value is
/// refused rather than sent to the catalog as an empty secret.
fn read_variable(variable: &'static str) -> Result<Option<String>, EnvCredentialError> {
    match std::env::var(variable) {
        Ok(value) if value.is_empty() => Err(EnvCredentialError::Empty { variable }),
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(EnvCredentialError::NotUtf8 { variable }),
    }
}

/// Combine the four variables' values into the catalog's authentication,
/// under the rule an `iceberg` secret follows: exactly one of a token and a
/// credential, and the OAuth2 settings only with the credential they refine.
fn parse_auth(
    token: Option<String>,
    credential: Option<String>,
    oauth2_server_uri: Option<String>,
    oauth2_scope: Option<String>,
) -> Result<Option<IcebergCatalogAuth>, EnvCredentialError> {
    match (token, credential) {
        (Some(_), Some(_)) => Err(EnvCredentialError::TokenAndCredential),
        (None, Some(credential)) => Ok(Some(IcebergCatalogAuth::OAuth2 {
            credential,
            server_uri: oauth2_server_uri,
            scope: oauth2_scope,
        })),
        (token, None) => {
            if oauth2_server_uri.is_some() {
                return Err(EnvCredentialError::OAuth2WithoutCredential {
                    variable: OAUTH2_SERVER_URI_VAR,
                });
            }
            if oauth2_scope.is_some() {
                return Err(EnvCredentialError::OAuth2WithoutCredential {
                    variable: OAUTH2_SCOPE_VAR,
                });
            }
            Ok(token.map(IcebergCatalogAuth::Token))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EnvCredentialError, parse_auth};
    use crate::IcebergCatalogAuth;

    fn some(value: &str) -> Option<String> {
        Some(value.to_string())
    }

    #[test]
    fn no_variables_mean_an_unauthenticated_catalog() {
        let auth = parse_auth(None, None, None, None).unwrap();

        assert!(auth.is_none());
    }

    #[test]
    fn a_token_alone_is_a_bearer_token() {
        let auth = parse_auth(some("t0k3n"), None, None, None).unwrap();

        assert!(matches!(auth, Some(IcebergCatalogAuth::Token(token)) if token == "t0k3n"));
    }

    #[test]
    fn a_credential_carries_its_oauth2_settings() {
        let auth = parse_auth(
            None,
            some("id:secret"),
            some("https://auth.example.com/token"),
            some("PRINCIPAL_ROLE:ALL"),
        )
        .unwrap();

        let Some(IcebergCatalogAuth::OAuth2 {
            credential,
            server_uri,
            scope,
        }) = auth
        else {
            panic!("a credential did not parse as OAuth2");
        };
        assert_eq!(credential, "id:secret");
        assert_eq!(
            server_uri.as_deref(),
            Some("https://auth.example.com/token")
        );
        assert_eq!(scope.as_deref(), Some("PRINCIPAL_ROLE:ALL"));
    }

    #[test]
    fn a_token_and_a_credential_together_are_refused() {
        let error = parse_auth(some("t0k3n"), some("id:secret"), None, None).unwrap_err();

        assert!(matches!(error, EnvCredentialError::TokenAndCredential));
    }

    #[test]
    fn oauth2_settings_without_a_credential_are_refused() {
        let with_token = parse_auth(some("t0k3n"), None, None, some("scope")).unwrap_err();
        let alone = parse_auth(None, None, some("https://auth"), None).unwrap_err();

        assert!(matches!(
            with_token,
            EnvCredentialError::OAuth2WithoutCredential { variable }
                if variable == "PIVOT_ICEBERG_OAUTH2_SCOPE"
        ));
        assert!(matches!(
            alone,
            EnvCredentialError::OAuth2WithoutCredential { variable }
                if variable == "PIVOT_ICEBERG_OAUTH2_SERVER_URI"
        ));
    }
}
