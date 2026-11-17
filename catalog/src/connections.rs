//! The connections a server has open, as `system.connections` reports them.
//!
//! Whatever accepts connections sits above this crate, so the registry lives
//! here, where the `system` datastore can read it. A frontend registers each
//! connection it accepts and marks every statement the connection runs; both
//! are handles whose drop undoes them, so a connection that ends or a statement
//! that fails part way leaves nothing behind to be reported.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

/// What a connection is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Waiting for its client to send a statement.
    Idle,
    /// Running a statement.
    Query,
    /// Receiving the rows of a `COPY ... FROM STDIN` from its client.
    Copy,
}

impl ConnectionState {
    /// The name the state answers to in the `state` column.
    pub fn name(self) -> &'static str {
        match self {
            ConnectionState::Idle => "idle",
            ConnectionState::Query => "query",
            ConnectionState::Copy => "copy",
        }
    }
}

/// One open connection, as it was when the registry was read.
#[derive(Debug, Clone)]
pub struct ConnectionStatus {
    /// Unique among the connections this registry has ever held.
    pub id: u64,
    pub user_name: String,
    /// What the client called itself when it connected, or an empty string if
    /// it did not say.
    pub application_name: String,
    pub client_address: SocketAddr,
    pub connected_at: SystemTime,
    pub state: ConnectionState,
    pub state_changed_at: SystemTime,
    /// The statement being run, or an empty string while idle.
    pub statement: String,
    /// Bytes the running copy has received from the client so far.
    pub copy_bytes: u64,
}

#[derive(Debug, Default)]
struct OpenConnections {
    registered_count: u64,
    by_id: BTreeMap<u64, ConnectionStatus>,
}

/// Every connection a server has open.
#[derive(Debug, Default)]
pub struct ConnectionRegistry {
    open: Mutex<OpenConnections>,
}

impl ConnectionRegistry {
    /// Add a connection, idle, for as long as the returned handle lives.
    pub fn register(
        self: &Arc<Self>,
        user_name: String,
        application_name: String,
        client_address: SocketAddr,
    ) -> RegisteredConnection {
        let now = SystemTime::now();
        let mut open = self.lock();
        open.registered_count += 1;
        let id = open.registered_count;
        open.by_id.insert(
            id,
            ConnectionStatus {
                id,
                user_name,
                application_name,
                client_address,
                connected_at: now,
                state: ConnectionState::Idle,
                state_changed_at: now,
                statement: String::new(),
                copy_bytes: 0,
            },
        );
        RegisteredConnection {
            registry: self.clone(),
            id,
        }
    }

    /// Every open connection, oldest first.
    pub fn read_all(&self) -> Vec<ConnectionStatus> {
        self.lock().by_id.values().cloned().collect()
    }

    /// Nothing done under the lock can panic part way through a change, so a
    /// poisoned lock still guards a consistent map.
    fn lock(&self) -> MutexGuard<'_, OpenConnections> {
        self.open
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A connection's place in the registry. Dropping it removes the connection.
#[derive(Debug)]
pub struct RegisteredConnection {
    registry: Arc<ConnectionRegistry>,
    id: u64,
}

impl RegisteredConnection {
    /// Mark the connection as running `statement` until the returned handle
    /// drops, which returns it to idle.
    pub fn run_statement(self: &Arc<Self>, statement: &str) -> RunningStatement {
        self.change_state(ConnectionState::Query, statement.to_string());
        RunningStatement {
            connection: self.clone(),
        }
    }

    fn change_state(&self, state: ConnectionState, statement: String) {
        self.update(|status| {
            status.state = state;
            status.state_changed_at = SystemTime::now();
            status.statement = statement;
            status.copy_bytes = 0;
        });
    }

    fn update(&self, change: impl FnOnce(&mut ConnectionStatus)) {
        let mut open = self.registry.lock();
        let status = open
            .by_id
            .get_mut(&self.id)
            .expect("a connection stays registered for as long as its handle lives");
        change(status);
    }
}

impl Drop for RegisteredConnection {
    fn drop(&mut self) {
        self.registry.lock().by_id.remove(&self.id);
    }
}

/// A statement a connection is running. Dropping it returns the connection to
/// idle, however the statement ended.
#[derive(Debug)]
pub struct RunningStatement {
    connection: Arc<RegisteredConnection>,
}

impl RunningStatement {
    /// The statement turned out to be a `COPY ... FROM STDIN`, and now waits
    /// for the client to send its rows.
    pub fn receive_copy(&self) {
        self.connection.update(|status| {
            status.state = ConnectionState::Copy;
            status.state_changed_at = SystemTime::now();
        });
    }

    /// The client sent `byte_count` more bytes of the copy's rows.
    pub fn add_copy_bytes(&self, byte_count: usize) {
        self.connection
            .update(|status| status.copy_bytes += byte_count as u64);
    }
}

impl Drop for RunningStatement {
    fn drop(&mut self) {
        self.connection
            .change_state(ConnectionState::Idle, String::new());
    }
}
