//! Host name resolution for the HTTP engine, without any I/O of its own.
//!
//! The engine opens a connection to an address, and a store's host name maps
//! to many addresses that change every few seconds. This module builds the
//! question a name server is asked, reads its answer, and keeps what each
//! host last resolved to. The engine sends the question and receives the
//! answer on the worker's ring, so resolving a name never blocks the worker.
//!
//! The C library's resolver is not used. It blocks its caller until the name
//! server answers, and waits five seconds before it asks again when a
//! question or its answer is lost.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};
use thiserror::Error;

/// How long an answer is used before the host is looked up again. Stores
/// publish their addresses for about this long.
pub(crate) const REFRESH_ADDRESSES_AFTER: Duration = Duration::from_secs(5);

/// How long an address is kept after a name server last gave it. A store
/// keeps serving at an address well after it stops publishing it, and
/// keeping addresses from several answers spreads connections over more of
/// the store's servers.
const MAX_ADDRESS_AGE: Duration = Duration::from_secs(30);

/// The most addresses kept for one host.
const MAX_ADDRESSES_PER_HOST: usize = 32;

/// How long an address is passed over after a connection to it failed or
/// stalled.
const AVOID_FAILED_ADDRESS_FOR: Duration = Duration::from_secs(10);

/// The largest answer a name server sends in one datagram to a question that
/// does not ask for more.
pub(crate) const MAX_ANSWER_LEN: usize = 512;

const ADDRESS_RECORD_TYPE: u16 = 1;
const INTERNET_CLASS: u16 = 1;
const HEADER_LEN: usize = 12;

#[derive(Debug, Clone, Error)]
pub enum DnsError {
    #[error("{host} cannot be resolved: /etc/resolv.conf names no name server")]
    NoNameServer { host: String },
    #[error("{host} is not a host name a name server can be asked about")]
    InvalidHostName { host: String },
    #[error("no name server answered for {host} in {attempts} attempts")]
    NoAnswer { host: String, attempts: u32 },
    #[error("the name server's answer is malformed")]
    MalformedAnswer,
    #[error("the name server's answer is for another question")]
    AnswerToAnotherQuestion,
    #[error("the name server's answer does not fit in one datagram")]
    TruncatedAnswer,
    #[error("the name server refused the question with code {code}")]
    Refused { code: u8 },
    #[error("the name server knows no address for the host")]
    NoAddresses,
}

/// The name servers to ask and the hosts whose address needs no asking, as
/// the machine configures them.
pub(crate) struct NameServers {
    servers: Vec<SocketAddr>,
    /// Host names with a fixed address, in lower case.
    fixed_addresses: HashMap<String, IpAddr>,
}

impl NameServers {
    /// Read the machine's name servers from `/etc/resolv.conf` and its fixed
    /// host addresses from `/etc/hosts`. A machine without one of the files
    /// has no name servers or no fixed addresses.
    pub fn load() -> Self {
        let resolver_configuration =
            std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
        let hosts_file = std::fs::read_to_string("/etc/hosts").unwrap_or_default();
        Self::parse(&resolver_configuration, &hosts_file)
    }

    fn parse(resolver_configuration: &str, hosts_file: &str) -> Self {
        let servers = resolver_configuration
            .lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                (words.next() == Some("nameserver")).then(|| words.next())?
            })
            .filter_map(|address| address.parse::<IpAddr>().ok())
            .map(|address| SocketAddr::new(address, 53))
            .collect();
        let mut fixed_addresses: HashMap<String, IpAddr> = HashMap::new();
        for line in hosts_file.lines() {
            let line = line.split('#').next().unwrap_or_default();
            let mut words = line.split_whitespace();
            let Some(Ok(address)) = words.next().map(str::parse::<IpAddr>) else {
                continue;
            };
            for name in words {
                // A name listed with both kinds of address keeps its IPv4 one,
                // the kind a name server is asked for.
                let known = fixed_addresses
                    .entry(name.to_ascii_lowercase())
                    .or_insert(address);
                if known.is_ipv6() && address.is_ipv4() {
                    *known = address;
                }
            }
        }
        Self {
            servers,
            fixed_addresses,
        }
    }

    /// Ask `servers` instead of the machine's name servers.
    #[cfg(test)]
    pub fn force_servers(&mut self, servers: Vec<SocketAddr>) {
        self.servers = servers;
    }

    /// The name server to send attempt number `attempt` of a lookup to. Each
    /// attempt goes to the next server, so one that is down costs a single
    /// attempt.
    pub fn choose_server(&self, attempt: u32) -> Option<SocketAddr> {
        if self.servers.is_empty() {
            return None;
        }
        Some(self.servers[attempt as usize % self.servers.len()])
    }

    /// The address of a `host` that is written as an address or listed in
    /// the machine's hosts file, which no name server is asked about.
    pub fn find_fixed_address(&self, host: &str) -> Option<IpAddr> {
        let unbracketed = host.trim_start_matches('[').trim_end_matches(']');
        if let Ok(address) = unbracketed.parse::<IpAddr>() {
            return Some(address);
        }
        self.fixed_addresses
            .get(&host.to_ascii_lowercase())
            .copied()
    }
}

/// Build the datagram that asks a name server for the IPv4 addresses of
/// `host`. The answer carries `id` back.
pub(crate) fn encode_question(id: u16, host: &str) -> Result<Vec<u8>, DnsError> {
    let invalid = || DnsError::InvalidHostName {
        host: host.to_string(),
    };
    let name = host.trim_end_matches('.');
    if name.is_empty() || name.len() > 253 {
        return Err(invalid());
    }
    let mut question = Vec::with_capacity(HEADER_LEN + name.len() + 6);
    question.extend_from_slice(&id.to_be_bytes());
    // Flags: a standard question that asks the server to resolve aliases.
    question.extend_from_slice(&0x0100u16.to_be_bytes());
    // One question, and no other records.
    question.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]);
    for label in name.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err(invalid());
        }
        question.push(label.len() as u8);
        question.extend_from_slice(label.as_bytes());
    }
    question.push(0);
    question.extend_from_slice(&ADDRESS_RECORD_TYPE.to_be_bytes());
    question.extend_from_slice(&INTERNET_CLASS.to_be_bytes());
    Ok(question)
}

/// Read the IPv4 addresses out of a name server's `answer` to the question
/// sent with `id`. A host that is an alias answers with the alias and the
/// addresses behind it; only the addresses are returned.
pub(crate) fn parse_answer(answer: &[u8], id: u16) -> Result<Vec<Ipv4Addr>, DnsError> {
    let read_u16 = |at: usize| -> Result<u16, DnsError> {
        let bytes = answer.get(at..at + 2).ok_or(DnsError::MalformedAnswer)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    };
    let skip_name = |mut at: usize| -> Result<usize, DnsError> {
        loop {
            let length = *answer.get(at).ok_or(DnsError::MalformedAnswer)? as usize;
            if length & 0xC0 == 0xC0 {
                // A pointer to a name written earlier ends this name.
                return Ok(at + 2);
            }
            if length == 0 {
                return Ok(at + 1);
            }
            at += 1 + length;
        }
    };

    let flags = read_u16(2)?;
    let is_answer = flags & 0x8000 != 0;
    if read_u16(0)? != id || !is_answer {
        return Err(DnsError::AnswerToAnotherQuestion);
    }
    if flags & 0x0200 != 0 {
        return Err(DnsError::TruncatedAnswer);
    }
    let code = (flags & 0x000F) as u8;
    if code != 0 {
        return Err(DnsError::Refused { code });
    }
    let question_count = read_u16(4)?;
    let record_count = read_u16(6)?;

    let mut at = HEADER_LEN;
    for _ in 0..question_count {
        at = skip_name(at)? + 4;
    }
    let mut addresses = Vec::new();
    for _ in 0..record_count {
        at = skip_name(at)?;
        let record_type = read_u16(at)?;
        let data_len = read_u16(at + 8)? as usize;
        let data = answer
            .get(at + 10..at + 10 + data_len)
            .ok_or(DnsError::MalformedAnswer)?;
        if record_type == ADDRESS_RECORD_TYPE && data_len == 4 {
            addresses.push(Ipv4Addr::new(data[0], data[1], data[2], data[3]));
        }
        at += 10 + data_len;
    }
    if addresses.is_empty() {
        return Err(DnsError::NoAddresses);
    }
    Ok(addresses)
}

/// What the address book knows for a host a connection is about to open to.
pub(crate) enum AddressChoice {
    /// Connect to this address.
    Fresh(IpAddr),
    /// Connect to this address, and look the host up again for the
    /// connections after this one.
    DueForRefresh(IpAddr),
    /// Nothing recent enough is known. The host is looked up first.
    Unknown,
}

struct KnownAddress {
    address: IpAddr,
    last_answered: Instant,
    avoided_until: Instant,
}

struct HostAddresses {
    addresses: Vec<KnownAddress>,
    answered_at: Instant,
    /// Index of the address the next connection goes to.
    next: usize,
}

/// The addresses each host last resolved to, for one worker. Connections to
/// a host go to its addresses in turn, so they spread over the store's
/// servers, and an address a connection just failed on is passed over for a
/// while.
pub(crate) struct AddressBook {
    hosts: HashMap<String, HostAddresses>,
    refresh_after: Duration,
}

impl AddressBook {
    pub fn new() -> Self {
        Self {
            hosts: HashMap::new(),
            refresh_after: REFRESH_ADDRESSES_AFTER,
        }
    }

    /// Look a host up again once its answer is `refresh_after` old.
    #[cfg(test)]
    pub fn force_refresh_after(&mut self, refresh_after: Duration) {
        self.refresh_after = refresh_after;
    }

    /// Choose the address the next connection to `host` goes to.
    pub fn choose_address(&mut self, host: &str, now: Instant) -> AddressChoice {
        let Some(known) = self.hosts.get_mut(host) else {
            return AddressChoice::Unknown;
        };
        let age = now.duration_since(known.answered_at);
        if age > MAX_ADDRESS_AGE {
            return AddressChoice::Unknown;
        }
        // The first address in turn that no connection failed on lately. When
        // every address failed lately, the one in turn is tried again.
        let count = known.addresses.len();
        let offset = (0..count)
            .find(|offset| known.addresses[(known.next + offset) % count].avoided_until <= now)
            .unwrap_or(0);
        let chosen = (known.next + offset) % count;
        known.next = (chosen + 1) % count;
        let address = known.addresses[chosen].address;
        if age > self.refresh_after {
            AddressChoice::DueForRefresh(address)
        } else {
            AddressChoice::Fresh(address)
        }
    }

    /// Record a name server's answer for `host`. Addresses from earlier
    /// answers stay until they are [`MAX_ADDRESS_AGE`] old.
    pub fn record_answer(&mut self, host: &str, answered: &[Ipv4Addr], now: Instant) {
        let known = self
            .hosts
            .entry(host.to_string())
            .or_insert_with(|| HostAddresses {
                addresses: Vec::new(),
                answered_at: now,
                // Workers start at different addresses of the same answer.
                next: rand::random::<u32>() as usize,
            });
        known.answered_at = now;
        for address in answered {
            let address = IpAddr::V4(*address);
            match known
                .addresses
                .iter_mut()
                .find(|kept| kept.address == address)
            {
                Some(kept) => kept.last_answered = now,
                None => known.addresses.push(KnownAddress {
                    address,
                    last_answered: now,
                    avoided_until: now,
                }),
            }
        }
        known
            .addresses
            .retain(|kept| now.duration_since(kept.last_answered) <= MAX_ADDRESS_AGE);
        known
            .addresses
            .sort_by_key(|kept| std::cmp::Reverse(kept.last_answered));
        known.addresses.truncate(MAX_ADDRESSES_PER_HOST);
        known.next %= known.addresses.len();
    }

    /// Record that a connection to `address` of `host` failed or stalled, so
    /// the next connections go elsewhere for a while.
    pub fn record_failure(&mut self, host: &str, address: IpAddr, now: Instant) {
        let Some(known) = self.hosts.get_mut(host) else {
            return;
        };
        if let Some(kept) = known
            .addresses
            .iter_mut()
            .find(|kept| kept.address == address)
        {
            kept.avoided_until = now + AVOID_FAILED_ADDRESS_FOR;
        }
    }
}
