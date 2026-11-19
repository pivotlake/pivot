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
const REFRESH_ADDRESSES_AFTER: Duration = Duration::from_secs(5);

/// How long an address is kept after a name server last gave it. A store
/// keeps serving at an address well after it stops publishing it, and
/// keeping addresses from several answers spreads connections over more of
/// the store's servers.
const FORGET_ADDRESSES_AFTER: Duration = Duration::from_secs(30);

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
/// The answer code of a name server that knows the name does not exist.
const NAME_UNKNOWN_CODE: u8 = 3;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DnsError {
    #[error("{host} cannot be resolved: /etc/resolv.conf names no name server")]
    NoNameServer { host: String },
    #[error("{host} is not a host name a name server can be asked about")]
    InvalidHostName { host: String },
    #[error("no name server answered for {host} in {attempts} attempts")]
    NoAnswer { host: String, attempts: u32 },
    #[error("no name server knows an address for {host}")]
    HostNotFound { host: String },
    #[error("the name server's answer is malformed")]
    MalformedAnswer,
    #[error("the name server's answer is for another question")]
    AnswerToAnotherQuestion,
    #[error("the name server's answer does not fit in one datagram")]
    TruncatedAnswer,
    #[error("the name server failed to answer, with code {code}")]
    ServerFailed { code: u8 },
    #[error("the name server knows no such name")]
    NameUnknown,
    #[error("the name server knows no address for the name")]
    NoAddresses,
}

/// The name servers to ask and the hosts whose address needs no asking, as
/// the machine configures them.
pub(crate) struct NameServers {
    servers: Vec<SocketAddr>,
    /// Domains a short host name is completed with, in the order they are
    /// tried.
    search_domains: Vec<String>,
    /// A host name with fewer dots than this is tried with the search
    /// domains before it is tried as written.
    dots_of_a_complete_name: usize,
    /// Host names with a fixed address, in lower case.
    fixed_addresses: HashMap<String, IpAddr>,
}

impl NameServers {
    /// Read the machine's name servers and search domains from
    /// `/etc/resolv.conf` and its fixed host addresses from `/etc/hosts`. A
    /// machine without one of the files has none of what the file lists.
    pub fn load() -> Self {
        let resolver_configuration =
            std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
        let hosts_file = std::fs::read_to_string("/etc/hosts").unwrap_or_default();
        Self::parse(&resolver_configuration, &hosts_file)
    }

    pub fn parse(resolver_configuration: &str, hosts_file: &str) -> Self {
        let mut servers = Vec::new();
        let mut search_domains = Vec::new();
        let mut dots_of_a_complete_name = 1;
        for line in resolver_configuration.lines() {
            let mut words = line.split_whitespace();
            match words.next() {
                Some("nameserver") => {
                    if let Some(Ok(address)) = words.next().map(str::parse::<IpAddr>) {
                        servers.push(SocketAddr::new(address, 53));
                    }
                }
                // The last of these lines is the one in effect.
                Some("search" | "domain") => {
                    search_domains = words
                        .map(|domain| domain.trim_end_matches('.').to_string())
                        .collect();
                }
                Some("options") => {
                    for option in words {
                        if let Some(Ok(dots)) =
                            option.strip_prefix("ndots:").map(str::parse::<usize>)
                        {
                            dots_of_a_complete_name = dots;
                        }
                    }
                }
                _ => {}
            }
        }
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
            search_domains,
            dots_of_a_complete_name,
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
        let name = host.trim_end_matches('.').to_ascii_lowercase();
        self.fixed_addresses.get(&name).copied()
    }

    /// The names a name server is asked about for `host`, in order, until one
    /// has an address. A host written with a final dot is complete and is
    /// asked about as written. Any other is also tried with each of the
    /// machine's search domains appended, before the name as written when it
    /// has few dots and after it otherwise.
    pub fn list_names_to_ask(&self, host: &str) -> Vec<String> {
        if host.ends_with('.') {
            return vec![host.to_string()];
        }
        let completed = self
            .search_domains
            .iter()
            .map(|domain| format!("{host}.{domain}"));
        let dots = host.matches('.').count();
        if dots >= self.dots_of_a_complete_name {
            std::iter::once(host.to_string()).chain(completed).collect()
        } else {
            completed.chain(std::iter::once(host.to_string())).collect()
        }
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
    match (flags & 0x000F) as u8 {
        0 => {}
        NAME_UNKNOWN_CODE => return Err(DnsError::NameUnknown),
        code => return Err(DnsError::ServerFailed { code }),
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
#[derive(Debug, PartialEq)]
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

/// The addresses each host resolved to in its recent answers, for one
/// worker. Connections to a host go to its addresses in turn, so they spread
/// over the store's servers, and an address a connection just failed on is
/// passed over for a while. A host whose last answer is too old is forgotten
/// and has to be looked up again.
pub(crate) struct AddressBook {
    hosts: HashMap<String, HostAddresses>,
    refresh_after: Duration,
    forget_after: Duration,
}

impl AddressBook {
    pub fn new() -> Self {
        Self {
            hosts: HashMap::new(),
            refresh_after: REFRESH_ADDRESSES_AFTER,
            forget_after: FORGET_ADDRESSES_AFTER,
        }
    }

    /// Look a host up again once its answer is `refresh_after` old, and
    /// forget what it resolved to once the answer is `forget_after` old.
    #[cfg(test)]
    pub fn force_address_ages(&mut self, refresh_after: Duration, forget_after: Duration) {
        self.refresh_after = refresh_after;
        self.forget_after = forget_after;
    }

    /// Choose the address the next connection to `host` goes to.
    pub fn choose_address(&mut self, host: &str, now: Instant) -> AddressChoice {
        let Some(known) = self.hosts.get_mut(host) else {
            return AddressChoice::Unknown;
        };
        let age = now.duration_since(known.answered_at);
        if age > self.forget_after {
            self.hosts.remove(host);
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

    /// Record a name server's answer for `host`. Addresses from its earlier
    /// answers stay until no answer has listed them for a while, and every
    /// other host whose last answer has grown too old is forgotten.
    pub fn record_answer(&mut self, host: &str, answered: &[Ipv4Addr], now: Instant) {
        let forget_after = self.forget_after;
        self.hosts
            .retain(|_, known| now.duration_since(known.answered_at) <= forget_after);
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
            .retain(|kept| now.duration_since(kept.last_answered) <= forget_after);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An answer to a question with id 7 about `a.test`, carrying an alias
    /// record and then one address record for each of `addresses`.
    fn build_answer(flags: [u8; 2], addresses: &[[u8; 4]]) -> Vec<u8> {
        let mut answer = vec![0, 7, flags[0], flags[1], 0, 1, 0, 0, 0, 0, 0, 0];
        answer[7] = addresses.len() as u8 + 1;
        answer.extend_from_slice(&[1, b'a', 4, b't', b'e', b's', b't', 0, 0, 1, 0, 1]);
        // The alias record: a pointer to the name, the alias type, the
        // internet class, a time to live, and the two-byte name it points to.
        answer.extend_from_slice(&[0xC0, 12, 0, 5, 0, 1, 0, 0, 0, 5, 0, 2, 0xC0, 12]);
        for address in addresses {
            answer.extend_from_slice(&[0xC0, 12, 0, 1, 0, 1, 0, 0, 0, 5, 0, 4]);
            answer.extend_from_slice(address);
        }
        answer
    }

    #[test]
    fn an_answer_with_an_alias_gives_the_addresses_behind_it() {
        let answer = build_answer([0x81, 0x80], &[[10, 0, 0, 1], [10, 0, 0, 2]]);

        let addresses = parse_answer(&answer, 7);

        assert_eq!(
            addresses,
            Ok(vec![Ipv4Addr::new(10, 0, 0, 1), Ipv4Addr::new(10, 0, 0, 2)])
        );
    }

    #[test]
    fn an_answer_cut_short_anywhere_is_an_error() {
        let answer = build_answer([0x81, 0x80], &[[10, 0, 0, 1]]);

        let outcomes: Vec<_> = (0..answer.len())
            .map(|len| parse_answer(&answer[..len], 7))
            .collect();

        assert!(outcomes.iter().all(|outcome| outcome.is_err()));
    }

    #[test]
    fn an_answer_says_why_it_carries_no_address() {
        let for_another_question = parse_answer(&build_answer([0x81, 0x80], &[[10, 0, 0, 1]]), 8);
        let truncated = parse_answer(&build_answer([0x83, 0x80], &[]), 7);
        let server_failed = parse_answer(&build_answer([0x81, 0x82], &[]), 7);
        let name_unknown = parse_answer(&build_answer([0x81, 0x83], &[]), 7);
        let no_addresses = parse_answer(&build_answer([0x81, 0x80], &[]), 7);

        assert_eq!(for_another_question, Err(DnsError::AnswerToAnotherQuestion));
        assert_eq!(truncated, Err(DnsError::TruncatedAnswer));
        assert_eq!(server_failed, Err(DnsError::ServerFailed { code: 2 }));
        assert_eq!(name_unknown, Err(DnsError::NameUnknown));
        assert_eq!(no_addresses, Err(DnsError::NoAddresses));
    }

    #[test]
    fn a_short_host_name_is_tried_with_the_search_domains_first() {
        let name_servers = NameServers::parse(
            "nameserver 10.0.0.2\nsearch storage.svc.cluster.local svc.cluster.local\noptions ndots:2\n",
            "",
        );

        let short = name_servers.list_names_to_ask("minio.storage");
        let long = name_servers.list_names_to_ask("bucket.s3.amazonaws.com");
        let complete = name_servers.list_names_to_ask("minio.");

        assert_eq!(
            short,
            [
                "minio.storage.storage.svc.cluster.local",
                "minio.storage.svc.cluster.local",
                "minio.storage"
            ]
        );
        assert_eq!(long[0], "bucket.s3.amazonaws.com");
        assert_eq!(complete, ["minio."]);
    }

    #[test]
    fn a_host_in_the_hosts_file_or_written_as_an_address_needs_no_name_server() {
        let name_servers = NameServers::parse(
            "",
            "::1 localhost\n127.0.0.1 localhost # loopback\n10.0.0.9 Store.Internal\n",
        );

        let listed_twice = name_servers.find_fixed_address("localhost");
        let other_case = name_servers.find_fixed_address("store.internal.");
        let written_as_address = name_servers.find_fixed_address("[::1]");
        let unlisted = name_servers.find_fixed_address("elsewhere.test");

        assert_eq!(listed_twice, Some(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert_eq!(other_case, Some("10.0.0.9".parse().unwrap()));
        assert_eq!(written_as_address, Some("::1".parse().unwrap()));
        assert_eq!(unlisted, None);
    }

    #[test]
    fn connections_go_to_the_addresses_of_a_host_in_turn() {
        let now = Instant::now();
        let mut book = AddressBook::new();
        let answered = [Ipv4Addr::new(10, 0, 0, 1), Ipv4Addr::new(10, 0, 0, 2)];
        book.record_answer("store.test", &answered, now);

        let chosen: Vec<_> = (0..4)
            .map(|_| book.choose_address("store.test", now))
            .collect();

        assert_ne!(chosen[0], chosen[1]);
        assert_eq!(chosen[0], chosen[2]);
        assert_eq!(chosen[1], chosen[3]);
    }

    #[test]
    fn an_address_a_connection_failed_on_is_passed_over() {
        let now = Instant::now();
        let mut book = AddressBook::new();
        let failed = Ipv4Addr::new(10, 0, 0, 1);
        let healthy = Ipv4Addr::new(10, 0, 0, 2);
        book.record_answer("store.test", &[failed, healthy], now);

        book.record_failure("store.test", IpAddr::V4(failed), now);

        let chosen: Vec<_> = (0..4)
            .map(|_| book.choose_address("store.test", now))
            .collect();
        assert_eq!(
            chosen,
            [(); 4].map(|_| AddressChoice::Fresh(IpAddr::V4(healthy)))
        );
    }

    #[test]
    fn an_answer_ages_into_a_refresh_and_then_is_forgotten() {
        let answered_at = Instant::now();
        let mut book = AddressBook::new();
        let address = Ipv4Addr::new(10, 0, 0, 1);
        book.record_answer("store.test", &[address], answered_at);

        let fresh = book.choose_address("store.test", answered_at + Duration::from_secs(4));
        let due = book.choose_address("store.test", answered_at + Duration::from_secs(6));
        let forgotten = book.choose_address("store.test", answered_at + Duration::from_secs(31));

        assert_eq!(fresh, AddressChoice::Fresh(IpAddr::V4(address)));
        assert_eq!(due, AddressChoice::DueForRefresh(IpAddr::V4(address)));
        assert_eq!(forgotten, AddressChoice::Unknown);
    }

    #[test]
    fn a_host_keeps_no_more_addresses_than_the_most_recent() {
        let now = Instant::now();
        let mut book = AddressBook::new();
        let answered: Vec<_> = (0..40).map(|last| Ipv4Addr::new(10, 0, 0, last)).collect();

        book.record_answer("store.test", &answered, now);

        let chosen: std::collections::HashSet<_> = (0..100)
            .map(|_| book.choose_address("store.test", now))
            .map(|choice| format!("{choice:?}"))
            .collect();
        assert_eq!(chosen.len(), MAX_ADDRESSES_PER_HOST);
    }
}
