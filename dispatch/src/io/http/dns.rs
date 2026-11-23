//! Host name lookups for the io_uring HTTP engine, and what each host last
//! resolved to.
//!
//! The lookups are done by c-ares on the worker's own thread. c-ares owns no
//! thread and never waits on the network: it names the sockets it wants
//! watched, [`HostResolver`] polls them on the worker's ring, and c-ares reads
//! and writes them without blocking once the ring reports them ready. It
//! follows `/etc/resolv.conf` and `/etc/hosts` as they were when the engine
//! started, asks for IPv4 and IPv6 addresses, repeats a question that got no
//! answer on the next name server, and asks again over TCP when an answer was
//! cut short. It keeps no answers: [`AddressBook`] holds what each host
//! resolved to, and has a host looked up again once the time to live of its
//! answer has passed.

use super::completion_tags::{HTTP_TAG, LOOKUP_TAG, LOOKUP_TIMER_TAG, POLL_CANCEL_TAG};
use c_ares::{AddrInfoHints, Channel, FdEventFlags, FdEvents, Options, ProcessFlags, Socket};
use io_uring::{IoUring, opcode, squeue, types};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a name server is given to answer the first time it is asked. A
/// name server on the machine or its network answers in a few milliseconds.
/// c-ares takes nothing shorter, and waits longer for each repeat.
const FIRST_ANSWER_DEADLINE: Duration = Duration::from_millis(250);

/// How many times each name server is asked before the lookup fails.
const ATTEMPTS_PER_NAME_SERVER: u32 = 4;

/// The soonest a host is looked up again after an answer, however short the
/// answer's time to live. A name server that keeps answers reports the time
/// a record has left, which can be nothing at all.
const SHORTEST_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

/// The latest a host is looked up again after an answer, however long the
/// answer's time to live, so a changed record is picked up within this long.
const LONGEST_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// How long a host's addresses are kept after its last answer. Past this a
/// connection waits for a new lookup. It outlasts a few failed refreshes.
const FORGET_ADDRESSES_AFTER: Duration = Duration::from_secs(60);

/// How long after its last request a host is still refreshed. Past this the
/// host is forgotten, and its next request waits for a lookup.
const FORGET_UNUSED_HOST_AFTER: Duration = Duration::from_secs(30);

/// The most hosts a worker keeps addresses for. A worker reads a few stores
/// at a time, and each host kept is looked up again every refresh interval.
const MAX_HOSTS: usize = 16;

/// The most addresses kept for one host.
const MAX_ADDRESSES_PER_HOST: usize = 32;

/// How long an address is passed over after a connection to it failed or
/// stalled.
const AVOID_FAILED_ADDRESS_FOR: Duration = Duration::from_secs(10);

/// A socket of c-ares that the ring polls for it.
struct WatchedSocket {
    socket: Socket,
    /// The poll events c-ares wants to hear of now, or none once it has
    /// closed the socket.
    wanted_events: u32,
    /// The events of the poll the ring holds for the socket, if it holds one.
    polled_events: Option<u32>,
    /// Whether the ring was asked to cancel that poll.
    cancelling: bool,
}

/// What c-ares has reported from inside its calls and the resolver has not
/// acted on yet.
#[derive(Default)]
struct Reported {
    /// Each socket whose watching changed with the poll events now wanted,
    /// in the order c-ares reported them.
    socket_changes: Vec<(Socket, u32)>,
    /// Each finished lookup's host with what was found for it.
    answers: Vec<(String, Result<LookupAnswer, c_ares::Error>)>,
}

/// Looks hosts up through c-ares, driven by one worker's ring.
pub(crate) struct HostResolver {
    channel: Channel,
    /// Filled by the callbacks c-ares runs inside the resolver's calls to it.
    /// The lock is never contended: those calls are all on the worker.
    reported: Arc<Mutex<Reported>>,
    /// Stable-index slab of the sockets polled for c-ares; the index (tagged
    /// with [`LOOKUP_TAG`]) is the `user_data` of a socket's poll.
    watched: Vec<Option<WatchedSocket>>,
    free_watched_slots: Vec<usize>,
    /// The timers in the ring, each with when it fires. Boxed so the address
    /// a timer's SQE carries stays put while the list grows.
    timers: Vec<(Instant, Box<types::Timespec>)>,
}

impl HostResolver {
    /// A resolver that asks the name servers of `/etc/resolv.conf`.
    pub fn new() -> Result<Self, c_ares::Error> {
        let reported = Arc::new(Mutex::new(Reported::default()));
        let reported_by_callback = reported.clone();
        let mut options = Options::new();
        options
            .set_timeout(FIRST_ANSWER_DEADLINE)
            .set_tries(ATTEMPTS_PER_NAME_SERVER)
            // The address book decides when a host is asked about again, so
            // c-ares keeps no answers of its own.
            .set_query_cache_max_ttl(0)
            .set_socket_state_callback(move |socket, readable, writable| {
                let mut wanted_events = 0;
                if readable {
                    wanted_events |= libc::POLLIN as u32;
                }
                if writable {
                    wanted_events |= libc::POLLOUT as u32;
                }
                reported_by_callback
                    .lock()
                    .unwrap()
                    .socket_changes
                    .push((socket, wanted_events));
            });
        Ok(Self {
            channel: Channel::with_options(options)?,
            reported,
            watched: Vec::new(),
            free_watched_slots: Vec::new(),
            timers: Vec::new(),
        })
    }

    /// Ask `servers` instead of the machine's name servers, so a test
    /// controls what a host resolves to.
    #[cfg(test)]
    pub fn force_name_servers(&mut self, servers: Vec<std::net::SocketAddr>) {
        self.channel
            .set_servers(servers.iter().map(|server| server.to_string()))
            .unwrap();
    }

    /// Start looking `host` up. Its answer is among
    /// [`take_answers`](Self::take_answers) once it is known, which is
    /// already on return for a host of `/etc/hosts`.
    pub fn start_lookup(&mut self, ring: &mut IoUring, host: &str) -> std::io::Result<()> {
        let reported = self.reported.clone();
        let looked_up_host = host.to_string();
        let hints = AddrInfoHints {
            socktype: libc::SOCK_STREAM,
            ..Default::default()
        };
        self.channel.get_addrinfo(host, None, &hints, move |found| {
            let answer = found.and_then(|found| {
                let addresses: Vec<IpAddr> =
                    found.nodes().filter_map(|node| node.ip_addr()).collect();
                if addresses.is_empty() {
                    return Err(c_ares::Error::ENODATA);
                }
                let seconds_to_live = found.nodes().map(|node| node.ttl()).min().unwrap_or(0);
                Ok(LookupAnswer {
                    addresses,
                    time_to_live: Duration::from_secs(seconds_to_live.max(0) as u64),
                })
            });
            reported
                .lock()
                .unwrap()
                .answers
                .push((looked_up_host, answer));
        });
        self.follow_sockets_and_deadlines(ring)
    }

    /// The lookups that finished since the last call, each host with what
    /// was found for it.
    pub fn take_answers(&mut self) -> Vec<(String, Result<LookupAnswer, c_ares::Error>)> {
        std::mem::take(&mut self.reported.lock().unwrap().answers)
    }

    /// Let c-ares act on the completion of a socket's poll or of a timer.
    pub fn on_cqe(
        &mut self,
        ring: &mut IoUring,
        user_data: u64,
        result: i32,
    ) -> std::io::Result<()> {
        if user_data & POLL_CANCEL_TAG != 0 {
            return Ok(());
        }
        let mut ready = Vec::new();
        if user_data & LOOKUP_TIMER_TAG != 0 {
            // Timers fire in the order they are due.
            let earliest = (0..self.timers.len())
                .min_by_key(|&index| self.timers[index].0)
                .expect("cqe for a timer in the ring");
            self.timers.swap_remove(earliest);
        } else {
            let slot = (user_data & !(HTTP_TAG | LOOKUP_TAG)) as usize;
            let watched = self.watched[slot]
                .as_mut()
                .expect("cqe for a watched socket");
            watched.polled_events = None;
            watched.cancelling = false;
            if watched.wanted_events != 0 && result != -libc::ECANCELED {
                // A poll that failed is passed on as readable, so c-ares
                // meets the error when it reads the socket.
                let ready_events = if result < 0 {
                    libc::POLLERR as u32
                } else {
                    result as u32
                };
                let mut events = FdEventFlags::empty();
                if ready_events & (libc::POLLIN | libc::POLLERR | libc::POLLHUP) as u32 != 0 {
                    events |= FdEventFlags::READ;
                }
                if ready_events & libc::POLLOUT as u32 != 0 {
                    events |= FdEventFlags::WRITE;
                }
                ready.push(FdEvents::new(watched.socket, events));
            }
        }
        // With no socket ready this still gives up on or repeats the
        // questions whose deadline has passed.
        self.channel
            .process_fds(&ready, ProcessFlags::empty())
            .map_err(std::io::Error::other)?;
        self.follow_sockets_and_deadlines(ring)
    }

    /// Bring the ring's polls in line with the sockets c-ares now wants
    /// watched, and have a timer fire by its next deadline.
    fn follow_sockets_and_deadlines(&mut self, ring: &mut IoUring) -> std::io::Result<()> {
        let socket_changes = std::mem::take(&mut self.reported.lock().unwrap().socket_changes);
        for (socket, wanted_events) in socket_changes {
            // A closed socket's number may be given to a new one while the
            // old poll is still in the ring, so only a socket still wanted
            // is the one c-ares means.
            let known = self
                .watched
                .iter_mut()
                .flatten()
                .find(|watched| watched.socket == socket && watched.wanted_events != 0);
            match known {
                Some(watched) => watched.wanted_events = wanted_events,
                None if wanted_events != 0 => {
                    let watched = Some(WatchedSocket {
                        socket,
                        wanted_events,
                        polled_events: None,
                        cancelling: false,
                    });
                    match self.free_watched_slots.pop() {
                        Some(slot) => self.watched[slot] = watched,
                        None => self.watched.push(watched),
                    }
                }
                None => {}
            }
        }

        for slot in 0..self.watched.len() {
            let Some(watched) = &mut self.watched[slot] else {
                continue;
            };
            let poll_user_data = HTTP_TAG | LOOKUP_TAG | slot as u64;
            match watched.polled_events {
                None if watched.wanted_events == 0 => {
                    self.watched[slot] = None;
                    self.free_watched_slots.push(slot);
                }
                None => {
                    watched.polled_events = Some(watched.wanted_events);
                    let poll =
                        opcode::PollAdd::new(types::Fd(watched.socket), watched.wanted_events)
                            .build()
                            .user_data(poll_user_data);
                    Self::submit(ring, &poll)?;
                }
                // The poll in the ring waits for other events than c-ares
                // wants now. Its completion brings the socket back here.
                Some(polled_events)
                    if polled_events != watched.wanted_events && !watched.cancelling =>
                {
                    watched.cancelling = true;
                    let cancel = opcode::AsyncCancel::new(poll_user_data)
                        .build()
                        .user_data(HTTP_TAG | LOOKUP_TAG | POLL_CANCEL_TAG);
                    Self::submit(ring, &cancel)?;
                }
                Some(_) => {}
            }
        }

        if let Some(until_deadline) = self.channel.timeout(None) {
            let deadline = Instant::now() + until_deadline;
            // A timer that fires earlier comes back here and sets the next.
            if !self
                .timers
                .iter()
                .any(|(fires_at, _)| *fires_at <= deadline)
            {
                let wait = Box::new(
                    types::Timespec::new()
                        .sec(until_deadline.as_secs())
                        .nsec(until_deadline.subsec_nanos()),
                );
                let timer = opcode::Timeout::new(&*wait as *const types::Timespec)
                    .build()
                    .user_data(HTTP_TAG | LOOKUP_TAG | LOOKUP_TIMER_TAG);
                self.timers.push((deadline, wait));
                Self::submit(ring, &timer)?;
            }
        }
        Ok(())
    }

    /// Push one SQE onto the worker's ring and flush.
    fn submit(ring: &mut IoUring, entry: &squeue::Entry) -> std::io::Result<()> {
        unsafe {
            ring.submission()
                .push(entry)
                .map_err(|_| std::io::Error::other("http submission queue full"))?;
        }
        ring.submit()?;
        Ok(())
    }
}

/// The addresses a lookup found for a host.
pub(crate) struct LookupAnswer {
    /// In the order the machine prefers to connect to them.
    pub addresses: Vec<IpAddr>,
    /// How long the name server said the answer holds, which is the shortest
    /// time to live among its records.
    pub time_to_live: Duration,
}

struct KnownAddress {
    address: IpAddr,
    last_answered: Instant,
    avoided_until: Instant,
}

struct HostAddresses {
    addresses: Vec<KnownAddress>,
    answered_at: Instant,
    /// When the host is looked up again.
    refresh_at: Instant,
    /// When a request for the host last started.
    last_used: Instant,
    /// Index of the address the next connection goes to.
    next: usize,
    /// Whether the machine prefers to reach the host over IPv6, which is
    /// the family of the best address in its last answer.
    prefers_ipv6: bool,
}

/// The addresses each host resolved to in its recent answers, for one
/// worker. Connections to a host go in turn to its addresses of the family
/// the machine prefers, so they spread over the store's servers. An address a
/// connection just failed on is passed over for a while, and so is its whole
/// family when the host has addresses of the other family that have not
/// failed.
///
/// A host is due for another lookup once its answer's time to live has
/// passed, and the worker's loop starts that lookup whether or not a
/// connection is opening. A host no request has used for a while is
/// forgotten, and so are addresses whose last answer is too old. The book
/// holds a limited number of hosts, and a new host takes the place of the
/// one whose last request is the oldest.
pub(crate) struct AddressBook {
    hosts: HashMap<String, HostAddresses>,
    shortest_refresh_interval: Duration,
    longest_refresh_interval: Duration,
}

impl AddressBook {
    pub fn new() -> Self {
        Self {
            hosts: HashMap::new(),
            shortest_refresh_interval: SHORTEST_REFRESH_INTERVAL,
            longest_refresh_interval: LONGEST_REFRESH_INTERVAL,
        }
    }

    /// Look a host up again between `shortest` and `longest` after its
    /// answer, so a test controls how soon an answer is refreshed.
    #[cfg(test)]
    pub fn force_refresh_intervals(&mut self, shortest: Duration, longest: Duration) {
        self.shortest_refresh_interval = shortest;
        self.longest_refresh_interval = longest;
    }

    /// Whether the book holds no host at all.
    pub fn is_empty(&self) -> bool {
        self.hosts.is_empty()
    }

    /// Record that a request for `host` started, so the host keeps being
    /// refreshed.
    pub fn record_use(&mut self, host: &str, now: Instant) {
        // A host written as an address, or one not looked up yet, has no
        // entry to keep.
        if let Some(known) = self.hosts.get_mut(host) {
            known.last_used = now;
        }
    }

    /// Choose the address the next connection to `host` goes to. There is
    /// none for a host that was never answered or whose last answer is too
    /// old, and such a host is looked up first.
    pub fn choose_address(&mut self, host: &str, now: Instant) -> Option<IpAddr> {
        let known = self.hosts.get_mut(host)?;
        if now.duration_since(known.answered_at) > FORGET_ADDRESSES_AFTER {
            self.hosts.remove(host);
            return None;
        }
        let addresses = &known.addresses;
        let has_family = |ipv6: bool| addresses.iter().any(|kept| kept.address.is_ipv6() == ipv6);
        let family_failed_lately = |ipv6: bool| {
            addresses
                .iter()
                .any(|kept| kept.address.is_ipv6() == ipv6 && kept.avoided_until > now)
        };
        let preferred = known.prefers_ipv6;
        let other_family_is_better = has_family(!preferred)
            && !family_failed_lately(!preferred)
            && family_failed_lately(preferred);
        let chosen_is_ipv6 = if has_family(preferred) && !other_family_is_better {
            preferred
        } else {
            !preferred
        };
        // The first address of the family in turn that no connection failed
        // on lately. When every one failed lately, the one in turn is tried
        // again.
        let count = addresses.len();
        let in_turn = || {
            (0..count)
                .map(|offset| (known.next + offset) % count)
                .filter(|&index| addresses[index].address.is_ipv6() == chosen_is_ipv6)
        };
        let chosen = in_turn()
            .find(|&index| addresses[index].avoided_until <= now)
            .or_else(|| in_turn().next())
            .expect("the chosen family has an address");
        known.next = (chosen + 1) % count;
        Some(known.addresses[chosen].address)
    }

    /// The hosts whose answer's time to live has passed, each to be looked
    /// up again. A host no request has used for a while is forgotten
    /// instead.
    pub fn take_hosts_due_for_refresh(&mut self, now: Instant) -> Vec<String> {
        let retry_interval = self.shortest_refresh_interval;
        let mut due = Vec::new();
        self.hosts.retain(|host, known| {
            if now.duration_since(known.last_used) > FORGET_UNUSED_HOST_AFTER {
                return false;
            }
            if known.refresh_at <= now {
                // A lookup that finds nothing is repeated after this long.
                known.refresh_at = now + retry_interval;
                due.push(host.clone());
            }
            true
        });
        due
    }

    /// Record the answer of a lookup of `host`. Addresses from its earlier
    /// answers stay until no answer has listed them for a while. A host new
    /// to a full book takes the place of the host whose last request is the
    /// oldest, and that host's next connection waits for a lookup.
    pub fn record_answer(&mut self, host: &str, answer: &LookupAnswer, now: Instant) {
        if self.hosts.len() >= MAX_HOSTS && !self.hosts.contains_key(host) {
            let longest_unused = self
                .hosts
                .iter()
                .min_by_key(|(_, known)| known.last_used)
                .map(|(host, _)| host.clone())
                .expect("a full book holds a host");
            self.hosts.remove(&longest_unused);
        }
        let known = self
            .hosts
            .entry(host.to_string())
            .or_insert_with(|| HostAddresses {
                addresses: Vec::new(),
                answered_at: now,
                refresh_at: now,
                last_used: now,
                // Workers start at different addresses of the same answer.
                next: rand::random::<u32>() as usize,
                prefers_ipv6: false,
            });
        known.answered_at = now;
        known.refresh_at = now
            + answer.time_to_live.clamp(
                self.shortest_refresh_interval,
                self.longest_refresh_interval,
            );
        known.prefers_ipv6 = answer.addresses[0].is_ipv6();
        for address in &answer.addresses {
            match known
                .addresses
                .iter_mut()
                .find(|kept| kept.address == *address)
            {
                Some(kept) => kept.last_answered = now,
                None => known.addresses.push(KnownAddress {
                    address: *address,
                    last_answered: now,
                    avoided_until: now,
                }),
            }
        }
        known
            .addresses
            .retain(|kept| now.duration_since(kept.last_answered) <= FORGET_ADDRESSES_AFTER);
        known
            .addresses
            .sort_by_key(|kept| std::cmp::Reverse(kept.last_answered));
        known.addresses.truncate(MAX_ADDRESSES_PER_HOST);
        known.next %= known.addresses.len();
    }

    /// Record that a connection to `address` of `host` failed or stalled, so
    /// the next connections go elsewhere for a while.
    pub fn record_failure(&mut self, host: &str, address: IpAddr, now: Instant) {
        // A host written as an address, or one forgotten since the
        // connection opened, has nothing to pass over.
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
    use std::net::{Ipv4Addr, Ipv6Addr};

    fn build_ipv4(last: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, last))
    }

    fn build_ipv6(last: u16) -> IpAddr {
        IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, last))
    }

    /// A book that was just answered `addresses` for `store.test`, to hold
    /// for `time_to_live`.
    fn build_answered_book(
        addresses: &[IpAddr],
        time_to_live: Duration,
        now: Instant,
    ) -> AddressBook {
        let mut book = AddressBook::new();
        let answer = LookupAnswer {
            addresses: addresses.to_vec(),
            time_to_live,
        };
        book.record_answer("store.test", &answer, now);
        book
    }

    #[test]
    fn connections_go_to_the_addresses_of_a_host_in_turn() {
        let now = Instant::now();
        let addresses = [build_ipv4(1), build_ipv4(2), build_ipv4(3)];
        let mut book = build_answered_book(&addresses, Duration::from_secs(5), now);

        let mut chosen = [(); 3].map(|_| book.choose_address("store.test", now));

        chosen.sort();
        assert_eq!(chosen, addresses.map(Some));
    }

    #[test]
    fn an_address_a_connection_failed_on_is_passed_over() {
        let now = Instant::now();
        let (failed, healthy) = (build_ipv4(1), build_ipv4(2));
        let mut book = build_answered_book(&[failed, healthy], Duration::from_secs(5), now);

        book.record_failure("store.test", failed, now);

        let chosen = [(); 4].map(|_| book.choose_address("store.test", now));
        assert_eq!(chosen, [Some(healthy); 4]);
    }

    #[test]
    fn connections_stay_in_the_family_of_the_best_address() {
        let now = Instant::now();
        let preferred = build_ipv6(1);
        let addresses = [preferred, build_ipv4(1), build_ipv4(2)];
        let mut book = build_answered_book(&addresses, Duration::from_secs(5), now);

        let chosen = [(); 3].map(|_| book.choose_address("store.test", now));

        assert_eq!(chosen, [Some(preferred); 3]);
    }

    #[test]
    fn a_family_with_a_failed_address_yields_to_the_other_family() {
        let now = Instant::now();
        let (failed, untried, other_family) = (build_ipv6(1), build_ipv6(2), build_ipv4(1));
        let addresses = [failed, untried, other_family];
        let mut book = build_answered_book(&addresses, Duration::from_secs(5), now);

        book.record_failure("store.test", failed, now);

        let chosen = [(); 3].map(|_| book.choose_address("store.test", now));
        assert_eq!(chosen, [Some(other_family); 3]);
    }

    #[test]
    fn a_host_is_due_for_refresh_once_its_time_to_live_has_passed() {
        let now = Instant::now();
        let mut book = build_answered_book(&[build_ipv4(1)], Duration::from_secs(10), now);

        let before = book.take_hosts_due_for_refresh(now + Duration::from_secs(9));
        let after = book.take_hosts_due_for_refresh(now + Duration::from_secs(11));

        assert_eq!(before, Vec::<String>::new());
        assert_eq!(after, ["store.test"]);
    }

    #[test]
    fn a_time_to_live_of_nothing_still_waits_the_shortest_interval() {
        let now = Instant::now();
        let mut book = build_answered_book(&[build_ipv4(1)], Duration::ZERO, now);

        let before = book.take_hosts_due_for_refresh(now + SHORTEST_REFRESH_INTERVAL / 2);
        let after = book.take_hosts_due_for_refresh(now + SHORTEST_REFRESH_INTERVAL * 2);

        assert_eq!(before, Vec::<String>::new());
        assert_eq!(after, ["store.test"]);
    }

    #[test]
    fn a_long_time_to_live_is_refreshed_after_the_longest_interval() {
        let now = Instant::now();
        let mut book = build_answered_book(&[build_ipv4(1)], Duration::from_secs(3600), now);
        book.record_use("store.test", now + LONGEST_REFRESH_INTERVAL);

        let due = book.take_hosts_due_for_refresh(now + LONGEST_REFRESH_INTERVAL * 2);

        assert_eq!(due, ["store.test"]);
    }

    #[test]
    fn a_host_no_request_uses_is_forgotten_instead_of_refreshed() {
        let now = Instant::now();
        let mut book = build_answered_book(&[build_ipv4(1)], Duration::from_secs(5), now);
        let later = now + FORGET_UNUSED_HOST_AFTER * 2;

        let due = book.take_hosts_due_for_refresh(later);

        assert_eq!(due, Vec::<String>::new());
        assert_eq!(book.choose_address("store.test", later), None);
    }

    #[test]
    fn a_host_new_to_a_full_book_takes_the_place_of_the_longest_unused() {
        let now = Instant::now();
        let answer = LookupAnswer {
            addresses: vec![build_ipv4(1)],
            time_to_live: Duration::from_secs(5),
        };
        let mut book = AddressBook::new();
        for index in 0..MAX_HOSTS {
            book.record_answer(&format!("store-{index}.test"), &answer, now);
        }
        for index in 1..MAX_HOSTS {
            book.record_use(&format!("store-{index}.test"), now + Duration::from_secs(1));
        }

        book.record_answer("new.test", &answer, now + Duration::from_secs(2));

        let later = now + Duration::from_secs(2);
        assert_eq!(book.choose_address("store-0.test", later), None);
        assert_eq!(
            book.choose_address("store-1.test", later),
            Some(build_ipv4(1))
        );
        assert_eq!(book.choose_address("new.test", later), Some(build_ipv4(1)));
    }

    #[test]
    fn addresses_whose_answer_is_too_old_are_not_used() {
        let now = Instant::now();
        let mut book = build_answered_book(&[build_ipv4(1)], Duration::from_secs(5), now);

        let chosen = book.choose_address("store.test", now + FORGET_ADDRESSES_AFTER * 2);

        assert_eq!(chosen, None);
    }
}
