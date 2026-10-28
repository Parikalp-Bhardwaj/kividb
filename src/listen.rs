//! The ports clients connect to, and `CONFIG SET port` and `tls-port`
//! moving them (#24).
//!
//! Every accept loop wraps its listener in one of the types here. To move
//! the port, `CONFIG SET` binds a new socket for each loop, before it
//! answers: a port that is in use is refused then, with Redis's `Unable to
//! listen on this port`, and nothing has changed. Each loop then takes one
//! of the new sockets and drops its old one, which closes the old port.
//! Connections already open stay open.
//!
//! Port 0 is no port: a loop then has no socket and waits, until the port
//! is set again. That is how the TLS port is off, and how it is opened
//! with `CONFIG SET tls-port` on a server that started without it.
//!
//! Servers are told apart by the ports they were started on: a test
//! process runs many, each with its own.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use tokio::sync::watch;

/// Which of a server's ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `port`: plaintext.
    Plain,
    /// `tls-port`.
    Tls,
}

impl Kind {
    fn what(self) -> &'static str {
        match self {
            Kind::Plain => "connections",
            Kind::Tls => "TLS connections",
        }
    }
}

/// A server, by the `port` and `tls-port` it was started with.
pub type Server = (u16, u16);

/// The server `config` started.
pub fn server_of(config: &crate::config::Config) -> Server {
    (config.port, config.tls_port)
}

struct ServerPort {
    ip: IpAddr,
    /// What clients connect to now; 0 when there is none.
    current: AtomicU16,
    /// Accept loops listening: as many sockets are bound for a new port.
    loops: AtomicUsize,
    /// The io_uring workers share the port, each with a socket of its own.
    reuse_port: bool,
    /// Sockets bound for the new port, one for each loop to take.
    pending: Mutex<Vec<std::net::TcpListener>>,
    moved: watch::Sender<u64>,
}

type Ports = HashMap<(Server, Kind), Arc<ServerPort>>;

static SERVERS: LazyLock<Mutex<Ports>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn server_port(server: Server, kind: Kind) -> Option<Arc<ServerPort>> {
    SERVERS.lock().get(&(server, kind)).cloned()
}

/// The port of `kind` that `server` listens on now; 0 when it has none.
pub fn current_port(server: Server, kind: Kind) -> u16 {
    server_port(server, kind).map_or(
        match kind {
            Kind::Plain => server.0,
            Kind::Tls => server.1,
        },
        |s| s.current.load(Ordering::SeqCst),
    )
}

/// One accept loop's part: told when the port moves.
pub struct PortWatch {
    server: Arc<ServerPort>,
    moved: watch::Receiver<u64>,
}

impl PortWatch {
    fn register(
        server: Server,
        kind: Kind,
        addr: SocketAddr,
        open: bool,
        reuse_port: bool,
    ) -> Self {
        let server = Arc::clone(SERVERS.lock().entry((server, kind)).or_insert_with(|| {
            Arc::new(ServerPort {
                ip: addr.ip(),
                current: AtomicU16::new(if open { addr.port() } else { 0 }),
                loops: AtomicUsize::new(0),
                reuse_port,
                pending: Mutex::new(Vec::new()),
                moved: watch::channel(0).0,
            })
        }));
        server.loops.fetch_add(1, Ordering::SeqCst);
        let moved = server.moved.subscribe();
        PortWatch { server, moved }
    }

    /// The socket for the new port once the port has moved, or `None` once
    /// it is closed.
    async fn next(&mut self) -> Option<std::net::TcpListener> {
        loop {
            if self.moved.changed().await.is_err() {
                // The sender lives as long as `server`: never.
                std::future::pending::<()>().await;
            }
            if self.server.current.load(Ordering::SeqCst) == 0 {
                return None;
            }
            if let Some(socket) = self.server.pending.lock().pop() {
                return Some(socket);
            }
        }
    }
}

impl Drop for PortWatch {
    fn drop(&mut self) {
        self.server.loops.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `CONFIG SET port` or `tls-port`: listen on `port` instead, or on none
/// for 0, or say why not.
pub fn move_to(server: Server, kind: Kind, port: u16) -> Result<(), String> {
    let Some(server) = server_port(server, kind) else {
        return Err("the server is not listening".into());
    };
    let old = server.current.load(Ordering::SeqCst);
    if port == old {
        return Ok(());
    }
    if port == 0 {
        // Each accept loop lets its socket go as it wakes to this, a moment
        // after the reply. It is not waited for: on io_uring this runs on
        // the very thread of one of those loops.
        server.pending.lock().clear();
        server.current.store(0, Ordering::SeqCst);
        server.moved.send_modify(|n| *n += 1);
        crate::log_info!(
            "No longer accepting {} on {}; the ones open stay open",
            kind.what(),
            SocketAddr::new(server.ip, old)
        );
        return Ok(());
    }
    let addr = SocketAddr::new(server.ip, port);
    let loops = server.loops.load(Ordering::SeqCst).max(1);
    let mut sockets = Vec::with_capacity(loops);
    for _ in 0..loops {
        sockets.push(bind(addr, server.reuse_port).map_err(|e| e.to_string())?);
    }
    *server.pending.lock() = sockets;
    server.current.store(port, Ordering::SeqCst);
    server.moved.send_modify(|n| *n += 1);
    crate::log_info!("Ready to accept {} on {addr}", kind.what());
    Ok(())
}

fn bind(addr: SocketAddr, reuse_port: bool) -> std::io::Result<std::net::TcpListener> {
    let socket = || {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            None,
        )?;
        // As the listener the server started with: a port in TIME_WAIT is
        // free.
        #[cfg(unix)]
        socket.set_reuse_address(true)?;
        Ok::<_, std::io::Error>(socket)
    };
    #[cfg(target_os = "linux")]
    if reuse_port {
        // With SO_REUSEPORT the kernel lets any socket that set it share
        // the port, another server's too: a probe without it is refused if
        // anything at all holds the port.
        socket()?.bind(&addr.into())?;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = reuse_port;
    let socket = socket()?;
    #[cfg(target_os = "linux")]
    if reuse_port {
        socket.set_reuse_port(true)?;
    }
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    socket.set_nonblocking(true)?;
    Ok(socket.into())
}

/// A Tokio accept loop's listener.
pub struct Listener {
    inner: Option<tokio::net::TcpListener>,
    watch: PortWatch,
}

impl Listener {
    /// The loop of `server`'s port of `kind`, listening with `inner` on
    /// `addr`; with no `inner`, on no port until one is set.
    pub fn new(
        inner: Option<tokio::net::TcpListener>,
        addr: SocketAddr,
        server: Server,
        kind: Kind,
    ) -> Self {
        let watch = PortWatch::register(server, kind, addr, inner.is_some(), false);
        Listener { inner, watch }
    }

    /// The next connection, on whichever port is current.
    pub async fn accept(&mut self) -> std::io::Result<(tokio::net::TcpStream, SocketAddr)> {
        loop {
            tokio::select! {
                accepted = async {
                    match &self.inner {
                        Some(inner) => inner.accept().await,
                        None => std::future::pending().await,
                    }
                } => return accepted,
                socket = self.watch.next() => match socket.map(tokio::net::TcpListener::from_std) {
                    None => self.inner = None,
                    Some(Ok(moved)) => self.inner = Some(moved),
                    Some(Err(e)) => crate::log_warn!("CONFIG SET port: {e}"),
                },
            }
        }
    }
}

/// An io_uring worker's listener.
#[cfg(target_os = "linux")]
pub struct UringListener {
    inner: Option<tokio_uring::net::TcpListener>,
    watch: PortWatch,
}

#[cfg(target_os = "linux")]
impl UringListener {
    /// As [`Listener::new`], for the plaintext port.
    pub fn new(
        inner: Option<tokio_uring::net::TcpListener>,
        server: Server,
        addr: SocketAddr,
    ) -> Self {
        let watch = PortWatch::register(server, Kind::Plain, addr, inner.is_some(), true);
        UringListener { inner, watch }
    }

    /// The next connection, on whichever port is current.
    pub async fn accept(&mut self) -> std::io::Result<(tokio_uring::net::TcpStream, SocketAddr)> {
        loop {
            tokio::select! {
                accepted = async {
                    match &self.inner {
                        Some(inner) => inner.accept().await,
                        None => std::future::pending().await,
                    }
                } => return accepted,
                socket = self.watch.next() => {
                    self.inner = socket.map(tokio_uring::net::TcpListener::from_std);
                }
            }
        }
    }
}
