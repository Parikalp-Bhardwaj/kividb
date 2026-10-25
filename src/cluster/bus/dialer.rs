// src/cluster/bus/dialer.rs
//
// Outbound bus connections.
//
// Two kinds. Request/response exchanges — MEET, the heartbeat PING, a vote
// request — each open a short-lived connection, send one message, and read
// the reply. Fire-and-forget traffic — forwarded PUBLISHes, FAIL broadcasts —
// goes over one long-lived link per peer, so a busy pub/sub channel does not
// cost a TCP handshake per message.

use crate::cluster::ClusterState;
use crate::cluster::bus::message::{BusMessage, decode, encode};
use crate::cluster::node::{ClusterNode, NodeFlags};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Connect to a peer's bus port, and take the opportunity to learn which of
/// our own addresses reaches it.
async fn connect(
    state: &ClusterState,
    addr: &str,
    timeout: Duration,
) -> Result<Box<dyn crate::node_tls::AsyncStream>, String> {
    let tcp = tokio::time::timeout(timeout, TcpStream::connect(addr))
        .await
        .map_err(|_| format!("connection to {addr} timed out"))?
        .map_err(|e| format!("connect {addr}: {e}"))?;
    let _ = tcp.set_nodelay(true);
    let local = tcp.local_addr().ok();
    let mut sock: Box<dyn crate::node_tls::AsyncStream> = if crate::node_tls::cluster() {
        let host = addr.rsplit_once(':').map_or(addr, |(h, _)| h);
        let tls = tokio::time::timeout(timeout, crate::node_tls::connect_async(host, tcp))
            .await
            .map_err(|_| format!("TLS handshake with {addr} timed out"))?
            .map_err(|e| format!("TLS handshake with {addr}: {e}"))?;
        Box::new(tls)
    } else {
        Box::new(tcp)
    };
    if let Some(secret) = state.bus_secret.get() {
        crate::cluster::bus::auth::answer(&mut sock, secret, timeout)
            .await
            .map_err(|e| format!("{addr} did not let us in: {e}"))?;
    }
    if let Some(local) = local {
        state.learn_self_ip(local.ip());
    }
    Ok(sock)
}

/// Send one message and wait up to `timeout` for one reply.
///
/// `Ok(None)` means the peer chose not to answer — a declined vote, say —
/// which is not an error.
pub async fn exchange(
    state: &ClusterState,
    addr: &str,
    msg: &BusMessage,
    timeout: Duration,
) -> Result<Option<BusMessage>, String> {
    let mut sock = connect(state, addr, timeout).await?;
    sock.write_all(&encode(msg))
        .await
        .map_err(|e| format!("write to {addr}: {e}"))?;
    state.stats_messages_sent.fetch_add(1, Ordering::Relaxed);

    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    let mut tmp = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        let n = match tokio::time::timeout(remaining, sock.read(&mut tmp)).await {
            Ok(Ok(0)) | Err(_) => return Ok(None), // closed, or no reply in time
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(format!("read from {addr}: {e}")),
        };
        buf.extend_from_slice(&tmp[..n]);
        match decode(&buf) {
            Ok(Some((reply, _consumed))) => {
                state
                    .stats_messages_received
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(Some(reply));
            }
            Ok(None) => {
                if crate::cluster::bus::message::exceeds_buffer_limit(&buf) {
                    return Err(format!("oversized reply from {addr}"));
                }
            }
            Err(e) => return Err(format!("decode reply from {addr}: {e}")),
        }
    }
}

/// The address a reply from `peer_ip` should be attributed to when the peer
/// does not state one itself.
fn peer_socket_addr(peer_ip: &str) -> std::net::SocketAddr {
    format!("{peer_ip}:0")
        .parse()
        .unwrap_or_else(|_| std::net::SocketAddr::from(([0, 0, 0, 0], 0)))
}

/// Send a MEET to the given peer, await its PONG, and incorporate the
/// peer into our routing table. This is what CLUSTER MEET fires off.
pub async fn send_meet(
    state: Arc<ClusterState>,
    peer_ip: String,
    peer_bus_port: u16,
) -> Result<(), String> {
    let addr = format!("{peer_ip}:{peer_bus_port}");

    // Build our MEET from current state.
    let meet = {
        let nodes = state.nodes.read();
        let me = nodes
            .get(&state.myself())
            .cloned()
            .unwrap_or_else(|| ClusterNode::new(state.myself(), String::new(), 0, 0));
        BusMessage::Meet {
            sender_id: state.myself(),
            sender_ip: state.advertised_ip(&me.ip),
            sender_client_port: me.client_port,
            sender_cluster_port: me.cluster_port,
            sender_config_epoch: me.config_epoch,
        }
    };

    match exchange(&state, &addr, &meet, Duration::from_secs(5)).await? {
        Some(reply @ BusMessage::Pong { .. }) => {
            // The answer to our own MEET is the one message from an unknown
            // node that is taken at its word.
            crate::cluster::bus::listener::accept_meet_reply(
                &state,
                reply,
                peer_socket_addr(&peer_ip),
            );
            state.save_soon();
            Ok(())
        }
        Some(_) => Err("unexpected reply to MEET".into()),
        None => Err("MEET reply timed out".into()),
    }
}

/// Add a HANDSHAKE placeholder for a peer we've never spoken to.
/// Generates a random temp id; the dialer task will replace it with the
/// real id once the PONG comes back. Returns true if a placeholder was
/// inserted (false if a node at this address is already known).
///
/// The placeholder is stamped with its creation time, and the failure
/// detector removes it if the handshake has not completed within a node
/// timeout — a MEET to an address nobody answers on used to leave a
/// `handshake` row in `CLUSTER NODES` for as long as the process lived.
pub fn insert_handshake_placeholder(state: &ClusterState, ip: &str, client_port: u16) -> bool {
    let mut nodes = state.nodes.write();
    // Already known by address?
    for n in nodes.values() {
        if n.ip == ip && n.client_port == client_port {
            return false;
        }
    }
    let temp_id = crate::cluster::node::generate_node_id();
    let mut node = ClusterNode::new(
        temp_id,
        ip.to_string(),
        client_port,
        client_port.saturating_add(10000),
    );
    node.flags |= NodeFlags::HANDSHAKE;
    node.ping_sent_ms = now_ms();
    nodes.insert(temp_id, node);
    true
}

/// Drop the placeholder for `ip:client_port`, if the handshake it stood for
/// has failed.
pub fn remove_handshake_placeholder(state: &ClusterState, ip: &str, client_port: u16) {
    state.nodes.write().retain(|_, n| {
        !(n.flags.contains(NodeFlags::HANDSHAKE) && n.ip == ip && n.client_port == client_port)
    });
}

/// The PING this node sends: who it is, what it thinks of others, and which
/// slots it owns.
pub(super) fn our_ping(state: &ClusterState) -> BusMessage {
    let nodes = state.nodes.read();
    let me = nodes
        .get(&state.myself())
        .cloned()
        .unwrap_or_else(|| ClusterNode::new(state.myself(), String::new(), 0, 0));
    let gossip = crate::cluster::bus::listener::sample_gossip(
        &nodes,
        state.myself(),
        crate::cluster::bus::listener::GOSSIP_SAMPLE,
    );
    drop(nodes);
    BusMessage::Ping {
        sender_id: state.myself(),
        sender_ip: state.advertised_ip(&me.ip),
        sender_client_port: me.client_port,
        sender_cluster_port: me.cluster_port,
        sender_config_epoch: me.config_epoch,
        sender_flags: me.flags & (NodeFlags::MASTER | NodeFlags::REPLICA),
        sender_master_id: me.master_id,
        sender_hostname: me.hostname.clone(),
        sender_announced_client_port: me.announced_client_port,
        gossip,
        slot_claims: state.my_slot_claims(),
    }
}

/// Send a PING to the given peer and apply the PONG it replies with.
pub async fn send_ping(
    state: Arc<ClusterState>,
    peer_ip: String,
    peer_bus_port: u16,
) -> Result<(), String> {
    let addr = format!("{peer_ip}:{peer_bus_port}");
    let ping = our_ping(&state);

    // Stamp ping_sent_ms on the peer record (best-effort lookup by address)
    // before the attempt, not after a successful write: a peer we cannot even
    // connect to must still age toward PFAIL. Stamped only when the previous
    // ping has been answered, so the timestamp marks the start of the silence
    // rather than moving forward with every retry.
    {
        let mut nodes = state.nodes.write();
        for n in nodes.values_mut() {
            if n.ip == peer_ip
                && n.cluster_port == peer_bus_port
                && n.ping_sent_ms <= n.pong_received_ms
            {
                n.ping_sent_ms = now_ms();
            }
        }
    }

    match exchange(&state, &addr, &ping, Duration::from_secs(3)).await? {
        Some(reply) => {
            crate::cluster::bus::listener::process_message(
                &state,
                reply,
                peer_socket_addr(&peer_ip),
            );
            Ok(())
        }
        None => Err("ping reply timed out".into()),
    }
}

// ── Long-lived links for fire-and-forget traffic ─────────────────────────────

/// Frames queued per peer before new ones are dropped. Pub/sub is
/// at-most-once; a peer that cannot keep up loses messages rather than
/// growing this node's memory.
const LINK_QUEUE: usize = 4096;

/// How long an idle link is kept open.
const LINK_IDLE: Duration = Duration::from_secs(60);

/// Queue `frame` for delivery to the peer at `addr` (`ip:bus_port`), starting
/// a link task for it if there is none. Never blocks; safe to call from any
/// thread. Returns false if the frame was dropped.
pub fn send_frame(state: &Arc<ClusterState>, addr: &str, frame: Vec<u8>) -> bool {
    let Some(runtime) = state.bus_runtime.get() else {
        return false;
    };
    let mut links = state.links.lock();
    if let Some(tx) = links.get(addr) {
        match tx.try_send(frame) {
            Ok(()) => return true,
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => return false,
            // The link task exited (idle timeout); start a fresh one below.
            Err(tokio::sync::mpsc::error::TrySendError::Closed(frame)) => {
                links.remove(addr);
                return start_link(state, runtime, &mut links, addr, frame);
            }
        }
    }
    start_link(state, runtime, &mut links, addr, frame)
}

fn start_link(
    state: &Arc<ClusterState>,
    runtime: &tokio::runtime::Handle,
    links: &mut ahash::AHashMap<String, tokio::sync::mpsc::Sender<Vec<u8>>>,
    addr: &str,
    first: Vec<u8>,
) -> bool {
    let (tx, rx) = tokio::sync::mpsc::channel::<Vec<u8>>(LINK_QUEUE);
    let queued = tx.try_send(first).is_ok();
    links.insert(addr.to_string(), tx);
    runtime.spawn(run_link(Arc::clone(state), addr.to_string(), rx));
    queued
}

/// Deliver queued frames to one peer over a connection that is kept open
/// between frames and re-established when it breaks.
async fn run_link(
    state: Arc<ClusterState>,
    addr: String,
    mut rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
) {
    let mut sock: Option<Box<dyn crate::node_tls::AsyncStream>> = None;
    loop {
        let frame = match tokio::time::timeout(LINK_IDLE, rx.recv()).await {
            Ok(Some(frame)) => frame,
            // Idle, or every sender gone: close. `send_frame` notices the
            // closed channel and starts a new link on the next frame.
            Ok(None) | Err(_) => return,
        };
        // One reconnect per frame: a frame that cannot be delivered on a
        // fresh connection is dropped rather than retried without end.
        for attempt in 0..2 {
            if sock.is_none() {
                match connect(&state, &addr, Duration::from_secs(3)).await {
                    Ok(s) => sock = Some(s),
                    Err(_) => break,
                }
            }
            let Some(s) = sock.as_mut() else { break };
            match tokio::time::timeout(Duration::from_secs(3), s.write_all(&frame)).await {
                Ok(Ok(())) => {
                    state.stats_messages_sent.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                _ => {
                    sock = None;
                    if attempt == 1 {
                        break;
                    }
                }
            }
        }
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
