//! TLS acceptor construction.
//!
//! Gated behind the `tls` feature — when built without it this module is
//! empty and the compiler eliminates all TLS-related branches.

#[cfg(feature = "tls")]
pub mod acceptor {
    use rustls_pemfile::{certs, private_key};
    use std::fs::File;
    use std::io::BufReader;
    use std::sync::Arc;
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::server::WebPkiClientVerifier;
    use tokio_rustls::rustls::server::danger::ClientCertVerifier;
    use tokio_rustls::rustls::{RootCertStore, ServerConfig};

    /// The files a server's TLS is set up from. `CONFIG SET` changes them
    /// while it runs (#156).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Files {
        pub cert: String,
        pub key: String,
        pub ca: String,
        pub auth_clients: String,
        pub client_cert: String,
        pub client_key: String,
    }

    /// The settings that make up [`Files`].
    pub const SETTINGS: [&str; 6] = [
        "tls-cert-file",
        "tls-key-file",
        "tls-ca-cert-file",
        "tls-auth-clients",
        "tls-client-cert-file",
        "tls-client-key-file",
    ];

    impl Files {
        pub fn of(config: &crate::config::Config) -> Self {
            Files {
                cert: config.tls_cert_file.clone(),
                key: config.tls_key_file.clone(),
                ca: config.tls_ca_cert_file.clone(),
                auth_clients: config.tls_auth_clients.clone(),
                client_cert: config.tls_client_cert_file.clone(),
                client_key: config.tls_client_key_file.clone(),
            }
        }

        fn slot(&mut self, name: &str) -> Option<&mut String> {
            Some(match name {
                "tls-cert-file" => &mut self.cert,
                "tls-key-file" => &mut self.key,
                "tls-ca-cert-file" => &mut self.ca,
                "tls-auth-clients" => &mut self.auth_clients,
                "tls-client-cert-file" => &mut self.client_cert,
                "tls-client-key-file" => &mut self.client_key,
                _ => return None,
            })
        }

        /// One of [`SETTINGS`] by its name.
        pub fn get(&self, name: &str) -> Option<String> {
            self.clone().slot(name).map(|value| value.clone())
        }

        /// Set one of [`SETTINGS`] by its name.
        pub fn set(&mut self, name: &str, value: &str) {
            if let Some(slot) = self.slot(name) {
                *slot = value.to_string();
            }
        }
    }

    /// The TLS acceptor of each server of this process, with the files it
    /// is set up from. A handshake takes the one there now, so that a new
    /// one serves every connection from then on; a test process runs
    /// several servers, hence one each. A server without TLS has the files
    /// and no acceptor, until `CONFIG SET tls-port` opens its TLS port.
    #[allow(clippy::type_complexity)]
    static LISTENERS: std::sync::LazyLock<
        parking_lot::Mutex<
            std::collections::HashMap<crate::listen::Server, (Files, Option<TlsAcceptor>)>,
        >,
    > = std::sync::LazyLock::new(Default::default);

    /// Build the acceptor of `server` from `files`, and serve every
    /// handshake from now on with it. Nothing changes if it cannot be built.
    pub fn install(server: crate::listen::Server, files: &Files) -> Result<TlsAcceptor, String> {
        let acceptor = build(files)?;
        LISTENERS
            .lock()
            .insert(server, (files.clone(), Some(acceptor.clone())));
        Ok(acceptor)
    }

    /// Keep `files` as what `server`'s TLS is set up from, without loading
    /// them: while its TLS port is closed a certificate may be set before
    /// its key. Opening the port loads them.
    pub fn keep(server: crate::listen::Server, files: &Files) {
        LISTENERS
            .lock()
            .entry(server)
            .and_modify(|(kept, _)| *kept = files.clone())
            .or_insert_with(|| (files.clone(), None));
    }

    /// The acceptor `server` serves TLS handshakes with now.
    pub fn current(server: crate::listen::Server) -> Option<TlsAcceptor> {
        LISTENERS
            .lock()
            .get(&server)
            .and_then(|(_, acceptor)| acceptor.clone())
    }

    /// The files `server`'s TLS is set up from now.
    pub fn files(server: crate::listen::Server) -> Option<Files> {
        LISTENERS
            .lock()
            .get(&server)
            .map(|(files, _)| files.clone())
    }

    /// The listener of the TLS port of the server `config` starts, with its
    /// acceptor: listening when TLS is set up, and on no port otherwise,
    /// until `CONFIG SET tls-port` opens one. Call it in a Tokio runtime.
    pub fn listener(config: &crate::config::Config) -> Result<crate::listen::Listener, String> {
        use crate::listen::{Kind, Listener};
        let server = crate::listen::server_of(config);
        let files = Files::of(config);
        let addr: std::net::SocketAddr = format!("{}:{}", config.bind, config.tls_port)
            .parse()
            .map_err(|_| format!("invalid TLS bind address '{}'", config.bind))?;
        if !config.tls_enabled() {
            keep(server, &files);
            return Ok(Listener::new(None, addr, server, Kind::Tls));
        }
        install(server, &files)?;
        let socket = std::net::TcpListener::bind(addr)
            .and_then(|socket| {
                socket.set_nonblocking(true)?;
                tokio::net::TcpListener::from_std(socket)
            })
            .map_err(|e| format!("can't listen on {addr}: {e}"))?;
        crate::log_info!("TLS listener   : {addr}");
        Ok(Listener::new(Some(socket), addr, server, Kind::Tls))
    }

    /// Build a `TlsAcceptor` from the PEM cert and key files of `files`.
    ///
    /// When `files.ca` is set, client certificates are verified against it
    /// as `files.auth_clients` (`tls-auth-clients`) says: `yes` requires
    /// one, `optional` accepts a client without one but refuses an invalid
    /// one, `no` does not ask. With no CA file clients are not asked for a
    /// certificate.
    ///
    /// Returns an error string for the log, before the process exits.
    pub fn build(files: &Files) -> Result<TlsAcceptor, String> {
        let (cert_path, key_path, ca_path, auth_clients) =
            (&files.cert, &files.key, &files.ca, &files.auth_clients);
        let cert_file = File::open(cert_path)
            .map_err(|e| format!("[TLS] Cannot open cert file '{cert_path}': {e}"))?;
        let key_file = File::open(key_path)
            .map_err(|e| format!("[TLS] Cannot open key file '{key_path}': {e}"))?;

        let certs = certs(&mut BufReader::new(cert_file))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("[TLS] Failed to parse cert: {e}"))?;

        let key = private_key(&mut BufReader::new(key_file))
            .map_err(|e| format!("[TLS] Failed to parse key: {e}"))?
            .ok_or_else(|| format!("[TLS] No private key found in '{key_path}'"))?;

        let builder = ServerConfig::builder();
        let builder = match client_verifier(ca_path, auth_clients)? {
            Some(verifier) => builder.with_client_cert_verifier(verifier),
            None => builder.with_no_client_auth(),
        };
        let config = builder
            .with_single_cert(certs, key)
            .map_err(|e| format!("[TLS] TLS config error: {e}"))?;

        Ok(TlsAcceptor::from(Arc::new(config)))
    }

    /// The verifier for client certificates, or `None` when clients are not
    /// asked for one.
    fn client_verifier(
        ca_path: &str,
        auth_clients: &str,
    ) -> Result<Option<Arc<dyn ClientCertVerifier>>, String> {
        if ca_path.is_empty() || auth_clients == "no" {
            return Ok(None);
        }

        let ca_file = File::open(ca_path)
            .map_err(|e| format!("[TLS] Cannot open CA cert file '{ca_path}': {e}"))?;
        let mut roots = RootCertStore::empty();
        for cert in certs(&mut BufReader::new(ca_file)) {
            let cert = cert.map_err(|e| format!("[TLS] Failed to parse CA cert: {e}"))?;
            roots
                .add(cert)
                .map_err(|e| format!("[TLS] Invalid CA cert in '{ca_path}': {e}"))?;
        }
        if roots.is_empty() {
            return Err(format!("[TLS] No CA certificate found in '{ca_path}'"));
        }

        let builder = WebPkiClientVerifier::builder(Arc::new(roots));
        let builder = if auth_clients == "optional" {
            builder.allow_unauthenticated()
        } else {
            builder
        };
        let verifier = builder
            .build()
            .map_err(|e| format!("[TLS] Client certificate verifier error: {e}"))?;
        Ok(Some(verifier))
    }
}
