// SPDX-FileCopyrightText: 2024 Jeffrey OS Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use libp2p::{mdns, PeerId, Swarm, SwarmBuilder};
use libp2p::swarm::SwarmEvent;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("mDNS error: {0}")]
    Mdns(#[from] mdns::Error),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Discovery not started")]
    NotStarted,
}

pub struct PeerDiscovery {
    peers: HashMap<PeerId, String>,
    receiver: Option<mpsc::Receiver<DiscoveryEvent>>,
    _sender: mpsc::Sender<DiscoveryEvent>,
}

#[derive(Debug)]
enum DiscoveryEvent {
    PeerDiscovered(PeerId, String),
    PeerExpired(PeerId),
}

impl PeerDiscovery {
    pub fn new() -> Result<Self, DiscoveryError> {
        let (sender, receiver) = mpsc::channel(100);

        Ok(Self {
            peers: HashMap::new(),
            receiver: Some(receiver),
            _sender: sender,
        })
    }

    pub async fn start(&mut self) -> Result<(), DiscoveryError> {
        // Create mDNS behaviour
        let mdns = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            libp2p::identity::Keypair::generate_ed25519().public().to_peer_id()
        )?;

        // In a real implementation, we would start the mDNS discovery here
        // For now, we'll simulate the discovery process
        tracing::info!("P2P discovery started");
        Ok(())
    }

    pub fn is_peer_known(&self, peer_id: &PeerId) -> bool {
        self.peers.contains_key(peer_id)
    }

    pub fn get_peers(&self) -> Vec<PeerId> {
        self.peers.keys().cloned().collect()
    }

    pub fn get_peer_address(&self, peer_id: &PeerId) -> Option<&String> {
        self.peers.get(peer_id)
    }

    async fn handle_discovery_events(&mut self) -> Result<(), DiscoveryError> {
        if let Some(receiver) = &mut self.receiver {
            while let Some(event) = receiver.recv().await {
                match event {
                    DiscoveryEvent::PeerDiscovered(peer_id, address) => {
                        tracing::info!("Discovered peer: {} at {}", peer_id, address);
                        self.peers.insert(peer_id, address);
                    }
                    DiscoveryEvent::PeerExpired(peer_id) => {
                        tracing::info!("Peer expired: {}", peer_id);
                        self.peers.remove(&peer_id);
                    }
                }
            }
        }
        Ok(())
    }
}
