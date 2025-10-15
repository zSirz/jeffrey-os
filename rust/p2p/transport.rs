// SPDX-FileCopyrightText: 2024 Jeffrey OS Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use libp2p::{noise, PeerId, Transport};
use libp2p::tcp::tokio as tcp_tokio;
use libp2p::websocket;
use crate::messages::Message;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Transport not started")]
    NotStarted,

    #[error("Send error: {0}")]
    Send(String),
}

pub struct P2PTransport {
    local_peer_id: PeerId,
    message_sender: Option<mpsc::Sender<(PeerId, Message)>>,
    message_receiver: Option<mpsc::Receiver<(PeerId, Message)>>,
}

impl P2PTransport {
    pub fn new() -> Result<Self, TransportError> {
        let local_keypair = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = local_keypair.public().to_peer_id();

        let (message_sender, message_receiver) = mpsc::channel(100);

        Ok(Self {
            local_peer_id,
            message_sender: Some(message_sender),
            message_receiver: Some(message_receiver),
        })
    }

    pub async fn start(&mut self) -> Result<(), TransportError> {
        // Create transport with noise security and TCP/WebSocket
        let tcp_transport = tcp_tokio::Transport::default();
        let ws_transport = websocket::WsConfig::new(tcp_tokio::Transport::default());

        let transport = tcp_transport
            .or_transport(ws_transport)
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::Config::new(&libp2p::identity::Keypair::generate_ed25519()).unwrap())
            .multiplex(libp2p::yamux::Config::default())
            .boxed();

        tracing::info!("P2P transport started for peer: {}", self.local_peer_id);
        Ok(())
    }

    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    pub async fn send_message(&mut self, peer_id: PeerId, message: Message) -> Result<(), TransportError> {
        // In a real implementation, this would send the message through the libp2p network
        tracing::info!("Sending message to {}: {:?}", peer_id, message);

        // Simulate sending by logging
        let serialized = serde_json::to_string(&message)?;
        tracing::debug!("Serialized message: {}", serialized);

        Ok(())
    }

    pub async fn receive_message(&mut self) -> Result<Option<(PeerId, Message)>, TransportError> {
        if let Some(receiver) = &mut self.message_receiver {
            Ok(receiver.recv().await)
        } else {
            Err(TransportError::NotStarted)
        }
    }

    pub fn is_connected_to(&self, _peer_id: &PeerId) -> bool {
        // In a real implementation, this would check actual connections
        false
    }
}
