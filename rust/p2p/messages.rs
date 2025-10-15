// SPDX-FileCopyrightText: 2024 Jeffrey OS Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid message format")]
    InvalidFormat,

    #[error("Message too large: {0} bytes")]
    TooLarge(usize),

    #[error("Unsupported message type")]
    UnsupportedType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Text,
    File,
    StatusUpdate,
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub message_type: MessageType,
    pub content: Vec<u8>,
    pub metadata: MessageMetadata,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub sender_name: String,
    pub recipient_name: Option<String>,
    pub encryption_used: bool,
    pub file_name: Option<String>,
    pub file_size: Option<usize>,
    pub checksum: Option<String>,
}

impl Message {
    pub fn new_text(sender_name: String, content: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: MessageType::Text,
            content: content.into_bytes(),
            metadata: MessageMetadata {
                sender_name,
                recipient_name: None,
                encryption_used: false,
                file_name: None,
                file_size: None,
                checksum: None,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    pub fn new_file(sender_name: String, file_name: String, content: Vec<u8>) -> Self {
        let file_size = content.len();
        let checksum = format!("{:x}", md5::compute(&content));

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: MessageType::File,
            content,
            metadata: MessageMetadata {
                sender_name,
                recipient_name: None,
                encryption_used: false,
                file_name: Some(file_name),
                file_size: Some(file_size),
                checksum: Some(checksum),
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    pub fn ping(sender_name: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: MessageType::Ping,
            content: Vec::new(),
            metadata: MessageMetadata {
                sender_name,
                recipient_name: None,
                encryption_used: false,
                file_name: None,
                file_size: None,
                checksum: None,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    pub fn pong(sender_name: String, ping_id: String) -> Self {
        Self {
            id: ping_id, // Use the same ID as the ping
            message_type: MessageType::Pong,
            content: Vec::new(),
            metadata: MessageMetadata {
                sender_name,
                recipient_name: None,
                encryption_used: false,
                file_name: None,
                file_size: None,
                checksum: None,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        }
    }

    pub fn get_text_content(&self) -> Result<String, MessageError> {
        if !matches!(self.message_type, MessageType::Text) {
            return Err(MessageError::InvalidFormat);
        }

        String::from_utf8(self.content.clone())
            .map_err(|_| MessageError::InvalidFormat)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, MessageError> {
        Ok(serde_json::to_vec(self)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, MessageError> {
        Ok(serde_json::from_slice(data)?)
    }

    pub fn size(&self) -> usize {
        self.content.len() + std::mem::size_of::<Self>()
    }

    pub fn validate(&self) -> Result<(), MessageError> {
        const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024; // 10MB

        if self.size() > MAX_MESSAGE_SIZE {
            return Err(MessageError::TooLarge(self.size()));
        }

        // Validate file messages
        if matches!(self.message_type, MessageType::File) {
            if self.metadata.file_name.is_none() {
                return Err(MessageError::InvalidFormat);
            }
            if self.metadata.file_size.is_none() {
                return Err(MessageError::InvalidFormat);
            }
            if self.metadata.checksum.is_none() {
                return Err(MessageError::InvalidFormat);
            }

            // Verify file size matches content
            if let Some(expected_size) = self.metadata.file_size {
                if expected_size != self.content.len() {
                    return Err(MessageError::InvalidFormat);
                }
            }
        }

        Ok(())
    }
}
