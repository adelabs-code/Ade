use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::io::Cursor;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::network::GossipMessage;

/// Network protocol version
pub const PROTOCOL_VERSION: u16 = 1;

/// Message header
#[derive(Debug, Clone)]
pub struct MessageHeader {
    pub version: u16,
    pub message_type: MessageType,
    pub payload_length: u32,
    pub checksum: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    PeerInfo = 0,
    BlockProposal = 1,
    TransactionBatch = 2,
    Vote = 3,
    Ping = 4,
    Pong = 5,
    Handshake = 6,
    GetPeers = 7,
    GetBlocks = 8,
}

impl MessageType {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::PeerInfo),
            1 => Some(Self::BlockProposal),
            2 => Some(Self::TransactionBatch),
            3 => Some(Self::Vote),
            4 => Some(Self::Ping),
            5 => Some(Self::Pong),
            6 => Some(Self::Handshake),
            7 => Some(Self::GetPeers),
            8 => Some(Self::GetBlocks),
            _ => None,
        }
    }
}

/// Protocol message encoder/decoder
pub struct Protocol;

impl Protocol {
    /// Encode a gossip message
    pub fn encode(message: &GossipMessage) -> Result<Vec<u8>> {
        let message_type = Self::get_message_type(message);
        let payload = bincode::serialize(message)?;
        
        let mut buffer = Vec::new();
        
        // Header
        buffer.write_u16::<LittleEndian>(PROTOCOL_VERSION)?;
        buffer.write_u8(message_type as u8)?;
        buffer.write_u32::<LittleEndian>(payload.len() as u32)?;
        
        // Checksum
        let checksum = Self::compute_checksum(&payload);
        buffer.write_u32::<LittleEndian>(checksum)?;
        
        // Payload
        buffer.extend_from_slice(&payload);
        
        Ok(buffer)
    }

    /// Decode a message
    pub fn decode(data: &[u8]) -> Result<GossipMessage> {
        let mut cursor = Cursor::new(data);
        
        // Read header
        let version = cursor.read_u16::<LittleEndian>()?;
        if version != PROTOCOL_VERSION {
            return Err(anyhow::anyhow!("Unsupported protocol version: {}", version));
        }
        
        let message_type_byte = cursor.read_u8()?;
        let _message_type = MessageType::from_u8(message_type_byte)
            .ok_or_else(|| anyhow::anyhow!("Invalid message type: {}", message_type_byte))?;
        
        let payload_length = cursor.read_u32::<LittleEndian>()? as usize;
        let expected_checksum = cursor.read_u32::<LittleEndian>()?;
        
        // Read payload
        let payload_start = cursor.position() as usize;
        let payload = &data[payload_start..payload_start + payload_length];
        
        // Verify checksum
        let actual_checksum = Self::compute_checksum(payload);
        if actual_checksum != expected_checksum {
            return Err(anyhow::anyhow!("Checksum mismatch"));
        }
        
        // Deserialize
        let message: GossipMessage = bincode::deserialize(payload)?;
        
        Ok(message)
    }

    /// Get message type from GossipMessage
    fn get_message_type(message: &GossipMessage) -> MessageType {
        match message {
            GossipMessage::PeerInfo { .. } => MessageType::PeerInfo,
            GossipMessage::BlockProposal { .. } => MessageType::BlockProposal,
            GossipMessage::TransactionBatch { .. } => MessageType::TransactionBatch,
            GossipMessage::Vote { .. } => MessageType::Vote,
            GossipMessage::Ping { .. } => MessageType::Ping,
            GossipMessage::Pong { .. } => MessageType::Pong,
        }
    }

    /// Compute CRC32 checksum
    fn compute_checksum(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    /// Validate message structure
    pub fn validate_message(data: &[u8]) -> Result<MessageHeader> {
        if data.len() < 11 {
            return Err(anyhow::anyhow!("Message too short"));
        }
        
        let mut cursor = Cursor::new(data);
        
        let version = cursor.read_u16::<LittleEndian>()?;
        let message_type_byte = cursor.read_u8()?;
        let payload_length = cursor.read_u32::<LittleEndian>()?;
        let checksum = cursor.read_u32::<LittleEndian>()?;
        
        let message_type = MessageType::from_u8(message_type_byte)
            .ok_or_else(|| anyhow::anyhow!("Invalid message type"))?;
        
        Ok(MessageHeader {
            version,
            message_type,
            payload_length,
            checksum,
        })
    }
}

/// Compression support
pub struct Compression;

impl Compression {
    /// Compress data
    pub fn compress(data: &[u8]) -> Result<Vec<u8>> {
        // In production, use actual compression (e.g., zstd, lz4)
        Ok(data.to_vec())
    }

    /// Decompress data
    pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
        // In production, use actual decompression
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_ping() {
        let message = GossipMessage::Ping {
            timestamp: 12345,
        };
        
        let encoded = Protocol::encode(&message).unwrap();
        let decoded = Protocol::decode(&encoded).unwrap();
        
        match decoded {
            GossipMessage::Ping { timestamp } => {
                assert_eq!(timestamp, 12345);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_encode_decode_peer_info() {
        let message = GossipMessage::PeerInfo {
            pubkey: vec![1u8; 32],
            address: "127.0.0.1:9900".to_string(),
            stake: 1000000,
            version: "1.0.0".to_string(),
        };
        
        let encoded = Protocol::encode(&message).unwrap();
        let decoded = Protocol::decode(&encoded).unwrap();
        
        match decoded {
            GossipMessage::PeerInfo { pubkey, address, stake, version } => {
                assert_eq!(pubkey, vec![1u8; 32]);
                assert_eq!(address, "127.0.0.1:9900");
                assert_eq!(stake, 1000000);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_checksum_validation() {
        let message = GossipMessage::Ping { timestamp: 12345 };
        let mut encoded = Protocol::encode(&message).unwrap();
        
        // Corrupt payload
        encoded[15] ^= 1;
        
        let result = Protocol::decode(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_message() {
        let message = GossipMessage::Ping { timestamp: 12345 };
        let encoded = Protocol::encode(&message).unwrap();
        
        let header = Protocol::validate_message(&encoded).unwrap();
        assert_eq!(header.version, PROTOCOL_VERSION);
        assert_eq!(header.message_type, MessageType::Ping);
    }

    #[test]
    fn test_invalid_version() {
        let message = GossipMessage::Ping { timestamp: 12345 };
        let mut encoded = Protocol::encode(&message).unwrap();
        
        // Change version
        encoded[0] = 99;
        
        let result = Protocol::decode(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression() {
        let data = vec![1u8; 1000];
        let compressed = Compression::compress(&data).unwrap();
        let decompressed = Compression::decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed);
    }
}

