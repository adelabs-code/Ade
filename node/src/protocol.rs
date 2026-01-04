use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::io::Cursor;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::network::GossipMessage;
use crate::compression::{Compressor, CompressionAlgorithm};

/// Network protocol version
pub const PROTOCOL_VERSION: u16 = 1;

/// Message header with compression support
#[derive(Debug, Clone)]
pub struct MessageHeader {
    pub version: u16,
    pub message_type: MessageType,
    pub payload_length: u32,
    pub checksum: u32,
    pub compressed: bool,
    pub compression_type: u8,
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

/// Protocol message encoder/decoder with compression
pub struct Protocol;

impl Protocol {
    /// Encode a gossip message with optional compression
    pub fn encode(message: &GossipMessage) -> Result<Vec<u8>> {
        Self::encode_with_compression(message, true)
    }

    /// Encode with compression control
    pub fn encode_with_compression(message: &GossipMessage, enable_compression: bool) -> Result<Vec<u8>> {
        let message_type = Self::get_message_type(message);
        let payload = bincode::serialize(message)?;

        // Determine if compression is worth it
        let (final_payload, compressed, compression_type) = if enable_compression && payload.len() > 1024 {
            // Use Zstd for good compression ratio and speed
            let compressor = Compressor::new(CompressionAlgorithm::Zstd(3));
            match compressor.compress(&payload) {
                Ok(compressed_payload) if compressed_payload.len() < payload.len() => {
                    (compressed_payload, true, 1u8) // 1 = Zstd
                }
                _ => (payload, false, 0u8)
            }
        } else {
            (payload, false, 0u8)
        };

        let mut buffer = Vec::new();
        
        // Header
        buffer.write_u16::<LittleEndian>(PROTOCOL_VERSION)?;
        buffer.write_u8(message_type as u8)?;
        buffer.write_u32::<LittleEndian>(final_payload.len() as u32)?;
        
        // Checksum
        let checksum = Self::compute_checksum(&final_payload);
        buffer.write_u32::<LittleEndian>(checksum)?;
        
        // Compression flags
        buffer.write_u8(if compressed { 1 } else { 0 })?;
        buffer.write_u8(compression_type)?;
        
        // Payload
        buffer.extend_from_slice(&final_payload);
        
        Ok(buffer)
    }

    /// Decode a message with decompression support
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
        let compressed = cursor.read_u8()? == 1;
        let compression_type = cursor.read_u8()?;
        
        // Read payload
        let payload_start = cursor.position() as usize;
        let payload_data = &data[payload_start..payload_start + payload_length];
        
        // Verify checksum
        let actual_checksum = Self::compute_checksum(payload_data);
        if actual_checksum != expected_checksum {
            return Err(anyhow::anyhow!("Checksum mismatch"));
        }
        
        // Decompress if needed
        let final_payload = if compressed {
            let decompressor = match compression_type {
                1 => Compressor::new(CompressionAlgorithm::Zstd(3)),
                2 => Compressor::new(CompressionAlgorithm::Lz4),
                _ => return Err(anyhow::anyhow!("Unknown compression type")),
            };
            
            decompressor.decompress(payload_data)?
        } else {
            payload_data.to_vec()
        };
        
        // Deserialize
        let message: GossipMessage = bincode::deserialize(&final_payload)?;
        
        Ok(message)
    }

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

    fn compute_checksum(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    pub fn validate_message(data: &[u8]) -> Result<MessageHeader> {
        if data.len() < 13 {
            return Err(anyhow::anyhow!("Message too short"));
        }
        
        let mut cursor = Cursor::new(data);
        
        let version = cursor.read_u16::<LittleEndian>()?;
        let message_type_byte = cursor.read_u8()?;
        let payload_length = cursor.read_u32::<LittleEndian>()?;
        let checksum = cursor.read_u32::<LittleEndian>()?;
        let compressed = cursor.read_u8()? == 1;
        let compression_type = cursor.read_u8()?;
        
        let message_type = MessageType::from_u8(message_type_byte)
            .ok_or_else(|| anyhow::anyhow!("Invalid message type"))?;
        
        Ok(MessageHeader {
            version,
            message_type,
            payload_length,
            checksum,
            compressed,
            compression_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_with_compression() {
        // Large message that benefits from compression
        let large_data = vec![1u8; 5000];
        let message = GossipMessage::TransactionBatch {
            transactions: vec![large_data],
        };
        
        let encoded = Protocol::encode(&message).unwrap();
        let decoded = Protocol::decode(&encoded).unwrap();
        
        match decoded {
            GossipMessage::TransactionBatch { transactions } => {
                assert_eq!(transactions.len(), 1);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_small_message_no_compression() {
        let message = GossipMessage::Ping { timestamp: 12345 };
        
        let encoded = Protocol::encode(&message).unwrap();
        
        // Check header
        let header = Protocol::validate_message(&encoded).unwrap();
        assert!(!header.compressed); // Should not compress small messages
    }
}
