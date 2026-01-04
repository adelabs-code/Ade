use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use ed25519_dalek::{Keypair, Signature, Signer, PublicKey, Verifier};
use anyhow::{Result, Context};
use std::io::{Write, Cursor};
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub signatures: Vec<Vec<u8>>,
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub header: MessageHeader,
    pub account_keys: Vec<Vec<u8>>,
    pub recent_blockhash: Vec<u8>,
    pub instructions: Vec<crate::instruction::Instruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub num_required_signatures: u8,
    pub num_readonly_signed_accounts: u8,
    pub num_readonly_unsigned_accounts: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransactionError {
    InvalidSignature(usize),
    InsufficientSignatures,
    InvalidAccountIndex,
    InvalidMessage,
    SerializationError(String),
    BlockhashExpired,
    DuplicateSignature,
}

impl std::fmt::Display for TransactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidSignature(idx) => write!(f, "Invalid signature at index {}", idx),
            Self::InsufficientSignatures => write!(f, "Insufficient signatures"),
            Self::InvalidAccountIndex => write!(f, "Invalid account index"),
            Self::InvalidMessage => write!(f, "Invalid message"),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::BlockhashExpired => write!(f, "Blockhash has expired"),
            Self::DuplicateSignature => write!(f, "Duplicate signature detected"),
        }
    }
}

impl std::error::Error for TransactionError {}

impl Transaction {
    /// Create a new transaction with signatures
    pub fn new(
        signers: &[&Keypair],
        instructions: Vec<crate::instruction::Instruction>,
        recent_blockhash: Vec<u8>,
    ) -> Result<Self> {
        if signers.is_empty() {
            return Err(anyhow::anyhow!(TransactionError::InsufficientSignatures));
        }

        let account_keys: Vec<Vec<u8>> = signers
            .iter()
            .map(|k| k.public.to_bytes().to_vec())
            .collect();

        let header = MessageHeader {
            num_required_signatures: signers.len() as u8,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };

        let message = Message {
            header,
            account_keys,
            recent_blockhash,
            instructions,
        };

        let message_bytes = message.serialize_for_signing()?;
        let mut signatures = Vec::new();

        for keypair in signers {
            let signature = keypair.sign(&message_bytes);
            signatures.push(signature.to_bytes().to_vec());
        }

        Ok(Self {
            signatures,
            message,
        })
    }

    /// Compute SHA-256 hash of the transaction
    pub fn hash(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        if let Ok(serialized) = self.serialize() {
            hasher.update(&serialized);
        }
        hasher.finalize().to_vec()
    }

    /// Get transaction signature (first signature)
    pub fn signature(&self) -> Option<&Vec<u8>> {
        self.signatures.first()
    }

    /// Verify all signatures in the transaction
    pub fn verify(&self) -> Result<bool, TransactionError> {
        if self.signatures.len() < self.message.header.num_required_signatures as usize {
            return Err(TransactionError::InsufficientSignatures);
        }

        let message_bytes = self.message.serialize_for_signing()
            .map_err(|e| TransactionError::SerializationError(e.to_string()))?;
        
        // Check for duplicate signatures
        let mut seen_signatures = std::collections::HashSet::new();
        for sig in &self.signatures {
            if !seen_signatures.insert(sig) {
                return Err(TransactionError::DuplicateSignature);
            }
        }

        // Verify each signature
        for (i, sig_bytes) in self.signatures.iter().enumerate() {
            if i >= self.message.account_keys.len() {
                return Err(TransactionError::InvalidAccountIndex);
            }

            let pubkey_bytes = &self.message.account_keys[i];
            let pubkey = PublicKey::from_bytes(pubkey_bytes)
                .map_err(|_| TransactionError::InvalidSignature(i))?;
            
            let signature = Signature::from_bytes(sig_bytes.as_slice())
                .map_err(|_| TransactionError::InvalidSignature(i))?;

            pubkey.verify_strict(&message_bytes, &signature)
                .map_err(|_| TransactionError::InvalidSignature(i))?;
        }

        Ok(true)
    }

    /// Check if transaction is properly signed
    pub fn is_signed(&self) -> bool {
        !self.signatures.is_empty() && 
        self.signatures.len() >= self.message.header.num_required_signatures as usize
    }

    /// Get all signers' public keys
    pub fn get_signers(&self) -> Vec<Vec<u8>> {
        self.message.account_keys
            .iter()
            .take(self.message.header.num_required_signatures as usize)
            .cloned()
            .collect()
    }

    /// Serialize transaction to bytes (compact format)
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Write number of signatures
        buffer.push(self.signatures.len() as u8);
        
        // Write each signature (64 bytes each)
        for sig in &self.signatures {
            if sig.len() != 64 {
                return Err(anyhow::anyhow!("Invalid signature length"));
            }
            buffer.extend_from_slice(sig);
        }
        
        // Write serialized message
        let message_bytes = self.message.serialize()?;
        buffer.extend_from_slice(&message_bytes);
        
        Ok(buffer)
    }

    /// Deserialize transaction from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        
        // Read number of signatures
        let num_signatures = cursor.read_u8()? as usize;
        
        // Read signatures
        let mut signatures = Vec::with_capacity(num_signatures);
        for _ in 0..num_signatures {
            let mut sig = vec![0u8; 64];
            std::io::Read::read_exact(&mut cursor, &mut sig)?;
            signatures.push(sig);
        }
        
        // Read message
        let message_start = cursor.position() as usize;
        let message = Message::deserialize(&data[message_start..])?;
        
        Ok(Self {
            signatures,
            message,
        })
    }

    /// Get transaction size in bytes
    pub fn size(&self) -> usize {
        self.serialize().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if transaction size is within limits
    pub fn is_size_valid(&self, max_size: usize) -> bool {
        self.size() <= max_size
    }
}

impl Message {
    /// Serialize message for signing (canonical format)
    pub fn serialize_for_signing(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Header (3 bytes)
        buffer.push(self.header.num_required_signatures);
        buffer.push(self.header.num_readonly_signed_accounts);
        buffer.push(self.header.num_readonly_unsigned_accounts);
        
        // Account keys
        Self::write_compact_u16(&mut buffer, self.account_keys.len())?;
        for key in &self.account_keys {
            if key.len() != 32 {
                return Err(anyhow::anyhow!("Invalid pubkey length"));
            }
            buffer.extend_from_slice(key);
        }
        
        // Recent blockhash (32 bytes)
        if self.recent_blockhash.len() != 32 {
            return Err(anyhow::anyhow!("Invalid blockhash length"));
        }
        buffer.extend_from_slice(&self.recent_blockhash);
        
        // Instructions
        Self::write_compact_u16(&mut buffer, self.instructions.len())?;
        for instruction in &self.instructions {
            buffer.extend_from_slice(&instruction.serialize()?);
        }
        
        Ok(buffer)
    }

    /// Serialize message (same as signing format for now)
    pub fn serialize(&self) -> Result<Vec<u8>> {
        self.serialize_for_signing()
    }

    /// Deserialize message from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        
        // Read header
        let num_required_signatures = cursor.read_u8()?;
        let num_readonly_signed_accounts = cursor.read_u8()?;
        let num_readonly_unsigned_accounts = cursor.read_u8()?;
        
        let header = MessageHeader {
            num_required_signatures,
            num_readonly_signed_accounts,
            num_readonly_unsigned_accounts,
        };
        
        // Read account keys
        let num_keys = Self::read_compact_u16(&mut cursor)? as usize;
        let mut account_keys = Vec::with_capacity(num_keys);
        for _ in 0..num_keys {
            let mut key = vec![0u8; 32];
            std::io::Read::read_exact(&mut cursor, &mut key)?;
            account_keys.push(key);
        }
        
        // Read blockhash
        let mut recent_blockhash = vec![0u8; 32];
        std::io::Read::read_exact(&mut cursor, &mut recent_blockhash)?;
        
        // Read instructions
        let num_instructions = Self::read_compact_u16(&mut cursor)? as usize;
        let mut instructions = Vec::with_capacity(num_instructions);
        
        let remaining = &data[cursor.position() as usize..];
        let mut offset = 0;
        
        for _ in 0..num_instructions {
            // Use deserialize_with_accounts to properly resolve account indices
            let instruction = crate::instruction::Instruction::deserialize_with_accounts(
                &remaining[offset..],
                &account_keys
            )?;
            offset += instruction.serialized_size();
            instructions.push(instruction);
        }
        
        Ok(Self {
            header,
            account_keys,
            recent_blockhash,
            instructions,
        })
    }

    /// Write compact u16 (short u16 encoding)
    fn write_compact_u16(buffer: &mut Vec<u8>, value: usize) -> Result<()> {
        let mut val = value;
        loop {
            let mut byte = (val & 0x7f) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            buffer.push(byte);
            if val == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Read compact u16
    fn read_compact_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        let mut value = 0u16;
        let mut shift = 0;
        
        loop {
            let byte = cursor.read_u8()?;
            value |= ((byte & 0x7f) as u16) << shift;
            
            if byte & 0x80 == 0 {
                break;
            }
            
            shift += 7;
            if shift >= 16 {
                return Err(anyhow::anyhow!("Compact u16 overflow"));
            }
        }
        
        Ok(value)
    }
}

pub struct TransactionBuilder {
    instructions: Vec<crate::instruction::Instruction>,
    signers: Vec<Keypair>,
    recent_blockhash: Option<Vec<u8>>,
    fee_payer: Option<usize>, // Index into signers
}

impl TransactionBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            signers: Vec::new(),
            recent_blockhash: None,
            fee_payer: None,
        }
    }

    pub fn add_instruction(mut self, instruction: crate::instruction::Instruction) -> Self {
        self.instructions.push(instruction);
        self
    }

    pub fn add_signer(mut self, signer: Keypair) -> Self {
        if self.fee_payer.is_none() {
            self.fee_payer = Some(self.signers.len());
        }
        self.signers.push(signer);
        self
    }

    pub fn set_fee_payer(mut self, index: usize) -> Self {
        self.fee_payer = Some(index);
        self
    }

    pub fn set_recent_blockhash(mut self, blockhash: Vec<u8>) -> Self {
        self.recent_blockhash = Some(blockhash);
        self
    }

    pub fn build(mut self) -> Result<Transaction> {
        let blockhash = self.recent_blockhash
            .ok_or_else(|| anyhow::anyhow!("Recent blockhash not set"))?;

        if self.signers.is_empty() {
            return Err(anyhow::anyhow!("No signers provided"));
        }

        // Reorder signers to put fee payer first
        if let Some(fee_payer_idx) = self.fee_payer {
            if fee_payer_idx != 0 && fee_payer_idx < self.signers.len() {
                self.signers.swap(0, fee_payer_idx);
            }
        }

        let signer_refs: Vec<&Keypair> = self.signers.iter().collect();
        Transaction::new(&signer_refs, self.instructions, blockhash)
    }
}

impl Default for TransactionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    #[test]
    fn test_transaction_creation() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let blockhash = vec![1u8; 32];
        
        let tx = Transaction::new(
            &[&keypair],
            vec![],
            blockhash,
        );
        
        assert!(tx.is_ok());
    }

    #[test]
    fn test_transaction_verification() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let blockhash = vec![1u8; 32];
        
        let tx = Transaction::new(&[&keypair], vec![], blockhash).unwrap();
        assert!(tx.verify().unwrap());
    }

    #[test]
    fn test_transaction_serialization() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let blockhash = vec![1u8; 32];
        
        let tx = Transaction::new(&[&keypair], vec![], blockhash).unwrap();
        let serialized = tx.serialize().unwrap();
        let deserialized = Transaction::deserialize(&serialized).unwrap();
        
        assert_eq!(tx.signatures, deserialized.signatures);
    }

    #[test]
    fn test_invalid_signature() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let blockhash = vec![1u8; 32];
        
        let mut tx = Transaction::new(&[&keypair], vec![], blockhash).unwrap();
        
        // Corrupt signature
        tx.signatures[0][0] ^= 1;
        
        assert!(tx.verify().is_err());
    }

    #[test]
    fn test_duplicate_signatures() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let blockhash = vec![1u8; 32];
        
        let mut tx = Transaction::new(&[&keypair], vec![], blockhash).unwrap();
        
        // Add duplicate signature
        let dup_sig = tx.signatures[0].clone();
        tx.signatures.push(dup_sig);
        
        assert!(matches!(tx.verify(), Err(TransactionError::DuplicateSignature)));
    }
}
