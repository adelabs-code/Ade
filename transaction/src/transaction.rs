use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use ed25519_dalek::{Keypair, Signature, Signer, PublicKey};
use anyhow::Result;

use crate::instruction::Instruction;

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
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub num_required_signatures: u8,
    pub num_readonly_signed_accounts: u8,
    pub num_readonly_unsigned_accounts: u8,
}

impl Transaction {
    pub fn new(
        signers: &[&Keypair],
        instructions: Vec<Instruction>,
        recent_blockhash: Vec<u8>,
    ) -> Result<Self> {
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

        let message_bytes = bincode::serialize(&message)?;
        let signatures: Vec<Vec<u8>> = signers
            .iter()
            .map(|keypair| {
                let signature = keypair.sign(&message_bytes);
                signature.to_bytes().to_vec()
            })
            .collect();

        Ok(Self {
            signatures,
            message,
        })
    }

    pub fn hash(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        let serialized = bincode::serialize(self).unwrap();
        hasher.update(&serialized);
        hasher.finalize().to_vec()
    }

    pub fn verify(&self) -> Result<bool> {
        let message_bytes = bincode::serialize(&self.message)?;
        
        for (i, sig_bytes) in self.signatures.iter().enumerate() {
            if i >= self.message.account_keys.len() {
                return Ok(false);
            }

            let pubkey_bytes = &self.message.account_keys[i];
            let pubkey = PublicKey::from_bytes(pubkey_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;
            
            let signature = Signature::from_bytes(sig_bytes.as_slice())
                .map_err(|e| anyhow::anyhow!("Invalid signature: {}", e))?;

            pubkey.verify_strict(&message_bytes, &signature)
                .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?;
        }

        Ok(true)
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

pub struct TransactionBuilder {
    instructions: Vec<Instruction>,
    signers: Vec<Keypair>,
    recent_blockhash: Option<Vec<u8>>,
}

impl TransactionBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            signers: Vec::new(),
            recent_blockhash: None,
        }
    }

    pub fn add_instruction(mut self, instruction: Instruction) -> Self {
        self.instructions.push(instruction);
        self
    }

    pub fn add_signer(mut self, signer: Keypair) -> Self {
        self.signers.push(signer);
        self
    }

    pub fn set_recent_blockhash(mut self, blockhash: Vec<u8>) -> Self {
        self.recent_blockhash = Some(blockhash);
        self
    }

    pub fn build(self) -> Result<Transaction> {
        let blockhash = self.recent_blockhash
            .ok_or_else(|| anyhow::anyhow!("Recent blockhash not set"))?;

        let signer_refs: Vec<&Keypair> = self.signers.iter().collect();
        Transaction::new(&signer_refs, self.instructions, blockhash)
    }
}

impl Default for TransactionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

