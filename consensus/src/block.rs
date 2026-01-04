use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use ade_transaction::Transaction;
use anyhow::{Result, Context};

pub type BlockHash = Vec<u8>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    pub slot: u64,
    pub parent_hash: BlockHash,
    pub transactions_root: BlockHash,
    pub accounts_root: BlockHash,
    pub timestamp: u64,
    pub validator: Vec<u8>,
    pub signature: Vec<u8>,
}

impl Block {
    pub fn new(
        slot: u64,
        parent_hash: BlockHash,
        transactions: Vec<Transaction>,
        validator: Vec<u8>,
    ) -> Self {
        let transactions_root = Self::compute_merkle_root(&transactions);
        let timestamp = current_timestamp();
        let accounts_root = vec![0u8; 32];

        let header = BlockHeader {
            slot,
            parent_hash,
            transactions_root,
            accounts_root,
            timestamp,
            validator,
            signature: Vec::new(),
        };

        Self {
            header,
            transactions,
        }
    }

    pub fn hash(&self) -> BlockHash {
        hash_header(&self.header)
    }

    fn compute_merkle_root(transactions: &[Transaction]) -> BlockHash {
        if transactions.is_empty() {
            return vec![0u8; 32];
        }

        let mut hashes: Vec<Vec<u8>> = transactions
            .iter()
            .map(|tx| tx.hash())
            .collect();

        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in hashes.chunks(2) {
                let hash = hash_pair(&chunk[0], chunk.get(1).unwrap_or(&chunk[0]));
                next_level.push(hash);
            }
            
            hashes = next_level;
        }

        hashes[0].clone()
    }

    pub fn validate_structure(&self) -> Result<()> {
        if self.header.slot == 0 && !self.header.parent_hash.is_empty() {
            return Err(anyhow::anyhow!("Genesis block should have empty parent hash"));
        }

        if !self.header.parent_hash.is_empty() && self.header.parent_hash.len() != 32 {
            return Err(anyhow::anyhow!("Invalid parent hash length"));
        }

        if self.header.validator.len() != 32 {
            return Err(anyhow::anyhow!("Invalid validator pubkey length"));
        }

        let computed_root = Self::compute_merkle_root(&self.transactions);
        if computed_root != self.header.transactions_root {
            return Err(anyhow::anyhow!("Transaction merkle root mismatch"));
        }

        Ok(())
    }

    pub fn validate_against_parent(&self, parent: &Block) -> Result<()> {
        if self.header.slot != parent.header.slot + 1 {
            return Err(anyhow::anyhow!(
                "Invalid slot sequence: {} != {} + 1",
                self.header.slot,
                parent.header.slot
            ));
        }

        let parent_hash = parent.hash();
        if self.header.parent_hash != parent_hash {
            return Err(anyhow::anyhow!("Parent hash mismatch"));
        }

        if self.header.timestamp <= parent.header.timestamp {
            return Err(anyhow::anyhow!("Block timestamp must be greater than parent"));
        }

        Ok(())
    }

    pub fn validate_transactions(&self) -> Result<()> {
        for (idx, tx) in self.transactions.iter().enumerate() {
            tx.verify().map_err(|e| {
                anyhow::anyhow!("Transaction {} verification failed: {}", idx, e)
            })?;
        }
        Ok(())
    }

    pub fn sign(&mut self, keypair: &ed25519_dalek::Keypair) -> Result<()> {
        use ed25519_dalek::Signer;
        
        let block_hash = self.hash();
        let signature = keypair.sign(&block_hash);
        self.header.signature = signature.to_bytes().to_vec();
        
        Ok(())
    }

    pub fn verify_signature(&self) -> Result<()> {
        use ed25519_dalek::{PublicKey, Signature, Verifier};
        
        if self.header.signature.is_empty() {
            return Err(anyhow::anyhow!("Block is not signed"));
        }

        let pubkey = PublicKey::from_bytes(&self.header.validator)
            .context("Invalid validator pubkey")?;
        
        let signature = Signature::from_bytes(&self.header.signature)
            .context("Invalid signature")?;
        
        let mut header_for_verification = self.header.clone();
        header_for_verification.signature = Vec::new();
        
        let hash = hash_header(&header_for_verification);
        
        pubkey.verify(&hash, &signature)
            .context("Signature verification failed")?;
        
        Ok(())
    }

    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

pub struct BlockBuilder {
    slot: u64,
    parent_hash: BlockHash,
    transactions: Vec<Transaction>,
    validator: Vec<u8>,
    accounts_root: Option<BlockHash>,
}

impl BlockBuilder {
    pub fn new(slot: u64, parent_hash: BlockHash, validator: Vec<u8>) -> Self {
        Self {
            slot,
            parent_hash,
            transactions: Vec::new(),
            validator,
            accounts_root: None,
        }
    }

    pub fn add_transaction(mut self, tx: Transaction) -> Self {
        self.transactions.push(tx);
        self
    }

    pub fn add_transactions(mut self, txs: Vec<Transaction>) -> Self {
        self.transactions.extend(txs);
        self
    }

    pub fn set_accounts_root(mut self, root: BlockHash) -> Self {
        self.accounts_root = Some(root);
        self
    }

    pub fn build(self) -> Block {
        let mut block = Block::new(self.slot, self.parent_hash, self.transactions, self.validator);
        
        if let Some(root) = self.accounts_root {
            block.header.accounts_root = root;
        }
        
        block
    }
}

pub struct BlockValidator {
    max_transactions_per_block: usize,
    max_block_size: usize,
    slot_duration_ms: u64,
}

impl BlockValidator {
    pub fn new() -> Self {
        Self {
            max_transactions_per_block: 10_000,
            max_block_size: 1_300_000,
            slot_duration_ms: 400,
        }
    }

    pub fn validate_block(&self, block: &Block, parent: Option<&Block>) -> Result<()> {
        block.validate_structure()?;

        if let Some(parent_block) = parent {
            block.validate_against_parent(parent_block)?;
        }

        block.verify_signature()?;
        block.validate_transactions()?;

        if block.transactions.len() > self.max_transactions_per_block {
            return Err(anyhow::anyhow!(
                "Too many transactions: {} > {}",
                block.transactions.len(),
                self.max_transactions_per_block
            ));
        }

        if let Ok(serialized) = block.serialize() {
            if serialized.len() > self.max_block_size {
                return Err(anyhow::anyhow!(
                    "Block too large: {} > {}",
                    serialized.len(),
                    self.max_block_size
                ));
            }
        }

        Ok(())
    }

    pub fn is_finalized(&self, block_slot: u64, current_slot: u64) -> bool {
        const FINALITY_THRESHOLD: u64 = 32;
        current_slot >= block_slot + FINALITY_THRESHOLD
    }
}

impl Default for BlockValidator {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn hash_header(header: &BlockHeader) -> Vec<u8> {
    let mut hasher = Sha256::new();
    let serialized = bincode::serialize(header).unwrap();
    hasher.update(&serialized);
    hasher.finalize().to_vec()
}

fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    #[test]
    fn test_block_creation() {
        let validator = vec![1u8; 32];
        let parent_hash = vec![0u8; 32];
        
        let block = Block::new(1, parent_hash, vec![], validator);
        
        assert_eq!(block.header.slot, 1);
        assert!(block.is_empty());
    }

    #[test]
    fn test_block_hash() {
        let block = Block::new(1, vec![0u8; 32], vec![], vec![1u8; 32]);
        let hash = block.hash();
        
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_block_signing() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let mut block = Block::new(1, vec![0u8; 32], vec![], keypair.public.to_bytes().to_vec());
        
        block.sign(&keypair).unwrap();
        assert!(!block.header.signature.is_empty());
        
        assert!(block.verify_signature().is_ok());
    }

    #[test]
    fn test_block_validation_against_parent() {
        let parent = Block::new(1, vec![0u8; 32], vec![], vec![1u8; 32]);
        let parent_hash = parent.hash();
        
        let child = Block::new(2, parent_hash, vec![], vec![1u8; 32]);
        
        assert!(child.validate_against_parent(&parent).is_ok());
    }

    #[test]
    fn test_invalid_slot_sequence() {
        let parent = Block::new(1, vec![0u8; 32], vec![], vec![1u8; 32]);
        let parent_hash = parent.hash();
        
        let child = Block::new(3, parent_hash, vec![], vec![1u8; 32]);
        
        assert!(child.validate_against_parent(&parent).is_err());
    }

    #[test]
    fn test_merkle_root() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        
        let block = Block::new(1, vec![0u8; 32], vec![tx], vec![1u8; 32]);
        
        assert_eq!(block.header.transactions_root.len(), 32);
        assert!(block.validate_structure().is_ok());
    }

    #[test]
    fn test_block_builder() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        
        let block = BlockBuilder::new(1, vec![0u8; 32], vec![1u8; 32])
            .add_transaction(tx)
            .build();
        
        assert_eq!(block.transaction_count(), 1);
    }

    #[test]
    fn test_block_validator() {
        let validator = BlockValidator::new();
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let mut block = Block::new(1, vec![0u8; 32], vec![], keypair.public.to_bytes().to_vec());
        block.sign(&keypair).unwrap();
        
        assert!(validator.validate_block(&block, None).is_ok());
    }

    #[test]
    fn test_finality() {
        let validator = BlockValidator::new();
        
        assert!(!validator.is_finalized(100, 120));
        assert!(validator.is_finalized(100, 132));
        assert!(validator.is_finalized(100, 200));
    }
}
