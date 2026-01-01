use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use ade_transaction::Transaction;

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
    pub timestamp: u64,
    pub validator: Vec<u8>,
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

        let header = BlockHeader {
            slot,
            parent_hash,
            transactions_root,
            timestamp,
            validator,
        };

        Self {
            header,
            transactions,
        }
    }

    pub fn hash(&self) -> BlockHash {
        let mut hasher = Sha256::new();
        let serialized = bincode::serialize(&self.header).unwrap();
        hasher.update(&serialized);
        hasher.finalize().to_vec()
    }

    fn compute_merkle_root(transactions: &[Transaction]) -> BlockHash {
        if transactions.is_empty() {
            return vec![0u8; 32];
        }

        let mut hasher = Sha256::new();
        for tx in transactions {
            hasher.update(&tx.hash());
        }
        hasher.finalize().to_vec()
    }

    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

pub struct BlockBuilder {
    slot: u64,
    parent_hash: BlockHash,
    transactions: Vec<Transaction>,
    validator: Vec<u8>,
}

impl BlockBuilder {
    pub fn new(slot: u64, parent_hash: BlockHash, validator: Vec<u8>) -> Self {
        Self {
            slot,
            parent_hash,
            transactions: Vec::new(),
            validator,
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

    pub fn build(self) -> Block {
        Block::new(self.slot, self.parent_hash, self.transactions, self.validator)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}


