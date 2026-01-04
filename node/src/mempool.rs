use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};

use ade_transaction::Transaction;

/// Transaction mempool with priority ordering
pub struct Mempool {
    /// Transactions indexed by signature
    transactions: Arc<RwLock<HashMap<Vec<u8>, MempoolTransaction>>>,
    /// Priority queue ordered by fee
    priority_queue: Arc<RwLock<BTreeMap<u64, Vec<Vec<u8>>>>>,
    /// Transactions by sender for nonce tracking
    by_sender: Arc<RwLock<HashMap<Vec<u8>, Vec<Vec<u8>>>>>,
    /// Configuration
    config: MempoolConfig,
    /// Statistics
    stats: Arc<RwLock<MempoolStats>>,
}

#[derive(Debug, Clone)]
pub struct MempoolConfig {
    pub max_transactions: usize,
    pub max_per_account: usize,
    pub min_fee: u64,
    pub expiration_time_secs: u64,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_transactions: 100_000,
            max_per_account: 64,
            min_fee: 5000,
            expiration_time_secs: 120,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MempoolTransaction {
    pub transaction: Transaction,
    pub signature: Vec<u8>,
    pub fee: u64,
    pub priority_fee: u64,
    pub sender: Vec<u8>,
    pub received_at: u64,
    pub size: usize,
    pub compute_budget: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolStats {
    pub total_transactions: usize,
    pub total_rejected: u64,
    pub total_expired: u64,
    pub total_evicted: u64,
    pub average_fee: u64,
    pub highest_fee: u64,
}

impl Default for MempoolStats {
    fn default() -> Self {
        Self {
            total_transactions: 0,
            total_rejected: 0,
            total_expired: 0,
            total_evicted: 0,
            average_fee: 0,
            highest_fee: 0,
        }
    }
}

impl Mempool {
    pub fn new(config: MempoolConfig) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            priority_queue: Arc::new(RwLock::new(BTreeMap::new())),
            by_sender: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(MempoolStats::default())),
        }
    }

    /// Add transaction to mempool
    pub fn add_transaction(&self, tx: Transaction, fee: u64, priority_fee: u64) -> Result<Vec<u8>> {
        let signature = tx.signature()
            .ok_or_else(|| anyhow::anyhow!("Transaction has no signature"))?
            .clone();

        // Check if already in mempool
        {
            let txs = self.transactions.read().unwrap();
            if txs.contains_key(&signature) {
                return Err(anyhow::anyhow!("Transaction already in mempool"));
            }
        }

        // Verify transaction
        tx.verify()
            .map_err(|e| anyhow::anyhow!("Transaction verification failed: {}", e))?;

        // Check fee requirement
        if fee < self.config.min_fee {
            let mut stats = self.stats.write().unwrap();
            stats.total_rejected += 1;
            return Err(anyhow::anyhow!("Fee too low: {} < {}", fee, self.config.min_fee));
        }

        let sender = tx.get_signers().first()
            .ok_or_else(|| anyhow::anyhow!("No sender found"))?
            .clone();

        // Check per-account limit
        {
            let by_sender = self.by_sender.read().unwrap();
            if let Some(sender_txs) = by_sender.get(&sender) {
                if sender_txs.len() >= self.config.max_per_account {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_rejected += 1;
                    return Err(anyhow::anyhow!("Too many transactions from this account"));
                }
            }
        }

        // Check total capacity
        {
            let txs = self.transactions.read().unwrap();
            if txs.len() >= self.config.max_transactions {
                // Evict lowest priority transaction
                self.evict_lowest_priority()?;
            }
        }

        let mempool_tx = MempoolTransaction {
            transaction: tx.clone(),
            signature: signature.clone(),
            fee,
            priority_fee,
            sender: sender.clone(),
            received_at: current_timestamp(),
            size: tx.size(),
            compute_budget: 1_000_000, // Default, would be parsed from tx
        };

        // Add to main storage
        {
            let mut txs = self.transactions.write().unwrap();
            txs.insert(signature.clone(), mempool_tx);
        }

        // Add to priority queue (using total fee as priority)
        {
            let total_fee = fee + priority_fee;
            let mut queue = self.priority_queue.write().unwrap();
            queue.entry(total_fee)
                .or_insert_with(Vec::new)
                .push(signature.clone());
        }

        // Add to sender index
        {
            let mut by_sender = self.by_sender.write().unwrap();
            by_sender.entry(sender)
                .or_insert_with(Vec::new)
                .push(signature.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_transactions += 1;
            
            let total_fee = fee + priority_fee;
            if total_fee > stats.highest_fee {
                stats.highest_fee = total_fee;
            }
            
            // Update average
            let total_txs = stats.total_transactions;
            stats.average_fee = (stats.average_fee * (total_txs - 1) + total_fee) / total_txs;
        }

        info!("Added transaction to mempool: {:?}", bs58::encode(&signature).into_string());
        Ok(signature)
    }

    /// Get transaction from mempool
    pub fn get_transaction(&self, signature: &[u8]) -> Option<MempoolTransaction> {
        let txs = self.transactions.read().unwrap();
        txs.get(signature).cloned()
    }

    /// Remove transaction from mempool
    pub fn remove_transaction(&self, signature: &[u8]) -> Option<MempoolTransaction> {
        let removed = {
            let mut txs = self.transactions.write().unwrap();
            txs.remove(signature)
        };

        if let Some(ref tx) = removed {
            // Remove from priority queue
            {
                let total_fee = tx.fee + tx.priority_fee;
                let mut queue = self.priority_queue.write().unwrap();
                if let Some(sigs) = queue.get_mut(&total_fee) {
                    sigs.retain(|s| s != signature);
                    if sigs.is_empty() {
                        queue.remove(&total_fee);
                    }
                }
            }

            // Remove from sender index
            {
                let mut by_sender = self.by_sender.write().unwrap();
                if let Some(sender_txs) = by_sender.get_mut(&tx.sender) {
                    sender_txs.retain(|s| s != signature);
                    if sender_txs.is_empty() {
                        by_sender.remove(&tx.sender);
                    }
                }
            }
        }

        removed
    }

    /// Get highest priority transactions
    pub fn get_top_transactions(&self, count: usize) -> Vec<MempoolTransaction> {
        let mut result = Vec::new();
        let queue = self.priority_queue.read().unwrap();
        
        // Iterate from highest fee to lowest
        for (_, signatures) in queue.iter().rev() {
            if result.len() >= count {
                break;
            }
            
            let txs = self.transactions.read().unwrap();
            for sig in signatures {
                if let Some(tx) = txs.get(sig) {
                    result.push(tx.clone());
                    if result.len() >= count {
                        break;
                    }
                }
            }
        }

        result
    }

    /// Get all transactions from mempool
    pub fn get_all_transactions(&self) -> Vec<MempoolTransaction> {
        let txs = self.transactions.read().unwrap();
        txs.values().cloned().collect()
    }

    /// Get transactions by sender
    pub fn get_transactions_by_sender(&self, sender: &[u8]) -> Vec<MempoolTransaction> {
        let by_sender = self.by_sender.read().unwrap();
        
        if let Some(signatures) = by_sender.get(sender) {
            let txs = self.transactions.read().unwrap();
            signatures.iter()
                .filter_map(|sig| txs.get(sig).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Prune expired transactions
    pub fn prune_expired(&self) -> usize {
        let cutoff = current_timestamp() - self.config.expiration_time_secs;
        let mut expired_sigs = Vec::new();

        {
            let txs = self.transactions.read().unwrap();
            for (sig, tx) in txs.iter() {
                if tx.received_at < cutoff {
                    expired_sigs.push(sig.clone());
                }
            }
        }

        let count = expired_sigs.len();
        for sig in expired_sigs {
            self.remove_transaction(&sig);
        }

        if count > 0 {
            let mut stats = self.stats.write().unwrap();
            stats.total_expired += count as u64;
            info!("Pruned {} expired transactions", count);
        }

        count
    }

    /// Evict lowest priority transaction
    fn evict_lowest_priority(&self) -> Result<()> {
        let to_evict = {
            let queue = self.priority_queue.read().unwrap();
            queue.iter()
                .next()
                .and_then(|(_, sigs)| sigs.first().cloned())
        };

        if let Some(sig) = to_evict {
            self.remove_transaction(&sig);
            
            let mut stats = self.stats.write().unwrap();
            stats.total_evicted += 1;
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("No transaction to evict"))
        }
    }

    /// Get mempool size
    pub fn size(&self) -> usize {
        self.transactions.read().unwrap().len()
    }

    /// Check if transaction exists
    pub fn contains(&self, signature: &[u8]) -> bool {
        self.transactions.read().unwrap().contains_key(signature)
    }

    /// Get statistics
    pub fn get_stats(&self) -> MempoolStats {
        let stats = self.stats.read().unwrap().clone();
        let mut updated_stats = stats;
        updated_stats.total_transactions = self.size();
        updated_stats
    }

    /// Clear mempool
    pub fn clear(&self) {
        self.transactions.write().unwrap().clear();
        self.priority_queue.write().unwrap().clear();
        self.by_sender.write().unwrap().clear();
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    fn create_test_transaction() -> Transaction {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap()
    }

    #[test]
    fn test_mempool_add() {
        let mempool = Mempool::new(MempoolConfig::default());
        let tx = create_test_transaction();
        
        let result = mempool.add_transaction(tx, 10000, 1000);
        assert!(result.is_ok());
        assert_eq!(mempool.size(), 1);
    }

    #[test]
    fn test_mempool_duplicate() {
        let mempool = Mempool::new(MempoolConfig::default());
        let tx = create_test_transaction();
        
        mempool.add_transaction(tx.clone(), 10000, 1000).unwrap();
        let result = mempool.add_transaction(tx, 10000, 1000);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_priority_ordering() {
        let mempool = Mempool::new(MempoolConfig::default());
        
        let tx1 = create_test_transaction();
        let tx2 = create_test_transaction();
        let tx3 = create_test_transaction();
        
        mempool.add_transaction(tx1, 5000, 0).unwrap();
        mempool.add_transaction(tx2, 10000, 5000).unwrap(); // Highest
        mempool.add_transaction(tx3, 8000, 0).unwrap();
        
        let top = mempool.get_top_transactions(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].fee + top[0].priority_fee, 15000); // Highest first
    }

    #[test]
    fn test_transaction_removal() {
        let mempool = Mempool::new(MempoolConfig::default());
        let tx = create_test_transaction();
        
        let sig = mempool.add_transaction(tx, 10000, 1000).unwrap();
        assert_eq!(mempool.size(), 1);
        
        let removed = mempool.remove_transaction(&sig);
        assert!(removed.is_some());
        assert_eq!(mempool.size(), 0);
    }

    #[test]
    fn test_fee_too_low() {
        let mempool = Mempool::new(MempoolConfig::default());
        let tx = create_test_transaction();
        
        let result = mempool.add_transaction(tx, 1000, 0);
        assert!(result.is_err());
        
        let stats = mempool.get_stats();
        assert_eq!(stats.total_rejected, 1);
    }

    #[test]
    fn test_per_account_limit() {
        let mut config = MempoolConfig::default();
        config.max_per_account = 2;
        let mempool = Mempool::new(config);
        
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        // Add 2 transactions from same sender
        for _ in 0..2 {
            let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
            mempool.add_transaction(tx, 10000, 0).unwrap();
        }
        
        // Third should fail
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        let result = mempool.add_transaction(tx, 10000, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_capacity_eviction() {
        let mut config = MempoolConfig::default();
        config.max_transactions = 3;
        let mempool = Mempool::new(config);
        
        // Add 3 transactions with different fees
        mempool.add_transaction(create_test_transaction(), 10000, 0).unwrap();
        mempool.add_transaction(create_test_transaction(), 15000, 0).unwrap();
        mempool.add_transaction(create_test_transaction(), 20000, 0).unwrap();
        
        // Add 4th - should evict lowest fee
        mempool.add_transaction(create_test_transaction(), 25000, 0).unwrap();
        
        assert_eq!(mempool.size(), 3);
        
        let stats = mempool.get_stats();
        assert_eq!(stats.total_evicted, 1);
    }

    #[test]
    fn test_get_transactions_by_sender() {
        let mempool = Mempool::new(MempoolConfig::default());
        
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let sender = keypair.public.to_bytes().to_vec();
        
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        mempool.add_transaction(tx, 10000, 0).unwrap();
        
        let sender_txs = mempool.get_transactions_by_sender(&sender);
        assert_eq!(sender_txs.len(), 1);
    }
}




