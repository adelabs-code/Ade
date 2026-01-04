use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

use ade_transaction::Transaction;
use crate::utils::{current_timestamp, to_base58};
use crate::errors::{MempoolError, MempoolResult};
use crate::storage::Storage;

/// Transaction mempool with priority ordering and DDoS protection
///
/// The mempool stores pending transactions and orders them by fee priority.
/// It enforces limits on capacity, per-account transactions, and validates
/// sender balance and nonce to prevent DDoS attacks with invalid transactions.
pub struct Mempool {
    /// Transactions indexed by signature
    transactions: Arc<RwLock<HashMap<Vec<u8>, MempoolTransaction>>>,
    /// Priority queue ordered by fee
    priority_queue: Arc<RwLock<BTreeMap<u64, Vec<Vec<u8>>>>>,
    /// Transactions by sender for nonce tracking
    by_sender: Arc<RwLock<HashMap<Vec<u8>, Vec<Vec<u8>>>>>,
    /// Expected nonce per sender (for sequential tx ordering)
    sender_nonces: Arc<RwLock<HashMap<Vec<u8>, u64>>>,
    /// Configuration
    config: MempoolConfig,
    /// Statistics
    stats: Arc<RwLock<MempoolStats>>,
    /// Optional storage reference for balance verification
    storage: Option<Arc<Storage>>,
    /// Cached account balances for fast verification
    balance_cache: Arc<RwLock<HashMap<Vec<u8>, u64>>>,
}

#[derive(Debug, Clone)]
pub struct MempoolConfig {
    /// Maximum total transactions in mempool
    pub max_transactions: usize,
    /// Maximum transactions per account
    pub max_per_account: usize,
    /// Minimum fee required
    pub min_fee: u64,
    /// Transaction expiration time in seconds
    pub expiration_time_secs: u64,
    /// Whether to verify sender balance before accepting transactions
    pub verify_balance: bool,
    /// Whether to verify nonce ordering
    pub verify_nonce: bool,
    /// Minimum balance required beyond the transaction amount + fee
    pub min_balance_buffer: u64,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_transactions: 100_000,
            max_per_account: 64,
            min_fee: 5000,
            expiration_time_secs: 120,
            verify_balance: true,
            verify_nonce: true,
            min_balance_buffer: 10_000, // Minimum extra lamports beyond tx cost
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
    /// Create a new mempool with given configuration
    pub fn new(config: MempoolConfig) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            priority_queue: Arc::new(RwLock::new(BTreeMap::new())),
            by_sender: Arc::new(RwLock::new(HashMap::new())),
            sender_nonces: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(MempoolStats::default())),
            storage: None,
            balance_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a mempool with storage for balance verification
    pub fn with_storage(config: MempoolConfig, storage: Arc<Storage>) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            priority_queue: Arc::new(RwLock::new(BTreeMap::new())),
            by_sender: Arc::new(RwLock::new(HashMap::new())),
            sender_nonces: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(MempoolStats::default())),
            storage: Some(storage),
            balance_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add transaction to mempool with full validation
    ///
    /// This validates:
    /// 1. Signature validity
    /// 2. Fee requirements
    /// 3. Sender balance (if storage is available)
    /// 4. Nonce ordering (if enabled)
    /// 5. Per-account limits
    ///
    /// # Arguments
    /// * `tx` - Transaction to add
    /// * `fee` - Base transaction fee
    /// * `priority_fee` - Additional priority fee
    ///
    /// # Returns
    /// Transaction signature if successful
    pub fn add_transaction(&self, tx: Transaction, fee: u64, priority_fee: u64) -> MempoolResult<Vec<u8>> {
        let signature = tx.signature()
            .ok_or(MempoolError::DuplicateTransaction)?
            .clone();

        // Check for duplicate transaction
        {
            let txs = self.transactions.read().unwrap();
            if txs.contains_key(&signature) {
                return Err(MempoolError::DuplicateTransaction);
            }
        }

        // Verify signature
        tx.verify()
            .map_err(|_| MempoolError::TransactionTooLarge { size: 0, max: 0 })?;

        // Check minimum fee
        if fee < self.config.min_fee {
            let mut stats = self.stats.write().unwrap();
            stats.total_rejected += 1;
            warn!("Transaction rejected: fee {} below minimum {}", fee, self.config.min_fee);
            return Err(MempoolError::FeeTooLow { fee, min_fee: self.config.min_fee });
        }

        let sender = tx.get_signers().first()
            .ok_or(MempoolError::DuplicateTransaction)?
            .clone();

        // Calculate total cost (amount + fee + priority_fee)
        let total_cost = tx.total_amount() + fee + priority_fee;

        // Verify sender balance (DDoS protection)
        if self.config.verify_balance {
            if let Err(e) = self.verify_sender_balance(&sender, total_cost) {
                let mut stats = self.stats.write().unwrap();
                stats.total_rejected += 1;
                warn!(
                    "Transaction rejected: sender {} insufficient balance for {} lamports",
                    to_base58(&sender), total_cost
                );
                return Err(e);
            }
        }

        // Verify nonce (prevent nonce gaps)
        if self.config.verify_nonce {
            if let Err(e) = self.verify_nonce(&sender, &tx) {
                let mut stats = self.stats.write().unwrap();
                stats.total_rejected += 1;
                warn!("Transaction rejected: nonce verification failed for {}", to_base58(&sender));
                return Err(e);
            }
        }

        // Check per-account limit
        {
            let by_sender = self.by_sender.read().unwrap();
            if let Some(sender_txs) = by_sender.get(&sender) {
                if sender_txs.len() >= self.config.max_per_account {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_rejected += 1;
                    warn!(
                        "Transaction rejected: account {} exceeds limit of {} pending transactions",
                        to_base58(&sender), self.config.max_per_account
                    );
                    return Err(MempoolError::AccountLimitExceeded);
                }
            }
        }

        // Evict if at capacity
        {
            let txs = self.transactions.read().unwrap();
            if txs.len() >= self.config.max_transactions {
                self.evict_lowest_priority()?;
            }
        }

        // Create mempool transaction
        let mempool_tx = MempoolTransaction {
            transaction: tx.clone(),
            signature: signature.clone(),
            fee,
            priority_fee,
            sender: sender.clone(),
            received_at: current_timestamp(),
            size: tx.size(),
            compute_budget: 1_000_000,
        };

        // Add to transactions map
        {
            let mut txs = self.transactions.write().unwrap();
            txs.insert(signature.clone(), mempool_tx);
        }

        // Add to priority queue
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
            by_sender.entry(sender.clone())
                .or_insert_with(Vec::new)
                .push(signature.clone());
        }

        // Update pending balance (deduct from cached balance)
        self.update_pending_balance(&sender, total_cost);

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_transactions += 1;
            
            let total_fee = fee + priority_fee;
            if total_fee > stats.highest_fee {
                stats.highest_fee = total_fee;
            }
            
            let total_txs = stats.total_transactions;
            stats.average_fee = (stats.average_fee * (total_txs - 1) + total_fee) / total_txs;
        }

        debug!("Added transaction to mempool: {}", to_base58(&signature));
        Ok(signature)
    }
    
    /// Verify sender has sufficient balance for the transaction
    fn verify_sender_balance(&self, sender: &[u8], required_amount: u64) -> MempoolResult<()> {
        // First check the cached balance (includes pending tx deductions)
        {
            let cache = self.balance_cache.read().unwrap();
            if let Some(&cached_balance) = cache.get(sender) {
                if cached_balance >= required_amount + self.config.min_balance_buffer {
                    return Ok(());
                }
                // Cached balance is insufficient
                return Err(MempoolError::InsufficientBalance);
            }
        }
        
        // Cache miss - fetch from storage
        if let Some(ref storage) = self.storage {
            match storage.get_account(sender) {
                Ok(Some(account_data)) => {
                    // Parse account to get balance
                    // Format: first 8 bytes are lamports
                    if account_data.len() >= 8 {
                        let lamports = u64::from_le_bytes(
                            account_data[0..8].try_into().unwrap_or([0u8; 8])
                        );
                        
                        // Cache the balance
                        {
                            let mut cache = self.balance_cache.write().unwrap();
                            cache.insert(sender.to_vec(), lamports);
                        }
                        
                        if lamports >= required_amount + self.config.min_balance_buffer {
                            return Ok(());
                        }
                    }
                    Err(MempoolError::InsufficientBalance)
                }
                Ok(None) => {
                    // Account doesn't exist - definitely can't pay
                    Err(MempoolError::InsufficientBalance)
                }
                Err(_) => {
                    // Storage error - allow transaction (fail safe)
                    warn!("Failed to verify balance from storage, allowing transaction");
                    Ok(())
                }
            }
        } else {
            // No storage configured - skip balance check
            Ok(())
        }
    }
    
    /// Verify transaction nonce is correct
    fn verify_nonce(&self, sender: &[u8], tx: &Transaction) -> MempoolResult<()> {
        let tx_nonce = tx.get_nonce().unwrap_or(0);
        
        let expected_nonce = {
            let nonces = self.sender_nonces.read().unwrap();
            nonces.get(sender).copied().unwrap_or(0)
        };
        
        // Allow nonce to be equal or greater (for out-of-order arrival)
        // But reject if too far in the future (> 64 nonces ahead)
        if tx_nonce < expected_nonce {
            warn!(
                "Nonce too low: got {}, expected >= {}",
                tx_nonce, expected_nonce
            );
            return Err(MempoolError::InvalidNonce);
        }
        
        if tx_nonce > expected_nonce + 64 {
            warn!(
                "Nonce too high: got {}, expected <= {}",
                tx_nonce, expected_nonce + 64
            );
            return Err(MempoolError::InvalidNonce);
        }
        
        Ok(())
    }
    
    /// Update the pending balance after adding a transaction
    fn update_pending_balance(&self, sender: &[u8], deduction: u64) {
        let mut cache = self.balance_cache.write().unwrap();
        if let Some(balance) = cache.get_mut(sender) {
            *balance = balance.saturating_sub(deduction);
        }
    }
    
    /// Update sender nonce after transaction confirmation
    pub fn update_sender_nonce(&self, sender: &[u8], new_nonce: u64) {
        let mut nonces = self.sender_nonces.write().unwrap();
        nonces.insert(sender.to_vec(), new_nonce);
    }
    
    /// Refresh balance cache for a sender
    pub fn refresh_balance_cache(&self, sender: &[u8], new_balance: u64) {
        let mut cache = self.balance_cache.write().unwrap();
        cache.insert(sender.to_vec(), new_balance);
    }
    
    /// Clear the balance cache
    pub fn clear_balance_cache(&self) {
        let mut cache = self.balance_cache.write().unwrap();
        cache.clear();
    }

    /// Get transaction from mempool by signature
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
    ///
    /// Returns up to `count` transactions ordered by total fee (descending)
    pub fn get_top_transactions(&self, count: usize) -> Vec<MempoolTransaction> {
        let mut result = Vec::new();
        let queue = self.priority_queue.read().unwrap();
        
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
    ///
    /// Removes transactions older than expiration_time_secs
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
    fn evict_lowest_priority(&self) -> MempoolResult<()> {
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
            Err(MempoolError::MempoolFull { current: 0, max: 0 })
        }
    }

    /// Get current mempool size
    pub fn size(&self) -> usize {
        self.transactions.read().unwrap().len()
    }

    /// Check if transaction exists in mempool
    pub fn contains(&self, signature: &[u8]) -> bool {
        self.transactions.read().unwrap().contains_key(signature)
    }

    /// Get mempool statistics
    pub fn get_stats(&self) -> MempoolStats {
        let stats = self.stats.read().unwrap().clone();
        let mut updated_stats = stats;
        updated_stats.total_transactions = self.size();
        updated_stats
    }

    /// Clear all transactions from mempool
    pub fn clear(&self) {
        self.transactions.write().unwrap().clear();
        self.priority_queue.write().unwrap().clear();
        self.by_sender.write().unwrap().clear();
    }
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
        mempool.add_transaction(tx2, 10000, 5000).unwrap();
        mempool.add_transaction(tx3, 8000, 0).unwrap();
        
        let top = mempool.get_top_transactions(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].fee + top[0].priority_fee, 15000);
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
}
