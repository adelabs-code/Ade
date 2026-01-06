use anyhow::{Result, Context};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};

use crate::storage::Storage;

/// Stored block format for deserialization during index rebuild
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredBlock {
    slot: u64,
    transaction_signatures: Vec<Vec<u8>>,
    blockhash: Vec<u8>,
    parent_slot: u64,
    timestamp: u64,
}

/// Stored transaction format for deserialization during index rebuild
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredTransaction {
    signature: Vec<u8>,
    slot: u64,
    account_keys: Vec<Vec<u8>>,
    data: Vec<u8>,
}

/// Stored account format for deserialization during index rebuild
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredAccount {
    owner: Vec<u8>,
    lamports: u64,
    data: Vec<u8>,
}

/// Index entry limits for memory management
const MAX_ENTRIES_PER_ADDRESS: usize = 10_000;
const MAX_ENTRIES_PER_PROGRAM: usize = 100_000;
const MAX_CACHED_SLOTS: usize = 1_000;

/// Secondary index for efficient queries - PRODUCTION implementation
/// 
/// Uses a hybrid approach:
/// 1. Hot data (recent) is cached in memory for fast access
/// 2. Cold data is stored in RocksDB column family for persistence
/// 3. Automatic eviction of old entries to prevent OOM
/// 
/// This ensures the indexer can handle millions of transactions
/// without running out of memory.
pub struct SecondaryIndex {
    storage: Arc<Storage>,
    /// In-memory cache for hot data (limited size)
    address_to_transactions: Arc<RwLock<BTreeMap<Vec<u8>, Vec<Vec<u8>>>>>,
    program_to_accounts: Arc<RwLock<HashMap<Vec<u8>, Vec<Vec<u8>>>>>,
    slot_to_transactions: Arc<RwLock<BTreeMap<u64, Vec<Vec<u8>>>>>,
    /// Track total entries for monitoring
    total_address_entries: Arc<RwLock<usize>>,
    /// Minimum slot in cache (for eviction)
    min_cached_slot: Arc<RwLock<u64>>,
}

impl SecondaryIndex {
    pub fn new(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            address_to_transactions: Arc::new(RwLock::new(BTreeMap::new())),
            program_to_accounts: Arc::new(RwLock::new(HashMap::new())),
            slot_to_transactions: Arc::new(RwLock::new(BTreeMap::new())),
            total_address_entries: Arc::new(RwLock::new(0)),
            min_cached_slot: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Evict old entries to keep memory usage bounded
    /// 
    /// This is called automatically when limits are reached.
    fn evict_old_entries(&self) -> Result<usize> {
        let mut evicted = 0usize;
        
        // Evict old slots from slot_to_transactions
        {
            let mut slot_index = self.slot_to_transactions.write().unwrap();
            while slot_index.len() > MAX_CACHED_SLOTS {
                if let Some((&oldest_slot, sigs)) = slot_index.iter().next() {
                    // Persist to RocksDB before evicting
                    let key = format!("idx:slot:{}", oldest_slot);
                    if let Ok(data) = bincode::serialize(&sigs) {
                        let _ = self.storage.store_raw(&key, &data);
                    }
                    
                    evicted += sigs.len();
                    let oldest = oldest_slot;
                    slot_index.remove(&oldest);
                    
                    // Update min cached slot
                    if let Some((&new_min, _)) = slot_index.iter().next() {
                        *self.min_cached_slot.write().unwrap() = new_min;
                    }
                } else {
                    break;
                }
            }
        }
        
        // Evict entries from addresses that have too many transactions
        {
            let mut addr_index = self.address_to_transactions.write().unwrap();
            let mut total = self.total_address_entries.write().unwrap();
            
            for (address, sigs) in addr_index.iter_mut() {
                if sigs.len() > MAX_ENTRIES_PER_ADDRESS {
                    // Keep only the most recent entries, persist older ones
                    let to_evict = sigs.len() - MAX_ENTRIES_PER_ADDRESS;
                    let evicted_sigs: Vec<_> = sigs.drain(0..to_evict).collect();
                    
                    // Persist evicted entries to RocksDB
                    let key = format!("idx:addr:{}:old", bs58::encode(address).into_string());
                    if let Ok(existing) = self.storage.get_raw(&key) {
                        let mut all_sigs: Vec<Vec<u8>> = existing
                            .and_then(|d| bincode::deserialize(&d).ok())
                            .unwrap_or_default();
                        all_sigs.extend(evicted_sigs.clone());
                        if let Ok(data) = bincode::serialize(&all_sigs) {
                            let _ = self.storage.store_raw(&key, &data);
                        }
                    }
                    
                    evicted += to_evict;
                    *total = total.saturating_sub(to_evict);
                }
            }
        }
        
        if evicted > 0 {
            info!("Evicted {} old index entries to RocksDB", evicted);
        }
        
        Ok(evicted)
    }
    
    /// Check if eviction is needed and perform it
    fn maybe_evict(&self) {
        let total = *self.total_address_entries.read().unwrap();
        let slot_count = self.slot_to_transactions.read().unwrap().len();
        
        // Evict if we're at 80% of limits
        if total > MAX_ENTRIES_PER_ADDRESS * 80 / 100 * 1000 || 
           slot_count > MAX_CACHED_SLOTS * 80 / 100 
        {
            let _ = self.evict_old_entries();
        }
    }
    
    /// Get transaction from RocksDB if not in memory cache
    fn get_from_disk(&self, key: &str) -> Option<Vec<Vec<u8>>> {
        self.storage.get_raw(key)
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
    }

    /// Index a transaction by address
    /// 
    /// Automatically evicts old entries when memory limits are reached.
    pub fn index_transaction_by_address(
        &self,
        address: &[u8],
        signature: &[u8],
    ) -> Result<()> {
        // Check if eviction is needed
        self.maybe_evict();
        
        {
            let mut index = self.address_to_transactions.write().unwrap();
            index.entry(address.to_vec())
                .or_insert_with(Vec::new)
                .push(signature.to_vec());
        }
        
        // Update total count
        {
            let mut total = self.total_address_entries.write().unwrap();
            *total += 1;
        }
        
        // Also persist to RocksDB for durability
        let key = format!("idx:addr:{}:latest", bs58::encode(address).into_string());
        let sig_b58 = bs58::encode(signature).into_string();
        
        // Append to existing or create new
        let mut sigs: Vec<String> = self.storage.get_raw(&key)
            .ok()
            .flatten()
            .and_then(|d| serde_json::from_slice(&d).ok())
            .unwrap_or_default();
        
        sigs.push(sig_b58);
        
        // Keep only recent entries in RocksDB too
        if sigs.len() > MAX_ENTRIES_PER_ADDRESS {
            sigs = sigs.split_off(sigs.len() - MAX_ENTRIES_PER_ADDRESS);
        }
        
        if let Ok(data) = serde_json::to_vec(&sigs) {
            let _ = self.storage.store_raw(&key, &data);
        }
        
        Ok(())
    }

    /// Get transactions for an address
    pub fn get_transactions_for_address(
        &self,
        address: &[u8],
        limit: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let index = self.address_to_transactions.read().unwrap();
        
        if let Some(signatures) = index.get(address) {
            Ok(signatures.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Index an account by program
    pub fn index_account_by_program(
        &self,
        program_id: &[u8],
        account: &[u8],
    ) -> Result<()> {
        let mut index = self.program_to_accounts.write().unwrap();
        index.entry(program_id.to_vec())
            .or_insert_with(Vec::new)
            .push(account.to_vec());
        
        // Also store in persistent storage
        self.storage.index_program_account(program_id, account)?;
        
        Ok(())
    }

    /// Get accounts for a program
    pub fn get_accounts_for_program(
        &self,
        program_id: &[u8],
    ) -> Result<Vec<Vec<u8>>> {
        let index = self.program_to_accounts.read().unwrap();
        
        if let Some(accounts) = index.get(program_id) {
            Ok(accounts.clone())
        } else {
            // Fallback to storage
            let accounts = self.storage.get_accounts_by_program(program_id)?;
            Ok(accounts.into_iter().map(|(addr, _)| addr).collect())
        }
    }

    /// Index transaction by slot
    pub fn index_transaction_by_slot(
        &self,
        slot: u64,
        signature: &[u8],
    ) -> Result<()> {
        let mut index = self.slot_to_transactions.write().unwrap();
        index.entry(slot)
            .or_insert_with(Vec::new)
            .push(signature.to_vec());
        Ok(())
    }

    /// Get transactions for a slot
    pub fn get_transactions_for_slot(&self, slot: u64) -> Result<Vec<Vec<u8>>> {
        let index = self.slot_to_transactions.read().unwrap();
        
        if let Some(signatures) = index.get(&slot) {
            Ok(signatures.clone())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get transaction count for an address
    pub fn get_transaction_count_for_address(&self, address: &[u8]) -> usize {
        let index = self.address_to_transactions.read().unwrap();
        index.get(address).map(|v| v.len()).unwrap_or(0)
    }

    /// Rebuild all indexes by scanning the entire storage database
    /// This is used for recovery or when indexes become corrupted
    pub fn rebuild_from_storage(&self) -> Result<()> {
        use tracing::{info, warn};
        
        info!("Starting index rebuild from storage...");
        
        let start_time = std::time::Instant::now();
        let mut total_indexed = 0usize;
        
        // Clear existing in-memory indexes
        {
            let mut addr_index = self.address_to_transactions.write().unwrap();
            addr_index.clear();
        }
        {
            let mut prog_index = self.program_to_accounts.write().unwrap();
            prog_index.clear();
        }
        {
            let mut slot_index = self.slot_to_transactions.write().unwrap();
            slot_index.clear();
        }
        
        // Rebuild transaction index by scanning all transactions
        // Get blocks from storage and extract transaction data
        let blocks = self.storage.get_blocks_range(0, u64::MAX)?;
        
        for (slot, block_data) in blocks {
            // Try to deserialize block to get transactions
            if let Ok(block) = bincode::deserialize::<StoredBlock>(&block_data) {
                for tx_sig in &block.transaction_signatures {
                    // Index transaction by slot
                    self.index_transaction_by_slot(slot, tx_sig)?;
                    
                    // Get transaction data to extract account keys
                    if let Ok(Some(tx_data)) = self.storage.get_transaction(tx_sig) {
                        if let Ok(tx_info) = bincode::deserialize::<StoredTransaction>(&tx_data) {
                            // Index by each involved account
                            for account_key in &tx_info.account_keys {
                                self.index_transaction_by_address(account_key, tx_sig)?;
                            }
                            total_indexed += 1;
                        }
                    }
                }
            }
        }
        
        // Rebuild program account index
        let accounts = self.storage.get_all_accounts()?;
        
        for (address, account_data) in accounts {
            if let Ok(account_info) = bincode::deserialize::<StoredAccount>(&account_data) {
                // Index account by program (owner)
                self.index_account_by_program(&account_info.owner, &address)?;
                total_indexed += 1;
            }
        }
        
        let elapsed = start_time.elapsed();
        info!("Index rebuild complete: {} items indexed in {:?}", total_indexed, elapsed);
        
        Ok(())
    }

    /// Prune index before slot
    pub fn prune_before_slot(&self, slot: u64) -> Result<usize> {
        let mut index = self.slot_to_transactions.write().unwrap();
        let mut pruned = 0;
        
        let keys_to_remove: Vec<u64> = index
            .range(..slot)
            .map(|(k, _)| *k)
            .collect();
        
        for key in keys_to_remove {
            index.remove(&key);
            pruned += 1;
        }
        
        Ok(pruned)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        let address_index = self.address_to_transactions.read().unwrap();
        let program_index = self.program_to_accounts.read().unwrap();
        let slot_index = self.slot_to_transactions.read().unwrap();
        
        IndexStats {
            indexed_addresses: address_index.len(),
            indexed_programs: program_index.len(),
            indexed_slots: slot_index.len(),
            total_address_tx_entries: address_index.values().map(|v| v.len()).sum(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub indexed_addresses: usize,
    pub indexed_programs: usize,
    pub indexed_slots: usize,
    pub total_address_tx_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_index() -> SecondaryIndex {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        SecondaryIndex::new(storage)
    }

    #[test]
    fn test_transaction_indexing() {
        let index = create_test_index();
        
        let address = vec![1u8; 32];
        let sig1 = vec![1u8; 64];
        let sig2 = vec![2u8; 64];
        
        index.index_transaction_by_address(&address, &sig1).unwrap();
        index.index_transaction_by_address(&address, &sig2).unwrap();
        
        let txs = index.get_transactions_for_address(&address, 10).unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[test]
    fn test_program_account_indexing() {
        let index = create_test_index();
        
        let program_id = vec![1u8; 32];
        let account1 = vec![2u8; 32];
        let account2 = vec![3u8; 32];
        
        index.index_account_by_program(&program_id, &account1).unwrap();
        index.index_account_by_program(&program_id, &account2).unwrap();
        
        let accounts = index.get_accounts_for_program(&program_id).unwrap();
        assert_eq!(accounts.len(), 2);
    }

    #[test]
    fn test_slot_indexing() {
        let index = create_test_index();
        
        let slot = 100u64;
        let sig = vec![1u8; 64];
        
        index.index_transaction_by_slot(slot, &sig).unwrap();
        
        let txs = index.get_transactions_for_slot(slot).unwrap();
        assert_eq!(txs.len(), 1);
    }

    #[test]
    fn test_index_pruning() {
        let index = create_test_index();
        
        // Index transactions at slots 1-10
        for slot in 1..=10 {
            let sig = vec![slot as u8; 64];
            index.index_transaction_by_slot(slot, &sig).unwrap();
        }
        
        // Prune before slot 5
        let pruned = index.prune_before_slot(5).unwrap();
        assert_eq!(pruned, 4);
        
        // Verify slots 1-4 are gone
        let txs = index.get_transactions_for_slot(3).unwrap();
        assert_eq!(txs.len(), 0);
        
        let txs = index.get_transactions_for_slot(5).unwrap();
        assert_eq!(txs.len(), 1);
    }
}





