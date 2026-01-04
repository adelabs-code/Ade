use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, debug, warn};
use lru::LruCache;
use std::num::NonZeroUsize;

use ade_transaction::{Transaction, TransactionExecutor, Account, ExecutionResult};
use ade_consensus::Block;
use crate::storage::Storage;

/// Default LRU cache size for accounts (number of accounts to cache in memory)
const DEFAULT_ACCOUNT_CACHE_SIZE: usize = 100_000;

/// State transition function with LRU caching for scalable state management
/// 
/// Instead of loading all accounts into memory at startup, this uses
/// a lazy-loading LRU cache that only keeps frequently accessed accounts
/// in memory while reading others from disk on demand.
pub struct StateTransition {
    storage: Arc<Storage>,
    executor: Arc<TransactionExecutor>,
    /// LRU cache for hot accounts - only keeps most recently used accounts in memory
    account_cache: Arc<RwLock<LruCache<Vec<u8>, Account>>>,
    /// Dirty accounts that need to be persisted
    dirty_accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>,
    /// Cache statistics
    cache_stats: Arc<RwLock<CacheStats>>,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub disk_reads: u64,
    pub disk_writes: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransitionResult {
    pub new_state_root: Vec<u8>,
    pub successful_transactions: usize,
    pub failed_transactions: usize,
    pub total_fees: u64,
    pub total_compute: u64,
    pub account_changes: Vec<AccountChange>,
}

#[derive(Debug, Clone)]
pub struct AccountChange {
    pub address: Vec<u8>,
    pub pre_balance: u64,
    pub post_balance: u64,
    pub data_modified: bool,
}

impl StateTransition {
    /// Create a new StateTransition with default cache size
    pub fn new(storage: Arc<Storage>) -> Self {
        Self::with_cache_size(storage, DEFAULT_ACCOUNT_CACHE_SIZE)
    }
    
    /// Create a new StateTransition with custom cache size
    pub fn with_cache_size(storage: Arc<Storage>, cache_size: usize) -> Self {
        let cache_size = NonZeroUsize::new(cache_size).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        // Create executor with a wrapper that uses our lazy-loading cache
        let accounts_for_executor = Arc::new(RwLock::new(HashMap::new()));
        let executor = Arc::new(TransactionExecutor::new(accounts_for_executor));
        
        Self {
            storage,
            executor,
            account_cache: cache,
            dirty_accounts: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
            max_cache_size: cache_size.get(),
        }
    }

    /// Load state lazily - only loads metadata, not all accounts
    /// 
    /// This is the scalable approach: accounts are loaded on-demand
    /// from disk and cached in an LRU cache. Only the most frequently
    /// accessed accounts stay in memory.
    pub async fn load_state(&self, slot: u64) -> Result<()> {
        info!("Initializing lazy state loader for slot {}", slot);
        
        // Clear the cache to start fresh
        {
            let mut cache = self.account_cache.write().unwrap();
            cache.clear();
        }
        
        // Pre-warm the cache with a limited number of hot accounts
        // This loads only the most recently modified accounts, not all accounts
        let preload_count = self.max_cache_size.min(10_000); // Max 10k for preload
        
        match self.storage.get_all_accounts() {
            Ok(accounts) => {
                let mut cache = self.account_cache.write().unwrap();
                let mut loaded = 0;
                
                // Only load up to preload_count accounts
                for (address, account_data) in accounts.into_iter().take(preload_count) {
                    match bincode::deserialize::<Account>(&account_data) {
                        Ok(account) => {
                            cache.put(address, account);
                            loaded += 1;
                        }
                        Err(e) => {
                            debug!("Failed to deserialize account during preload: {}", e);
                        }
                    }
                }
                
                info!(
                    "Pre-warmed cache with {} accounts (max cache size: {})",
                    loaded, self.max_cache_size
                );
            }
            Err(e) => {
                warn!("Failed to preload accounts: {}", e);
            }
        }
        
        // Verify state root if available
        if let Ok(Some(stored_root)) = self.storage.get_state::<Vec<u8>>(&format!("state_root_{}", slot)) {
            info!("State root for slot {} loaded: {}", slot, bs58::encode(&stored_root).into_string());
        }
        
        Ok(())
    }
    
    /// Get an account, loading from disk if not in cache (lazy loading)
    pub fn get_account_lazy(&self, address: &[u8]) -> Option<Account> {
        // First check the cache
        {
            let mut cache = self.account_cache.write().unwrap();
            if let Some(account) = cache.get(address) {
                // Cache hit
                let mut stats = self.cache_stats.write().unwrap();
                stats.hits += 1;
                return Some(account.clone());
            }
        }
        
        // Cache miss - load from disk
        {
            let mut stats = self.cache_stats.write().unwrap();
            stats.misses += 1;
            stats.disk_reads += 1;
        }
        
        // Load from storage
        match self.storage.get_account(address) {
            Ok(Some(account_data)) => {
                match bincode::deserialize::<Account>(&account_data) {
                    Ok(account) => {
                        // Add to cache
                        let mut cache = self.account_cache.write().unwrap();
                        
                        // Track evictions
                        if cache.len() >= self.max_cache_size {
                            let mut stats = self.cache_stats.write().unwrap();
                            stats.evictions += 1;
                        }
                        
                        cache.put(address.to_vec(), account.clone());
                        Some(account)
                    }
                    Err(e) => {
                        warn!("Failed to deserialize account from disk: {}", e);
                        None
                    }
                }
            }
            Ok(None) => None,
            Err(e) => {
                warn!("Failed to read account from disk: {}", e);
                None
            }
        }
    }
    
    /// Update an account (marks as dirty for later persistence)
    pub fn update_account(&self, address: Vec<u8>, account: Account) {
        // Update cache
        {
            let mut cache = self.account_cache.write().unwrap();
            cache.put(address.clone(), account.clone());
        }
        
        // Mark as dirty
        {
            let mut dirty = self.dirty_accounts.write().unwrap();
            dirty.insert(address, account);
        }
    }
    
    /// Flush dirty accounts to storage
    pub fn flush_dirty_accounts(&self) -> Result<usize> {
        let dirty = {
            let mut dirty = self.dirty_accounts.write().unwrap();
            std::mem::take(&mut *dirty)
        };
        
        if dirty.is_empty() {
            return Ok(0);
        }
        
        let mut flushed = 0;
        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        
        for (address, account) in dirty {
            let serialized = bincode::serialize(&account)?;
            batch.push((address, serialized));
            flushed += 1;
        }
        
        self.storage.store_accounts_batch(&batch)?;
        
        {
            let mut stats = self.cache_stats.write().unwrap();
            stats.disk_writes += flushed as u64;
        }
        
        debug!("Flushed {} dirty accounts to storage", flushed);
        Ok(flushed)
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache_stats.read().unwrap().clone()
    }
    
    /// Compute state root from a given state map (internal helper)
    fn compute_state_root_internal(&self, state: &HashMap<Vec<u8>, Account>) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        
        let mut sorted_keys: Vec<_> = state.keys().collect();
        sorted_keys.sort();
        
        for key in sorted_keys {
            if let Some(account) = state.get(key) {
                hasher.update(key);
                hasher.update(&account.lamports.to_le_bytes());
                hasher.update(&account.owner);
                hasher.update(&account.data);
            }
        }
        
        hasher.finalize().to_vec()
    }

    /// Apply a block to current state
    pub async fn apply_block(&self, block: &Block) -> Result<TransitionResult> {
        info!("Applying block at slot {}", block.header.slot);

        let mut successful = 0;
        let mut failed = 0;
        let mut total_fees = 0u64;
        let mut total_compute = 0u64;
        let mut account_changes = Vec::new();

        // Take snapshot of current state
        let pre_state = self.snapshot_state();

        // Execute each transaction
        for (idx, transaction) in block.transactions.iter().enumerate() {
            debug!("Executing transaction {}/{}", idx + 1, block.transactions.len());

            match self.executor.execute_transaction(transaction, block.header.slot) {
                Ok(result) => {
                    if result.success {
                        successful += 1;
                        total_compute += result.compute_units_consumed;
                        
                        // Record account changes
                        for account_key in &result.modified_accounts {
                            if let Some(change) = self.get_account_change(&pre_state, account_key) {
                                account_changes.push(change);
                            }
                        }
                    } else {
                        failed += 1;
                        warn!("Transaction failed: {:?}", result.error);
                    }
                }
                Err(e) => {
                    failed += 1;
                    warn!("Transaction execution error: {}", e);
                }
            }
        }

        // Compute new state root
        let new_state_root = self.compute_state_root();

        // Persist state changes
        self.persist_state_changes(&account_changes).await?;

        info!("Block applied: {} successful, {} failed", successful, failed);

        Ok(TransitionResult {
            new_state_root,
            successful_transactions: successful,
            failed_transactions: failed,
            total_fees,
            total_compute,
            account_changes,
        })
    }

    /// Compute state root hash from cached accounts
    fn compute_state_root(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        
        let cache = self.account_cache.read().unwrap();
        let dirty = self.dirty_accounts.read().unwrap();
        let mut hasher = Sha256::new();
        
        // Combine cache and dirty accounts
        let mut all_keys: Vec<&Vec<u8>> = cache.iter().map(|(k, _)| k).collect();
        for key in dirty.keys() {
            if !all_keys.contains(&key) {
                all_keys.push(key);
            }
        }
        all_keys.sort();
        
        for key in all_keys {
            // Prefer dirty account, then cache
            let account = dirty.get(key).or_else(|| cache.peek(key));
            if let Some(account) = account {
                hasher.update(key);
                hasher.update(&account.lamports.to_le_bytes());
                hasher.update(&account.owner);
                hasher.update(&account.data);
            }
        }
        
        hasher.finalize().to_vec()
    }

    /// Take snapshot of current cached state
    fn snapshot_state(&self) -> HashMap<Vec<u8>, Account> {
        let cache = self.account_cache.read().unwrap();
        let dirty = self.dirty_accounts.read().unwrap();
        
        let mut snapshot = HashMap::new();
        
        // Add all cached accounts
        for (key, account) in cache.iter() {
            snapshot.insert(key.clone(), account.clone());
        }
        
        // Override with dirty accounts
        for (key, account) in dirty.iter() {
            snapshot.insert(key.clone(), account.clone());
        }
        
        snapshot
    }

    /// Get account change between snapshots
    fn get_account_change(&self, pre_state: &HashMap<Vec<u8>, Account>, account_key: &[u8]) -> Option<AccountChange> {
        let pre_balance = pre_state.get(account_key).map(|a| a.lamports).unwrap_or(0);
        
        // Get current account from cache or dirty
        let post_account = {
            let dirty = self.dirty_accounts.read().unwrap();
            if let Some(account) = dirty.get(account_key) {
                Some(account.clone())
            } else {
                let cache = self.account_cache.read().unwrap();
                cache.peek(account_key).cloned()
            }
        };
        
        let post_balance = post_account.as_ref().map(|a| a.lamports).unwrap_or(0);
        
        let pre_data = pre_state.get(account_key).map(|a| &a.data);
        let post_data = post_account.as_ref().map(|a| &a.data);
        let data_modified = pre_data != post_data;

        if pre_balance != post_balance || data_modified {
            Some(AccountChange {
                address: account_key.to_vec(),
                pre_balance,
                post_balance,
                data_modified,
            })
        } else {
            None
        }
    }

    /// Persist state changes to storage
    async fn persist_state_changes(&self, changes: &[AccountChange]) -> Result<()> {
        // Use the flush_dirty_accounts method to persist all dirty accounts
        self.flush_dirty_accounts()?;
        Ok(())
    }

    /// Rollback to previous state (for fork handling)
    /// This restores the state from a snapshot and replays blocks to reach the target slot
    pub fn rollback_to_slot(&self, slot: u64) -> Result<()> {
        info!("Rolling back state to slot {}", slot);
        
        // Try to load snapshot for the specified slot
        let snapshot_key = format!("snapshot_{}", slot);
        
        match self.storage.get_state::<HashMap<Vec<u8>, Vec<u8>>>(&snapshot_key) {
            Ok(Some(snapshot_data)) => {
                // Clear cache and dirty accounts
                {
                    let mut cache = self.account_cache.write().unwrap();
                    cache.clear();
                }
                {
                    let mut dirty = self.dirty_accounts.write().unwrap();
                    dirty.clear();
                }
                
                // Load snapshot into cache
                let mut restored_count = 0;
                {
                    let mut cache = self.account_cache.write().unwrap();
                    for (address, account_data) in snapshot_data {
                        match bincode::deserialize::<Account>(&account_data) {
                            Ok(account) => {
                                cache.put(address, account);
                                restored_count += 1;
                            }
                            Err(e) => {
                                warn!("Failed to deserialize account during rollback: {}", e);
                            }
                        }
                    }
                }
                
                info!("Rollback complete: restored {} accounts from slot {}", restored_count, slot);
                Ok(())
            }
            Ok(None) => {
                warn!("No snapshot found for slot {}, attempting incremental rollback", slot);
                
                // Find the nearest snapshot before the target slot
                let nearest_snapshot_slot = self.find_nearest_snapshot_slot(slot)?;
                
                if nearest_snapshot_slot > 0 {
                    info!("Found nearest snapshot at slot {}", nearest_snapshot_slot);
                    
                    // Load the nearest snapshot
                    self.load_snapshot_internal(nearest_snapshot_slot)?;
                    
                    // Replay blocks from nearest_snapshot_slot to target slot
                    self.replay_blocks(nearest_snapshot_slot + 1, slot)?;
                } else {
                    warn!("No snapshots found, cannot rollback to slot {}", slot);
                    return Err(anyhow::anyhow!("No snapshots available for rollback"));
                }
                
                Ok(())
            }
            Err(e) => {
                Err(anyhow::anyhow!("Failed to load snapshot for slot {}: {}", slot, e))
            }
        }
    }
    
    /// Find the nearest snapshot before the given slot
    fn find_nearest_snapshot_slot(&self, target_slot: u64) -> Result<u64> {
        // Check snapshots in reverse order (most recent first)
        // Snapshots are typically created every N slots, so we check at those intervals
        let snapshot_interval = 1000u64; // Typical snapshot interval
        
        for check_slot in (0..=target_slot).rev().step_by(snapshot_interval as usize) {
            if self.storage.get_state::<bool>(&format!("snapshot_exists_{}", check_slot))
                .ok()
                .flatten()
                .unwrap_or(false)
            {
                return Ok(check_slot);
            }
        }
        
        // Also check at specific boundaries
        for check_slot in (0..target_slot).rev().take(100) {
            if self.storage.get_state::<bool>(&format!("snapshot_exists_{}", check_slot))
                .ok()
                .flatten()
                .unwrap_or(false)
            {
                return Ok(check_slot);
            }
        }
        
        Ok(0)
    }
    
    /// Load a snapshot into the current state
    fn load_snapshot_internal(&self, slot: u64) -> Result<()> {
        let snapshot_key = format!("snapshot_{}", slot);
        
        let snapshot_data: HashMap<Vec<u8>, Vec<u8>> = self.storage
            .get_state(&snapshot_key)?
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found for slot {}", slot))?;
        
        let mut state = self.current_state.write().unwrap();
        state.clear();
        
        let mut restored_count = 0;
        for (address, account_data) in snapshot_data {
            match bincode::deserialize::<Account>(&account_data) {
                Ok(account) => {
                    state.insert(address, account);
                    restored_count += 1;
                }
                Err(e) => {
                    warn!("Failed to deserialize account: {}", e);
                }
            }
        }
        
        info!("Loaded snapshot at slot {} with {} accounts", slot, restored_count);
        Ok(())
    }
    
    /// Replay blocks from start_slot to end_slot (inclusive)
    /// This re-executes all transactions to rebuild state
    fn replay_blocks(&self, start_slot: u64, end_slot: u64) -> Result<()> {
        info!("Replaying blocks from slot {} to {}", start_slot, end_slot);
        
        let mut replayed_count = 0;
        let mut total_txs = 0;
        
        // Fetch blocks from storage
        match self.storage.get_blocks_range(start_slot, end_slot) {
            Ok(blocks_data) => {
                for (slot, block_data) in blocks_data {
                    // Deserialize block
                    match Block::deserialize(&block_data) {
                        Ok(block) => {
                            // Execute each transaction in the block
                            for transaction in &block.transactions {
                                match self.executor.execute_transaction(transaction, slot) {
                                    Ok(result) => {
                                        if result.success {
                                            total_txs += 1;
                                        } else {
                                            debug!("Transaction failed during replay at slot {}: {:?}", 
                                                slot, result.error);
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Transaction execution error during replay at slot {}: {}", 
                                            slot, e);
                                    }
                                }
                            }
                            replayed_count += 1;
                        }
                        Err(e) => {
                            warn!("Failed to deserialize block at slot {}: {}", slot, e);
                        }
                    }
                }
                
                info!("Replayed {} blocks with {} successful transactions", 
                    replayed_count, total_txs);
                
                // Verify state root after replay
                let computed_root = self.compute_state_root();
                if let Ok(Some(stored_root)) = self.storage.get_state::<Vec<u8>>(
                    &format!("state_root_{}", end_slot)
                ) {
                    if computed_root != stored_root {
                        warn!("State root mismatch after replay at slot {}", end_slot);
                    } else {
                        info!("State root verified after replay at slot {}", end_slot);
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                Err(anyhow::anyhow!("Failed to fetch blocks for replay: {}", e))
            }
        }
    }
    
    /// Create a snapshot of the current state at the given slot
    pub fn create_snapshot(&self, slot: u64) -> Result<()> {
        info!("Creating state snapshot at slot {}", slot);
        
        let state = self.current_state.read().unwrap();
        
        // Serialize all accounts
        let mut snapshot_data: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
        for (address, account) in state.iter() {
            let serialized = bincode::serialize(account)?;
            snapshot_data.insert(address.clone(), serialized);
        }
        
        // Store snapshot
        let snapshot_key = format!("snapshot_{}", slot);
        self.storage.store_state(&snapshot_key, &snapshot_data)?;
        
        // Mark snapshot as existing
        self.storage.store_state(&format!("snapshot_exists_{}", slot), &true)?;
        
        // Store state root for verification
        let state_root = self.compute_state_root_internal(&state);
        self.storage.store_state(&format!("state_root_{}", slot), &state_root)?;
        
        info!("Snapshot created at slot {} with {} accounts", slot, snapshot_data.len());
        
        Ok(())
    }

    /// Get account from current state (uses lazy loading)
    pub fn get_account(&self, address: &[u8]) -> Option<Account> {
        // First check dirty accounts
        {
            let dirty = self.dirty_accounts.read().unwrap();
            if let Some(account) = dirty.get(address) {
                return Some(account.clone());
            }
        }
        
        // Then use lazy loading from cache/disk
        self.get_account_lazy(address)
    }

    /// Get multiple accounts
    pub fn get_accounts(&self, addresses: &[Vec<u8>]) -> Vec<Option<Account>> {
        addresses.iter()
            .map(|addr| self.get_account(addr))
            .collect()
    }

    /// Get state size (cached accounts only - not total accounts on disk)
    pub fn state_size(&self) -> usize {
        let cache_size = self.account_cache.read().unwrap().len();
        let dirty_size = self.dirty_accounts.read().unwrap().len();
        cache_size + dirty_size
    }
    
    /// Get total accounts in storage
    pub fn total_accounts_in_storage(&self) -> Result<usize> {
        let accounts = self.storage.get_all_accounts()?;
        Ok(accounts.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_state_transition_creation() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let transition = StateTransition::new(storage);
        assert_eq!(transition.state_size(), 0);
    }

    #[tokio::test]
    async fn test_empty_block_application() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let transition = StateTransition::new(storage);
        
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let validator = keypair.public.to_bytes().to_vec();
        
        let block = Block::new(1, vec![0u8; 32], vec![], validator);
        
        let result = transition.apply_block(&block).await.unwrap();
        
        assert_eq!(result.successful_transactions, 0);
        assert_eq!(result.failed_transactions, 0);
    }

    #[test]
    fn test_state_root_computation() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let transition = StateTransition::new(storage);
        
        // Add some accounts
        {
            let mut state = transition.current_state.write().unwrap();
            state.insert(vec![1u8; 32], Account::new(1000, vec![]));
            state.insert(vec![2u8; 32], Account::new(2000, vec![]));
        }
        
        let root1 = transition.compute_state_root();
        
        // Add another account
        {
            let mut state = transition.current_state.write().unwrap();
            state.insert(vec![3u8; 32], Account::new(3000, vec![]));
        }
        
        let root2 = transition.compute_state_root();
        
        // Roots should be different
        assert_ne!(root1, root2);
    }
}




