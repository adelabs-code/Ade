use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, debug, warn};

use ade_transaction::{Transaction, TransactionExecutor, Account, ExecutionResult};
use ade_consensus::Block;
use crate::storage::Storage;

/// State transition function
pub struct StateTransition {
    storage: Arc<Storage>,
    executor: Arc<TransactionExecutor>,
    current_state: Arc<RwLock<HashMap<Vec<u8>, Account>>>,
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
    pub fn new(storage: Arc<Storage>) -> Self {
        let accounts = Arc::new(RwLock::new(HashMap::new()));
        let executor = Arc::new(TransactionExecutor::new(accounts.clone()));
        
        Self {
            storage,
            executor,
            current_state: accounts,
        }
    }

    /// Load state from storage
    pub async fn load_state(&self, slot: u64) -> Result<()> {
        info!("Loading state for slot {}", slot);
        
        // In production, this would load from storage
        // For now, we maintain in-memory state
        
        Ok(())
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

    /// Compute state root hash
    fn compute_state_root(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        
        let state = self.current_state.read().unwrap();
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

    /// Take snapshot of current state
    fn snapshot_state(&self) -> HashMap<Vec<u8>, Account> {
        self.current_state.read().unwrap().clone()
    }

    /// Get account change between snapshots
    fn get_account_change(&self, pre_state: &HashMap<Vec<u8>, Account>, account_key: &[u8]) -> Option<AccountChange> {
        let current_state = self.current_state.read().unwrap();
        
        let pre_balance = pre_state.get(account_key).map(|a| a.lamports).unwrap_or(0);
        let post_balance = current_state.get(account_key).map(|a| a.lamports).unwrap_or(0);
        
        let pre_data = pre_state.get(account_key).map(|a| &a.data);
        let post_data = current_state.get(account_key).map(|a| &a.data);
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
        let state = self.current_state.read().unwrap();
        
        for change in changes {
            if let Some(account) = state.get(&change.address) {
                let serialized = bincode::serialize(account)?;
                self.storage.store_account(&change.address, &serialized)?;
            }
        }
        
        Ok(())
    }

    /// Rollback to previous state (for fork handling)
    pub fn rollback_to_slot(&self, slot: u64) -> Result<()> {
        info!("Rolling back state to slot {}", slot);
        
        // In production, this would restore from storage snapshot
        
        Ok(())
    }

    /// Get account from current state
    pub fn get_account(&self, address: &[u8]) -> Option<Account> {
        self.current_state.read().unwrap().get(address).cloned()
    }

    /// Get multiple accounts
    pub fn get_accounts(&self, addresses: &[Vec<u8>]) -> Vec<Option<Account>> {
        let state = self.current_state.read().unwrap();
        addresses.iter()
            .map(|addr| state.get(addr).cloned())
            .collect()
    }

    /// Get state size
    pub fn state_size(&self) -> usize {
        self.current_state.read().unwrap().len()
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

