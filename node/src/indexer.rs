use anyhow::{Result, Context};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

use crate::storage::Storage;

/// Secondary index for efficient queries
pub struct SecondaryIndex {
    storage: Arc<Storage>,
    address_to_transactions: Arc<RwLock<BTreeMap<Vec<u8>, Vec<Vec<u8>>>>>,
    program_to_accounts: Arc<RwLock<HashMap<Vec<u8>, Vec<Vec<u8>>>>>,
    slot_to_transactions: Arc<RwLock<BTreeMap<u64, Vec<Vec<u8>>>>>,
}

impl SecondaryIndex {
    pub fn new(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            address_to_transactions: Arc::new(RwLock::new(BTreeMap::new())),
            program_to_accounts: Arc::new(RwLock::new(HashMap::new())),
            slot_to_transactions: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Index a transaction by address
    pub fn index_transaction_by_address(
        &self,
        address: &[u8],
        signature: &[u8],
    ) -> Result<()> {
        let mut index = self.address_to_transactions.write().unwrap();
        index.entry(address.to_vec())
            .or_insert_with(Vec::new)
            .push(signature.to_vec());
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

    /// Rebuild index from storage
    pub fn rebuild_from_storage(&self) -> Result<()> {
        // In production, this would scan the entire database
        // and rebuild all indexes
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

