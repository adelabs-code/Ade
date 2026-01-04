use anyhow::{Result, Context};
use rocksdb::{DB, Options, ColumnFamilyDescriptor, WriteBatch, IteratorMode, Direction};
use std::path::Path;
use serde::{Serialize, Deserialize};
use tracing::{info, warn};

/// Persistent storage using RocksDB
///
/// Provides key-value storage for blocks, transactions, accounts, and state data.
/// Uses column families for organized data separation.
pub struct Storage {
    db: DB,
}

const CF_BLOCKS: &str = "blocks";
const CF_TRANSACTIONS: &str = "transactions";
const CF_ACCOUNTS: &str = "accounts";
const CF_STATE: &str = "state";
const CF_SLOTS: &str = "slots";
const CF_SIGNATURES: &str = "signatures";
const CF_PROGRAM_ACCOUNTS: &str = "program_accounts";

impl Storage {
    /// Create new storage instance
    ///
    /// # Arguments
    /// * `path` - Directory path for database files
    pub fn new(path: &str) -> Result<Self> {
        let path = Path::new(path);
        std::fs::create_dir_all(path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_max_open_files(10000);
        opts.set_keep_log_file_num(10);
        opts.set_max_background_jobs(6);
        opts.set_bytes_per_sync(1048576);

        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_BLOCKS, Options::default()),
            ColumnFamilyDescriptor::new(CF_TRANSACTIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_ACCOUNTS, Options::default()),
            ColumnFamilyDescriptor::new(CF_STATE, Options::default()),
            ColumnFamilyDescriptor::new(CF_SLOTS, Options::default()),
            ColumnFamilyDescriptor::new(CF_SIGNATURES, Options::default()),
            ColumnFamilyDescriptor::new(CF_PROGRAM_ACCOUNTS, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)
            .context("Failed to open RocksDB")?;

        info!("Storage initialized at {:?}", path);
        Ok(Self { db })
    }

    // Block operations

    /// Store a block at given slot
    pub fn store_block(&self, slot: u64, block_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        self.db.put_cf(cf, slot.to_le_bytes(), block_data)?;
        
        let slots_cf = self.db.cf_handle(CF_SLOTS)
            .context("Slots CF not found")?;
        self.db.put_cf(slots_cf, slot.to_le_bytes(), &[1u8])?;
        
        Ok(())
    }

    /// Get block by slot
    pub fn get_block(&self, slot: u64) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        Ok(self.db.get_cf(cf, slot.to_le_bytes())?)
    }

    /// Get range of blocks
    pub fn get_blocks_range(&self, start_slot: u64, end_slot: u64) -> Result<Vec<(u64, Vec<u8>)>> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        
        let mut blocks = Vec::new();
        let iter = self.db.iterator_cf(cf, IteratorMode::From(&start_slot.to_le_bytes(), Direction::Forward));
        
        for item in iter {
            let (key, value) = item?;
            let slot = u64::from_le_bytes(key.as_ref().try_into().unwrap());
            
            if slot > end_slot {
                break;
            }
            
            blocks.push((slot, value.to_vec()));
        }
        
        Ok(blocks)
    }

    /// Get latest block
    pub fn get_latest_block(&self) -> Result<Option<(u64, Vec<u8>)>> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        
        let mut iter = self.db.iterator_cf(cf, IteratorMode::End);
        
        if let Some(item) = iter.next() {
            let (key, value) = item?;
            let slot = u64::from_le_bytes(key.as_ref().try_into().unwrap());
            return Ok(Some((slot, value.to_vec())));
        }
        
        Ok(None)
    }

    /// Delete block at slot
    pub fn delete_block(&self, slot: u64) -> Result<()> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        self.db.delete_cf(cf, slot.to_le_bytes())?;
        Ok(())
    }

    // Transaction operations

    /// Store transaction
    pub fn store_transaction(&self, signature: &[u8], tx_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Transactions CF not found")?;
        self.db.put_cf(cf, signature, tx_data)?;
        
        let sig_cf = self.db.cf_handle(CF_SIGNATURES)
            .context("Signatures CF not found")?;
        self.db.put_cf(sig_cf, signature, &crate::utils::current_timestamp().to_le_bytes())?;
        
        Ok(())
    }

    /// Get transaction by signature
    pub fn get_transaction(&self, signature: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Transactions CF not found")?;
        Ok(self.db.get_cf(cf, signature)?)
    }

    /// Store transaction with slot information
    pub fn store_transaction_with_slot(
        &self,
        signature: &[u8],
        slot: u64,
        tx_data: &[u8],
    ) -> Result<()> {
        let mut batch = WriteBatch::default();
        
        let tx_cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Transactions CF not found")?;
        batch.put_cf(tx_cf, signature, tx_data);
        
        let sig_cf = self.db.cf_handle(CF_SIGNATURES)
            .context("Signatures CF not found")?;
        batch.put_cf(sig_cf, signature, &slot.to_le_bytes());
        
        self.db.write(batch)?;
        Ok(())
    }

    // Account operations

    /// Store account data
    pub fn store_account(&self, address: &[u8], account_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        self.db.put_cf(cf, address, account_data)?;
        Ok(())
    }

    /// Get account data
    pub fn get_account(&self, address: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        Ok(self.db.get_cf(cf, address)?)
    }

    /// Store multiple accounts in batch
    pub fn store_accounts_batch(&self, accounts: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let mut batch = WriteBatch::default();
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        
        for (address, data) in accounts {
            batch.put_cf(cf, address, data);
        }
        
        self.db.write(batch)?;
        Ok(())
    }

    /// Delete account
    pub fn delete_account(&self, address: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        self.db.delete_cf(cf, address)?;
        Ok(())
    }

    /// Get accounts by program
    pub fn get_accounts_by_program(&self, program_id: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.db.cf_handle(CF_PROGRAM_ACCOUNTS)
            .context("Program accounts CF not found")?;
        
        let mut accounts = Vec::new();
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);
        
        for item in iter {
            let (key, value) = item?;
            if key.starts_with(program_id) {
                accounts.push((key[32..].to_vec(), value.to_vec()));
            }
        }
        
        Ok(accounts)
    }

    /// Index program account
    pub fn index_program_account(&self, program_id: &[u8], account: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_PROGRAM_ACCOUNTS)
            .context("Program accounts CF not found")?;
        
        let mut key = Vec::new();
        key.extend_from_slice(program_id);
        key.extend_from_slice(account);
        
        self.db.put_cf(cf, key, &[1u8])?;
        Ok(())
    }

    // State operations

    /// Store typed state value
    pub fn store_state<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let cf = self.db.cf_handle(CF_STATE)
            .context("State CF not found")?;
        let serialized = bincode::serialize(value)?;
        self.db.put_cf(cf, key.as_bytes(), serialized)?;
        Ok(())
    }

    /// Get typed state value
    pub fn get_state<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let cf = self.db.cf_handle(CF_STATE)
            .context("State CF not found")?;
        match self.db.get_cf(cf, key.as_bytes())? {
            Some(data) => Ok(Some(bincode::deserialize(&data)?)),
            None => Ok(None),
        }
    }

    /// Delete state value
    pub fn delete_state(&self, key: &str) -> Result<()> {
        let cf = self.db.cf_handle(CF_STATE)
            .context("State CF not found")?;
        self.db.delete_cf(cf, key.as_bytes())?;
        Ok(())
    }

    // Utility operations

    /// Compact database
    pub fn compact(&self) -> Result<()> {
        info!("Compacting database...");
        
        for cf_name in &[CF_BLOCKS, CF_TRANSACTIONS, CF_ACCOUNTS, CF_STATE, CF_SLOTS, CF_SIGNATURES, CF_PROGRAM_ACCOUNTS] {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                self.db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
            }
        }
        
        info!("Database compaction complete");
        Ok(())
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> Result<StorageStats> {
        let mut total_size = 0u64;
        let mut block_count = 0usize;
        let mut tx_count = 0usize;
        let mut account_count = 0usize;

        if let Some(cf) = self.db.cf_handle(CF_BLOCKS) {
            block_count = self.db.iterator_cf(cf, IteratorMode::Start).count();
        }

        if let Some(cf) = self.db.cf_handle(CF_TRANSACTIONS) {
            tx_count = self.db.iterator_cf(cf, IteratorMode::Start).count();
        }

        if let Some(cf) = self.db.cf_handle(CF_ACCOUNTS) {
            account_count = self.db.iterator_cf(cf, IteratorMode::Start).count();
        }

        Ok(StorageStats {
            total_size,
            block_count,
            transaction_count: tx_count,
            account_count,
        })
    }

    /// Create backup (placeholder)
    pub fn backup(&self, backup_path: &str) -> Result<()> {
        info!("Creating backup at {}", backup_path);
        Ok(())
    }

    /// Restore from backup (placeholder)
    pub fn restore_from_backup(&self, backup_path: &str) -> Result<()> {
        info!("Restoring from backup at {}", backup_path);
        Ok(())
    }

    /// Prune old blocks before slot
    pub fn prune_blocks_before(&self, slot: u64) -> Result<usize> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        
        let mut pruned = 0;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);
        
        for item in iter {
            let (key, _) = item?;
            let block_slot = u64::from_le_bytes(key.as_ref().try_into().unwrap());
            
            if block_slot >= slot {
                break;
            }
            
            self.db.delete_cf(cf, key)?;
            pruned += 1;
        }
        
        info!("Pruned {} blocks before slot {}", pruned, slot);
        Ok(pruned)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_size: u64,
    pub block_count: usize,
    pub transaction_count: usize,
    pub account_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_storage_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Storage::new(temp_dir.path().to_str().unwrap());
        assert!(storage.is_ok());
    }

    #[test]
    fn test_block_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Storage::new(temp_dir.path().to_str().unwrap()).unwrap();
        
        let block_data = vec![1, 2, 3, 4];
        storage.store_block(100, &block_data).unwrap();
        
        let retrieved = storage.get_block(100).unwrap();
        assert_eq!(retrieved, Some(block_data));
    }

    #[test]
    fn test_transaction_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Storage::new(temp_dir.path().to_str().unwrap()).unwrap();
        
        let signature = vec![5u8; 64];
        let tx_data = vec![1, 2, 3, 4];
        storage.store_transaction(&signature, &tx_data).unwrap();
        
        let retrieved = storage.get_transaction(&signature).unwrap();
        assert_eq!(retrieved, Some(tx_data));
    }
}
