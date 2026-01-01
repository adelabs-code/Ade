use anyhow::{Result, Context};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::path::Path;
use serde::{Serialize, Deserialize};

pub struct Storage {
    db: DB,
}

const CF_BLOCKS: &str = "blocks";
const CF_TRANSACTIONS: &str = "transactions";
const CF_ACCOUNTS: &str = "accounts";
const CF_STATE: &str = "state";

impl Storage {
    pub fn new(path: &str) -> Result<Self> {
        let path = Path::new(path);
        std::fs::create_dir_all(path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_BLOCKS, Options::default()),
            ColumnFamilyDescriptor::new(CF_TRANSACTIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_ACCOUNTS, Options::default()),
            ColumnFamilyDescriptor::new(CF_STATE, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)
            .context("Failed to open RocksDB")?;

        Ok(Self { db })
    }

    pub fn store_block(&self, slot: u64, block_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        self.db.put_cf(cf, slot.to_le_bytes(), block_data)?;
        Ok(())
    }

    pub fn get_block(&self, slot: u64) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_BLOCKS)
            .context("Blocks CF not found")?;
        Ok(self.db.get_cf(cf, slot.to_le_bytes())?)
    }

    pub fn store_transaction(&self, signature: &[u8], tx_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Transactions CF not found")?;
        self.db.put_cf(cf, signature, tx_data)?;
        Ok(())
    }

    pub fn get_transaction(&self, signature: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_TRANSACTIONS)
            .context("Transactions CF not found")?;
        Ok(self.db.get_cf(cf, signature)?)
    }

    pub fn store_account(&self, address: &[u8], account_data: &[u8]) -> Result<()> {
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        self.db.put_cf(cf, address, account_data)?;
        Ok(())
    }

    pub fn get_account(&self, address: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.db.cf_handle(CF_ACCOUNTS)
            .context("Accounts CF not found")?;
        Ok(self.db.get_cf(cf, address)?)
    }

    pub fn store_state<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let cf = self.db.cf_handle(CF_STATE)
            .context("State CF not found")?;
        let serialized = bincode::serialize(value)?;
        self.db.put_cf(cf, key.as_bytes(), serialized)?;
        Ok(())
    }

    pub fn get_state<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let cf = self.db.cf_handle(CF_STATE)
            .context("State CF not found")?;
        match self.db.get_cf(cf, key.as_bytes())? {
            Some(data) => Ok(Some(bincode::deserialize(&data)?)),
            None => Ok(None),
        }
    }
}


