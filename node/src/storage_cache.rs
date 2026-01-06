use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use lru::LruCache;
use std::num::NonZeroUsize;

/// Write-through cache for storage layer
pub struct StorageCache {
    block_cache: Arc<RwLock<LruCache<u64, Vec<u8>>>>,
    account_cache: Arc<RwLock<LruCache<Vec<u8>, Vec<u8>>>>,
    tx_cache: Arc<RwLock<LruCache<Vec<u8>, Vec<u8>>>>,
}

impl StorageCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap();
        
        Self {
            block_cache: Arc::new(RwLock::new(LruCache::new(cap))),
            account_cache: Arc::new(RwLock::new(LruCache::new(cap))),
            tx_cache: Arc::new(RwLock::new(LruCache::new(cap))),
        }
    }

    pub fn get_block(&self, slot: u64) -> Option<Vec<u8>> {
        self.block_cache.write().unwrap().get(&slot).cloned()
    }

    pub fn put_block(&self, slot: u64, data: Vec<u8>) {
        self.block_cache.write().unwrap().put(slot, data);
    }

    pub fn get_account(&self, address: &[u8]) -> Option<Vec<u8>> {
        self.account_cache.write().unwrap().get(address).cloned()
    }

    pub fn put_account(&self, address: Vec<u8>, data: Vec<u8>) {
        self.account_cache.write().unwrap().put(address, data);
    }

    pub fn clear(&self) {
        self.block_cache.write().unwrap().clear();
        self.account_cache.write().unwrap().clear();
        self.tx_cache.write().unwrap().clear();
    }
}



