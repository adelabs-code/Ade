use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};
use ed25519_dalek::Keypair;

use ade_consensus::{Block, BlockBuilder};
use ade_transaction::{Transaction, TransactionExecutor};
use crate::mempool::{Mempool, MempoolTransaction};
use crate::fee_market::FeeMarket;
use crate::storage::Storage;

/// Block producer for validators
pub struct BlockProducer {
    keypair: Keypair,
    mempool: Arc<Mempool>,
    fee_market: Arc<FeeMarket>,
    storage: Arc<Storage>,
    config: ProducerConfig,
}

#[derive(Debug, Clone)]
pub struct ProducerConfig {
    pub max_transactions_per_block: usize,
    pub max_block_compute: u64,
    pub max_block_size: usize,
    pub block_time_ms: u64,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            max_transactions_per_block: 10_000,
            max_block_compute: 48_000_000,
            max_block_size: 1_300_000,
            block_time_ms: 400,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockProductionResult {
    pub block: Block,
    pub transaction_count: usize,
    pub total_fees: u64,
    pub total_compute: u64,
    pub execution_time_ms: u64,
    pub rejected_count: usize,
}

impl BlockProducer {
    pub fn new(
        keypair: Keypair,
        mempool: Arc<Mempool>,
        fee_market: Arc<FeeMarket>,
        storage: Arc<Storage>,
        config: ProducerConfig,
    ) -> Self {
        Self {
            keypair,
            mempool,
            fee_market,
            storage,
            config,
        }
    }

    /// Produce a block for given slot
    pub async fn produce_block(&self, slot: u64, parent_hash: Vec<u8>) -> Result<BlockProductionResult> {
        let start_time = std::time::Instant::now();
        
        info!("Producing block for slot {}", slot);

        // 1. Select transactions from mempool
        let selected_txs = self.select_transactions().await?;
        debug!("Selected {} transactions from mempool", selected_txs.len());

        // 2. Pack transactions into block
        let (packed_txs, total_fees, rejected) = self.pack_transactions(selected_txs).await?;
        debug!("Packed {} transactions, rejected {}", packed_txs.len(), rejected);

        // 3. Build block
        let validator_pubkey = self.keypair.public.to_bytes().to_vec();
        let mut block = BlockBuilder::new(slot, parent_hash, validator_pubkey)
            .add_transactions(packed_txs.clone())
            .build();

        // 4. Sign block
        block.sign(&self.keypair)?;
        debug!("Block signed");

        // 5. Remove included transactions from mempool
        for tx in &packed_txs {
            if let Some(sig) = tx.signature() {
                self.mempool.remove_transaction(sig);
            }
        }

        // 6. Record fees for fee market
        let priority_fees: Vec<u64> = packed_txs.iter()
            .filter_map(|tx| {
                // Extract priority fee from transaction
                Some(0u64) // Placeholder
            })
            .collect();

        self.fee_market.record_block(
            slot,
            packed_txs.len(),
            self.config.max_transactions_per_block,
            priority_fees,
        );

        let execution_time = start_time.elapsed().as_millis() as u64;

        info!("Block produced for slot {} with {} transactions in {}ms", 
            slot, packed_txs.len(), execution_time);

        Ok(BlockProductionResult {
            block,
            transaction_count: packed_txs.len(),
            total_fees,
            total_compute: 0, // Would be calculated during execution
            execution_time_ms: execution_time,
            rejected_count: rejected,
        })
    }

    /// Select transactions from mempool
    async fn select_transactions(&self) -> Result<Vec<MempoolTransaction>> {
        // Get highest priority transactions
        let candidates = self.mempool.get_top_transactions(self.config.max_transactions_per_block * 2);
        
        debug!("Considering {} candidate transactions", candidates.len());
        Ok(candidates)
    }

    /// Pack transactions into block with validation
    async fn pack_transactions(
        &self,
        candidates: Vec<MempoolTransaction>,
    ) -> Result<(Vec<Transaction>, u64, usize)> {
        let mut packed = Vec::new();
        let mut total_fees = 0u64;
        let mut total_compute = 0u64;
        let mut total_size = 0usize;
        let mut rejected = 0usize;

        for candidate in candidates {
            // Check compute budget
            if total_compute + candidate.compute_budget > self.config.max_block_compute {
                debug!("Skipping tx - compute budget exceeded");
                rejected += 1;
                continue;
            }

            // Check size
            if total_size + candidate.size > self.config.max_block_size {
                debug!("Skipping tx - block size limit");
                rejected += 1;
                continue;
            }

            // Check transaction count
            if packed.len() >= self.config.max_transactions_per_block {
                debug!("Block full - max transactions reached");
                break;
            }

            // Verify transaction is still valid
            match candidate.transaction.verify() {
                Ok(true) => {
                    packed.push(candidate.transaction.clone());
                    total_fees += candidate.fee + candidate.priority_fee;
                    total_compute += candidate.compute_budget;
                    total_size += candidate.size;
                }
                _ => {
                    warn!("Invalid transaction, skipping");
                    rejected += 1;
                }
            }
        }

        Ok((packed, total_fees, rejected))
    }

    /// Get producer statistics
    pub fn get_stats(&self) -> ProducerStats {
        ProducerStats {
            mempool_size: self.mempool.size(),
            base_fee: self.fee_market.get_base_fee(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProducerStats {
    pub mempool_size: usize,
    pub base_fee: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_block_producer_creation() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let mempool = Arc::new(Mempool::new(Default::default()));
        let fee_market = Arc::new(FeeMarket::new(Default::default()));
        
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let producer = BlockProducer::new(
            keypair,
            mempool,
            fee_market,
            storage,
            ProducerConfig::default(),
        );
        
        let stats = producer.get_stats();
        assert_eq!(stats.mempool_size, 0);
    }

    #[tokio::test]
    async fn test_empty_block_production() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let mempool = Arc::new(Mempool::new(Default::default()));
        let fee_market = Arc::new(FeeMarket::new(Default::default()));
        
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let producer = BlockProducer::new(
            keypair,
            mempool,
            fee_market,
            storage,
            ProducerConfig::default(),
        );
        
        let result = producer.produce_block(1, vec![0u8; 32]).await.unwrap();
        
        assert_eq!(result.transaction_count, 0);
        assert_eq!(result.block.header.slot, 1);
    }
}




