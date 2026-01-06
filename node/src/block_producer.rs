use anyhow::{Result, Context};
use std::sync::Arc;
use tracing::{info, warn, debug};
use ed25519_dalek::Keypair;

use ade_consensus::{Block, BlockBuilder};
use ade_transaction::{Transaction, TransactionExecutor};
use crate::mempool::{Mempool, MempoolTransaction};
use crate::fee_market::FeeMarket;
use crate::storage::Storage;

/// Block producer for validators with intelligent transaction selection
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
    pub min_fee_multiplier: f64,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            max_transactions_per_block: 10_000,
            max_block_compute: 48_000_000,
            max_block_size: 1_300_000,
            block_time_ms: 400,
            min_fee_multiplier: 1.0,
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

    pub async fn produce_block(&self, slot: u64, parent_hash: Vec<u8>) -> Result<BlockProductionResult> {
        let start_time = std::time::Instant::now();
        
        info!("Producing block for slot {}", slot);

        // 1. Get base fee from fee market
        let base_fee = self.fee_market.get_base_fee();
        let min_acceptable_fee = (base_fee as f64 * self.config.min_fee_multiplier) as u64;

        // 2. Select transactions with fee filtering
        let selected_txs = self.select_transactions_smart(min_acceptable_fee).await?;
        debug!("Selected {} transactions from mempool", selected_txs.len());

        // 3. Pack transactions with validation
        let (packed_txs, total_fees, total_compute, rejected) = 
            self.pack_transactions_optimized(selected_txs).await?;
        debug!("Packed {} transactions, rejected {}, total compute: {}", 
            packed_txs.len(), rejected, total_compute);

        // 4. Build block
        let validator_pubkey = self.keypair.public.to_bytes().to_vec();
        let mut block = BlockBuilder::new(slot, parent_hash, validator_pubkey)
            .add_transactions(packed_txs.clone())
            .build();

        // 5. Sign block
        block.sign(&self.keypair)?;
        debug!("Block signed");

        // 6. Remove included transactions from mempool
        let mut removed_count = 0;
        for tx in &packed_txs {
            if let Some(sig) = tx.signature() {
                if self.mempool.remove_transaction(sig).is_some() {
                    removed_count += 1;
                }
            }
        }
        debug!("Removed {} transactions from mempool", removed_count);

        // 7. Extract priority fees from transactions
        let priority_fees = self.extract_priority_fees(&packed_txs);

        // 8. Record block data in fee market
        self.fee_market.record_block(
            slot,
            packed_txs.len(),
            self.config.max_transactions_per_block,
            priority_fees,
        );

        let execution_time = start_time.elapsed().as_millis() as u64;

        info!("Block produced for slot {} with {} transactions in {}ms (total fees: {}, compute: {})", 
            slot, packed_txs.len(), execution_time, total_fees, total_compute);

        Ok(BlockProductionResult {
            block,
            transaction_count: packed_txs.len(),
            total_fees,
            total_compute,
            execution_time_ms: execution_time,
            rejected_count: rejected,
        })
    }

    /// Smart transaction selection with fee-based filtering
    async fn select_transactions_smart(&self, min_fee: u64) -> Result<Vec<MempoolTransaction>> {
        let mut candidates = self.mempool.get_top_transactions(self.config.max_transactions_per_block * 2);
        
        // Filter by minimum fee
        candidates.retain(|tx| {
            let total_fee = tx.fee + tx.priority_fee;
            total_fee >= min_fee
        });
        
        debug!("After fee filtering: {} candidates (min fee: {})", candidates.len(), min_fee);
        Ok(candidates)
    }

    /// Optimized transaction packing with compute tracking
    async fn pack_transactions_optimized(
        &self,
        candidates: Vec<MempoolTransaction>,
    ) -> Result<(Vec<Transaction>, u64, u64, usize)> {
        let mut packed = Vec::new();
        let mut total_fees = 0u64;
        let mut total_compute = 0u64;
        let mut total_size = 0usize;
        let mut rejected = 0usize;

        for candidate in candidates {
            // Check compute budget
            if total_compute + candidate.compute_budget > self.config.max_block_compute {
                debug!("Skipping tx - compute budget exceeded (current: {}, adding: {}, max: {})",
                    total_compute, candidate.compute_budget, self.config.max_block_compute);
                rejected += 1;
                continue;
            }

            // Check size
            if total_size + candidate.size > self.config.max_block_size {
                debug!("Skipping tx - block size limit (current: {}, adding: {}, max: {})",
                    total_size, candidate.size, self.config.max_block_size);
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
                    let tx_fee = candidate.fee + candidate.priority_fee;
                    
                    packed.push(candidate.transaction.clone());
                    total_fees += tx_fee;
                    total_compute += candidate.compute_budget;
                    total_size += candidate.size;
                    
                    debug!("Packed tx with fee: {}, compute: {}", tx_fee, candidate.compute_budget);
                }
                _ => {
                    warn!("Invalid transaction, skipping");
                    rejected += 1;
                }
            }
        }

        info!("Packing complete: {} txs, {} total fees, {} compute, {} bytes",
            packed.len(), total_fees, total_compute, total_size);

        Ok((packed, total_fees, total_compute, rejected))
    }

    /// Extract priority fees from transactions
    /// 
    /// Parses ComputeBudget program instructions to extract priority fees.
    /// 
    /// ComputeBudget instruction discriminators:
    /// - 0: RequestUnitsDeprecated (legacy)
    /// - 1: RequestHeapFrame
    /// - 2: SetComputeUnitLimit (4 bytes)
    /// - 3: SetComputeUnitPrice (8 bytes) - this is what we want
    fn extract_priority_fees(&self, transactions: &[Transaction]) -> Vec<u64> {
        // ComputeBudget program ID (11111111111111111111111111111112)
        const COMPUTE_BUDGET_PROGRAM_ID: [u8; 32] = [
            3, 6, 70, 111, 229, 33, 23, 50, 255, 236, 173, 186, 114, 195, 155, 231,
            188, 140, 229, 187, 197, 247, 18, 107, 44, 67, 155, 58, 64, 0, 0, 0
        ];
        
        // SetComputeUnitPrice discriminator
        const SET_COMPUTE_UNIT_PRICE: u8 = 3;
        
        transactions.iter()
            .map(|tx| {
                let mut priority_fee = 0u64;
                let mut compute_units = 200_000u32; // Default compute units
                
                for instruction in &tx.message.instructions {
                    // Check if this is a ComputeBudget instruction
                    let is_compute_budget = instruction.program_id.len() == 32 
                        && instruction.program_id == COMPUTE_BUDGET_PROGRAM_ID;
                    
                    if !is_compute_budget {
                        continue;
                    }
                    
                    if instruction.data.is_empty() {
                        continue;
                    }
                    
                    match instruction.data[0] {
                        // SetComputeUnitPrice (discriminator: 3, data: 8 bytes u64)
                        3 if instruction.data.len() >= 9 => {
                            if let Ok(bytes) = instruction.data[1..9].try_into() {
                                priority_fee = u64::from_le_bytes(bytes);
                            }
                        }
                        // SetComputeUnitLimit (discriminator: 2, data: 4 bytes u32)
                        2 if instruction.data.len() >= 5 => {
                            if let Ok(bytes) = instruction.data[1..5].try_into() {
                                compute_units = u32::from_le_bytes(bytes);
                            }
                        }
                        // RequestUnitsDeprecated (discriminator: 0, legacy format)
                        0 if instruction.data.len() >= 9 => {
                            // Legacy: units (4 bytes) + additional_fee (4 bytes)
                            if let Ok(units_bytes) = instruction.data[1..5].try_into() {
                                compute_units = u32::from_le_bytes(units_bytes);
                            }
                            if let Ok(fee_bytes) = instruction.data[5..9].try_into() {
                                // Legacy fee format - convert to micro-lamports per CU
                                let legacy_fee = u32::from_le_bytes(fee_bytes) as u64;
                                priority_fee = legacy_fee * 1_000_000 / compute_units.max(1) as u64;
                            }
                        }
                        _ => {}
                    }
                }
                
                // Calculate total priority fee: micro-lamports per CU * CUs / 1M
                // priority_fee is in micro-lamports per compute unit
                (priority_fee * compute_units as u64) / 1_000_000
            })
            .collect()
    }

    /// Get producer statistics
    pub fn get_stats(&self) -> ProducerStats {
        ProducerStats {
            mempool_size: self.mempool.size(),
            base_fee: self.fee_market.get_base_fee(),
            average_block_time: 400, // Would track actual
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProducerStats {
    pub mempool_size: usize,
    pub base_fee: u64,
    pub average_block_time: u64,
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
            mempool.clone(),
            fee_market,
            storage,
            ProducerConfig::default(),
        );
        
        let result = producer.produce_block(1, vec![0u8; 32]).await.unwrap();
        
        assert_eq!(result.transaction_count, 0);
        assert_eq!(mempool.size(), 0);
    }

    #[tokio::test]
    async fn test_priority_fee_extraction() {
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
        
        // Test priority fee extraction
        let tx = Transaction::new(&[&Keypair::generate(&mut csprng)], vec![], vec![1u8; 32]).unwrap();
        let fees = producer.extract_priority_fees(&[tx]);
        
        assert!(!fees.is_empty());
    }
}
