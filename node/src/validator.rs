use anyhow::Result;
use ed25519_dalek::Keypair;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use crate::storage::Storage;
use crate::block_producer::{BlockProducer, ProducerConfig};
use crate::mempool::Mempool;
use crate::fee_market::FeeMarket;
use crate::state_transition::StateTransition;
use ade_consensus::{ProofOfStake, Block};

/// Validator with integrated block production
pub struct Validator {
    keypair: Keypair,
    storage: Arc<Storage>,
    block_producer: Option<Arc<BlockProducer>>,
    state_transition: Arc<StateTransition>,
    current_slot: Arc<RwLock<u64>>,
}

impl Validator {
    pub fn new(keypair: Keypair, storage: Arc<Storage>) -> Result<Self> {
        let state_transition = Arc::new(StateTransition::new(storage.clone()));
        
        Ok(Self {
            keypair,
            storage,
            block_producer: None,
            state_transition,
            current_slot: Arc::new(RwLock::new(0)),
        })
    }

    /// Initialize with block producer
    pub fn with_block_producer(
        mut self,
        mempool: Arc<Mempool>,
        fee_market: Arc<FeeMarket>,
        config: ProducerConfig,
    ) -> Self {
        let producer = Arc::new(BlockProducer::new(
            self.keypair.clone(),
            mempool,
            fee_market,
            self.storage.clone(),
            config,
        ));

        self.block_producer = Some(producer);
        self
    }

    /// Start validator
    pub async fn start(&self) -> Result<()> {
        info!("Starting validator with pubkey: {:?}", 
            bs58::encode(self.keypair.public.to_bytes()).into_string());

        if self.block_producer.is_none() {
            warn!("Validator started without block producer (read-only mode)");
            return Ok(());
        }

        // Main validator loop
        loop {
            let slot = self.get_next_slot().await;
            
            // Check if we are the leader
            if self.is_leader(slot).await {
                info!("We are leader for slot {}", slot);
                
                if let Err(e) = self.produce_and_process_block(slot).await {
                    warn!("Failed to produce block for slot {}: {}", slot, e);
                }
            }

            // Wait for next slot
            tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
        }
    }

    /// Produce and process a block
    async fn produce_and_process_block(&self, slot: u64) -> Result<()> {
        let producer = self.block_producer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Block producer not initialized"))?;

        // Get parent hash
        let parent_hash = self.get_parent_hash(slot).await?;

        // Produce block
        let production_result = producer.produce_block(slot, parent_hash).await?;
        
        info!("Produced block for slot {}: {} transactions, {} fees",
            slot,
            production_result.transaction_count,
            production_result.total_fees
        );

        // Apply state transition
        let transition_result = self.state_transition.apply_block(&production_result.block).await?;
        
        debug!("State transition: {} successful, {} failed",
            transition_result.successful_transactions,
            transition_result.failed_transactions
        );

        // Store block
        let block_data = production_result.block.serialize()?;
        self.storage.store_block(slot, &block_data)?;

        // Store transactions
        for tx in &production_result.block.transactions {
            if let Some(sig) = tx.signature() {
                let tx_data = tx.serialize()?;
                self.storage.store_transaction_with_slot(sig, slot, &tx_data)?;
            }
        }

        info!("Block {} persisted to storage", slot);

        Ok(())
    }

    /// Get parent block hash
    async fn get_parent_hash(&self, slot: u64) -> Result<Vec<u8>> {
        if slot == 0 {
            return Ok(vec![0u8; 32]); // Genesis
        }

        let parent_slot = slot - 1;
        
        if let Some(block_data) = self.storage.get_block(parent_slot)? {
            let block = Block::deserialize(&block_data)?;
            Ok(block.hash())
        } else {
            // Default to zero hash if parent not found
            Ok(vec![0u8; 32])
        }
    }

    /// Check if this validator is the leader for the slot
    async fn is_leader(&self, slot: u64) -> bool {
        // In production, check against leader schedule from PoS
        // For now, simplified check
        true
    }

    /// Get next slot
    async fn get_next_slot(&self) -> u64 {
        let mut current = self.current_slot.write().await;
        *current += 1;
        *current
    }

    /// Get validator public key
    pub fn get_public_key(&self) -> &ed25519_dalek::PublicKey {
        &self.keypair.public
    }

    /// Get current slot
    pub async fn get_current_slot(&self) -> u64 {
        *self.current_slot.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_validator_creation() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let validator = Validator::new(keypair, storage);
        assert!(validator.is_ok());
    }

    #[tokio::test]
    async fn test_validator_with_producer() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let mempool = Arc::new(Mempool::new(Default::default()));
        let fee_market = Arc::new(FeeMarket::new(Default::default()));
        
        let validator = Validator::new(keypair, storage).unwrap()
            .with_block_producer(mempool, fee_market, Default::default());
        
        assert!(validator.block_producer.is_some());
    }
}
