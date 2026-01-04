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

/// Validator with integrated block production and PoS leader selection
pub struct Validator {
    keypair: Keypair,
    storage: Arc<Storage>,
    block_producer: Option<Arc<BlockProducer>>,
    state_transition: Arc<StateTransition>,
    current_slot: Arc<RwLock<u64>>,
    /// Proof of stake consensus for leader selection
    proof_of_stake: Arc<RwLock<ProofOfStake>>,
    /// Epoch length in slots
    epoch_length: u64,
}

impl Validator {
    /// Default epoch length (number of slots per epoch)
    const DEFAULT_EPOCH_LENGTH: u64 = 432_000; // ~2 days at 400ms slots
    /// Default minimum stake requirement (in lamports)
    const DEFAULT_MIN_STAKE: u64 = 1_000_000_000; // 1 SOL equivalent

    pub fn new(keypair: Keypair, storage: Arc<Storage>) -> Result<Self> {
        let state_transition = Arc::new(StateTransition::new(storage.clone()));
        let proof_of_stake = ProofOfStake::new(Self::DEFAULT_MIN_STAKE, Self::DEFAULT_EPOCH_LENGTH);
        
        Ok(Self {
            keypair,
            storage,
            block_producer: None,
            state_transition,
            current_slot: Arc::new(RwLock::new(0)),
            proof_of_stake: Arc::new(RwLock::new(proof_of_stake)),
            epoch_length: Self::DEFAULT_EPOCH_LENGTH,
        })
    }

    /// Create a validator with custom PoS configuration
    pub fn with_pos_config(
        keypair: Keypair, 
        storage: Arc<Storage>,
        min_stake: u64,
        epoch_length: u64,
    ) -> Result<Self> {
        let state_transition = Arc::new(StateTransition::new(storage.clone()));
        let proof_of_stake = ProofOfStake::new(min_stake, epoch_length);
        
        Ok(Self {
            keypair,
            storage,
            block_producer: None,
            state_transition,
            current_slot: Arc::new(RwLock::new(0)),
            proof_of_stake: Arc::new(RwLock::new(proof_of_stake)),
            epoch_length,
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

    /// Register this validator in the PoS system
    pub async fn register_as_validator(&self, stake: u64, commission: u8) -> Result<()> {
        let validator_info = ade_consensus::ValidatorInfo {
            pubkey: self.keypair.public.to_bytes().to_vec(),
            stake,
            commission,
            last_vote_slot: 0,
            active: true,
            activated_epoch: 0,
            deactivation_epoch: None,
        };

        let mut pos = self.proof_of_stake.write().await;
        pos.register_validator(validator_info)?;
        
        info!("Registered validator with {} stake", stake);
        Ok(())
    }

    /// Get the proof of stake instance for external access
    pub fn get_proof_of_stake(&self) -> Arc<RwLock<ProofOfStake>> {
        self.proof_of_stake.clone()
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

    /// Check if this validator is the leader for the slot using PoS consensus
    async fn is_leader(&self, slot: u64) -> bool {
        let pos = self.proof_of_stake.read().await;
        
        // Get the leader for this slot from the PoS leader selection
        let leader = pos.select_leader(slot);
        
        match leader {
            Some(leader_pubkey) => {
                // Compare the selected leader with our public key
                let our_pubkey = self.keypair.public.to_bytes().to_vec();
                let is_leader = leader_pubkey == our_pubkey;
                
                if is_leader {
                    debug!("We are the leader for slot {}", slot);
                } else {
                    debug!(
                        "Slot {} leader is {}, we are {}",
                        slot,
                        bs58::encode(&leader_pubkey).into_string(),
                        bs58::encode(&our_pubkey).into_string()
                    );
                }
                
                is_leader
            }
            None => {
                // No validators registered, or no active validators
                // In a development/test environment, allow block production
                // In production, this would be false
                let active_validators = pos.get_active_validators();
                if active_validators.is_empty() {
                    debug!("No active validators, allowing block production for slot {}", slot);
                    true // Allow in test mode
                } else {
                    warn!("Could not determine leader for slot {}", slot);
                    false
                }
            }
        }
    }

    /// Get the leader schedule for the current epoch
    pub async fn get_leader_schedule(&self, epoch: u64) -> Vec<Vec<u8>> {
        let pos = self.proof_of_stake.read().await;
        pos.compute_leader_schedule(epoch)
    }

    /// Get the current epoch based on slot
    pub async fn get_current_epoch(&self) -> u64 {
        let slot = *self.current_slot.read().await;
        slot / self.epoch_length
    }

    /// Record a vote from this validator
    pub async fn record_vote(&self, slot: u64) -> Result<()> {
        let mut pos = self.proof_of_stake.write().await;
        let pubkey = self.keypair.public.to_bytes().to_vec();
        pos.record_vote(&pubkey, slot)?;
        debug!("Recorded vote for slot {}", slot);
        Ok(())
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

    #[tokio::test]
    async fn test_validator_registration() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let validator = Validator::with_pos_config(
            keypair,
            storage,
            100_000, // min stake
            1000,    // epoch length
        ).unwrap();

        // Register with sufficient stake
        let result = validator.register_as_validator(200_000, 5).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_is_leader_with_registration() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let validator = Validator::with_pos_config(
            keypair,
            storage,
            100_000,
            1000,
        ).unwrap();

        // Register this validator
        validator.register_as_validator(200_000, 5).await.unwrap();

        // Since we're the only validator, we should be the leader
        let is_leader = validator.is_leader(100).await;
        assert!(is_leader);
    }

    #[tokio::test]
    async fn test_leader_schedule() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
        
        let validator = Validator::with_pos_config(
            keypair,
            storage,
            100_000,
            10, // short epoch for testing
        ).unwrap();

        // Register this validator
        validator.register_as_validator(200_000, 5).await.unwrap();

        // Get leader schedule for epoch 0
        let schedule = validator.get_leader_schedule(0).await;
        assert_eq!(schedule.len(), 10); // epoch length
    }
}
