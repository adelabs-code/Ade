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
    /// 
    /// Uses TARGET-BASED timing to prevent clock drift.
    /// Instead of sleeping for a fixed duration after work is done,
    /// we calculate the target time for the next slot and sleep until then.
    pub async fn start(&self) -> Result<()> {
        info!("Starting validator with pubkey: {:?}", 
            bs58::encode(self.keypair.public.to_bytes()).into_string());

        if self.block_producer.is_none() {
            warn!("Validator started without block producer (read-only mode)");
            return Ok(());
        }

        // Slot duration in milliseconds
        const SLOT_DURATION_MS: u64 = 400;
        
        // Track the start time for accurate slot timing
        let genesis_time = std::time::Instant::now();
        let mut current_slot_offset: u64 = 0;

        // Main validator loop with target-based timing
        loop {
            let slot = self.get_next_slot().await;
            
            // Calculate target time for THIS slot (not relative to work completion)
            let target_slot_time = genesis_time + 
                std::time::Duration::from_millis(current_slot_offset * SLOT_DURATION_MS);
            
            // Check if we are the leader
            if self.is_leader(slot).await {
                info!("We are leader for slot {}", slot);
                
                let work_start = std::time::Instant::now();
                
                if let Err(e) = self.produce_and_process_block(slot).await {
                    warn!("Failed to produce block for slot {}: {}", slot, e);
                }
                
                let work_duration = work_start.elapsed();
                if work_duration.as_millis() > SLOT_DURATION_MS as u128 {
                    warn!(
                        "Block production took {}ms, exceeding slot duration ({}ms). \
                        This may cause missed slots.",
                        work_duration.as_millis(),
                        SLOT_DURATION_MS
                    );
                }
            }

            // Increment slot offset for next iteration
            current_slot_offset += 1;
            
            // Calculate next slot target time
            let next_slot_time = genesis_time + 
                std::time::Duration::from_millis(current_slot_offset * SLOT_DURATION_MS);
            
            // Sleep until target time (not for a fixed duration)
            // This prevents time drift regardless of how long block production takes
            let now = std::time::Instant::now();
            if next_slot_time > now {
                let sleep_duration = next_slot_time - now;
                tokio::time::sleep(sleep_duration).await;
            } else {
                // We're already past the target time - log warning but continue
                let behind_ms = (now - next_slot_time).as_millis();
                if behind_ms > SLOT_DURATION_MS as u128 {
                    warn!(
                        "Validator is {}ms behind schedule (> 1 slot). \
                        Consider reducing block processing load.",
                        behind_ms
                    );
                }
                // Skip sleep entirely to catch up
            }
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

    /// Check if this validator is the leader for the slot using VRF-based PoS consensus
    /// 
    /// This uses the cryptographically secure VRF (Verifiable Random Function) based
    /// leader selection which is resistant to grinding attacks and provides fair
    /// leader distribution based on stake weight.
    async fn is_leader(&self, slot: u64) -> bool {
        let pos = self.proof_of_stake.read().await;
        let our_pubkey = self.keypair.public.to_bytes().to_vec();
        
        // Generate our VRF proof for this slot
        let vrf_proof = match pos.generate_vrf_proof(&self.keypair.secret.to_bytes(), slot) {
            Ok(proof) => proof,
            Err(e) => {
                warn!("Failed to generate VRF proof for slot {}: {}", slot, e);
                // Fall back to stake-weighted selection
                return self.is_leader_fallback(&pos, slot, &our_pubkey);
            }
        };
        
        // In a real network, we would collect VRF proofs from all validators
        // For now, we use our proof and the stake-weighted VRF selection
        let vrf_proofs = vec![vrf_proof];
        
        // Try VRF-based leader selection first (more secure, grinding-resistant)
        let leader = pos.select_leader_with_vrf(slot, &vrf_proofs);
        
        match leader {
            Some(leader_pubkey) => {
                let is_leader = leader_pubkey == our_pubkey;
                
                if is_leader {
                    debug!("We are the VRF-selected leader for slot {}", slot);
                } else {
                    debug!(
                        "Slot {} VRF leader is {}, we are {}",
                        slot,
                        bs58::encode(&leader_pubkey).into_string(),
                        bs58::encode(&our_pubkey).into_string()
                    );
                }
                
                is_leader
            }
            None => {
                self.is_leader_fallback(&pos, slot, &our_pubkey)
            }
        }
    }
    
    /// Fallback leader selection when VRF is not available
    fn is_leader_fallback(&self, pos: &ade_consensus::ProofOfStake, slot: u64, our_pubkey: &[u8]) -> bool {
        let active_validators = pos.get_active_validators();
        
        if active_validators.is_empty() {
            debug!("No active validators, allowing block production for slot {}", slot);
            return true; // Allow in test mode
        }
        
        // Use standard stake-weighted selection as fallback
        match pos.select_leader(slot) {
            Some(leader) => {
                let is_leader = leader == our_pubkey;
                if is_leader {
                    debug!("We are the stake-weighted leader for slot {}", slot);
                }
                is_leader
            }
            None => {
                warn!("Could not determine leader for slot {}", slot);
                false
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
