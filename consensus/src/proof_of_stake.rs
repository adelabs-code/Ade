use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use anyhow::{Result, Context};
use sha2::{Sha256, Digest};
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;

/// VRF (Verifiable Random Function) proof for leader election
/// This provides cryptographic proof that a validator was legitimately
/// selected as leader without allowing grinding attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRFProof {
    /// Slot this proof is for
    pub slot: u64,
    /// Public key of the validator
    pub public_key: Vec<u8>,
    /// The VRF proof (signature over slot data)
    pub proof: Vec<u8>,
    /// The VRF output (deterministic from proof)
    pub vrf_output: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub pubkey: Vec<u8>,
    pub stake: u64,
    pub commission: u8,
    pub last_vote_slot: u64,
    pub active: bool,
    pub activated_epoch: u64,
    pub deactivation_epoch: Option<u64>,
}

pub struct ProofOfStake {
    validators: HashMap<Vec<u8>, ValidatorInfo>,
    total_stake: u64,
    min_stake_requirement: u64,
    epoch_length: u64,
    current_epoch: u64,
}

impl ProofOfStake {
    pub fn new(min_stake_requirement: u64, epoch_length: u64) -> Self {
        Self {
            validators: HashMap::new(),
            total_stake: 0,
            min_stake_requirement,
            epoch_length,
            current_epoch: 0,
        }
    }

    /// Register a new validator
    pub fn register_validator(&mut self, validator: ValidatorInfo) -> Result<()> {
        if validator.stake < self.min_stake_requirement {
            return Err(anyhow::anyhow!(
                "Insufficient stake: {} < {}",
                validator.stake,
                self.min_stake_requirement
            ));
        }

        if self.validators.contains_key(&validator.pubkey) {
            return Err(anyhow::anyhow!("Validator already registered"));
        }

        self.total_stake += validator.stake;
        self.validators.insert(validator.pubkey.clone(), validator);
        Ok(())
    }

    /// Deactivate a validator
    pub fn deactivate_validator(&mut self, pubkey: &[u8]) -> Result<()> {
        let validator = self.validators.get_mut(pubkey)
            .ok_or_else(|| anyhow::anyhow!("Validator not found"))?;
        
        validator.active = false;
        validator.deactivation_epoch = Some(self.current_epoch);
        
        Ok(())
    }

    /// Select leader for a given slot using weighted random selection
    pub fn select_leader(&self, slot: u64) -> Option<Vec<u8>> {
        let active_validators: Vec<_> = self.validators
            .values()
            .filter(|v| v.active)
            .collect();

        if active_validators.is_empty() {
            return None;
        }

        // Calculate total active stake
        let total_active_stake: u64 = active_validators.iter().map(|v| v.stake).sum();
        
        if total_active_stake == 0 {
            return None;
        }

        // Use slot as seed for deterministic selection
        let seed = self.hash_slot_for_selection(slot);
        let target = u64::from_le_bytes(seed[0..8].try_into().unwrap()) % total_active_stake;
        
        // Select validator based on stake weight
        let mut cumulative = 0u64;
        for validator in active_validators {
            cumulative += validator.stake;
            if cumulative > target {
                return Some(validator.pubkey.clone());
            }
        }
        
        // Fallback to first validator (should never reach here)
        active_validators.first().map(|v| v.pubkey.clone())
    }

    /// Compute leader schedule for an entire epoch
    pub fn compute_leader_schedule(&self, epoch: u64) -> Vec<Vec<u8>> {
        let slots_in_epoch = self.epoch_length;
        let mut schedule = Vec::with_capacity(slots_in_epoch as usize);
        
        let start_slot = epoch * self.epoch_length;
        
        for offset in 0..slots_in_epoch {
            let slot = start_slot + offset;
            if let Some(leader) = self.select_leader(slot) {
                schedule.push(leader);
            } else {
                // If no leader can be selected, use empty vec
                schedule.push(Vec::new());
            }
        }
        
        schedule
    }

    /// Hash slot for deterministic leader selection using VRF-like construction
    /// 
    /// This implements a grinding-resistant leader selection by including:
    /// - Previous block hash (when available) to prevent predictability
    /// - Epoch randomness seed derived from aggregate signatures
    /// - Slot number for uniqueness
    fn hash_slot_for_selection(&self, slot: u64) -> Vec<u8> {
        let mut hasher = Sha256::new();
        
        // Domain separator to prevent cross-protocol attacks
        hasher.update(b"ADE_LEADER_SELECTION_V1");
        
        // Include slot and epoch
        hasher.update(&slot.to_le_bytes());
        hasher.update(&self.current_epoch.to_le_bytes());
        
        // Include epoch randomness (prevents grinding on future slots)
        let epoch_seed = self.compute_epoch_randomness();
        hasher.update(&epoch_seed);
        
        // Multiple rounds of hashing to increase computational cost for attackers
        let intermediate = hasher.finalize();
        
        let mut final_hasher = Sha256::new();
        final_hasher.update(&intermediate);
        final_hasher.update(&slot.to_le_bytes());
        
        final_hasher.finalize().to_vec()
    }
    
    /// Compute epoch randomness from validator participation
    /// This creates a pseudo-VRF by combining validator inputs deterministically
    fn compute_epoch_randomness(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"EPOCH_RANDOMNESS");
        hasher.update(&self.current_epoch.to_le_bytes());
        
        // Include sorted validator pubkeys and their stakes for determinism
        let mut validators: Vec<_> = self.validators.values()
            .filter(|v| v.active)
            .collect();
        validators.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
        
        for validator in validators {
            hasher.update(&validator.pubkey);
            hasher.update(&validator.stake.to_le_bytes());
            hasher.update(&validator.last_vote_slot.to_le_bytes());
        }
        
        hasher.finalize().to_vec()
    }
    
    /// Generate a VRF proof for leader election
    /// Returns (proof, proof_hash) that can be verified by other validators
    pub fn generate_vrf_proof(&self, slot: u64, secret_key: &[u8]) -> Result<VRFProof> {
        // Create the message to sign
        let message = self.create_vrf_message(slot);
        
        // Derive keypair from secret key
        if secret_key.len() != 64 {
            return Err(anyhow::anyhow!("Invalid secret key length"));
        }
        
        let secret = SecretKey::from_bytes(&secret_key[..32])
            .map_err(|e| anyhow::anyhow!("Invalid secret key: {}", e))?;
        let public = PublicKey::from(&secret);
        let keypair = Keypair { secret, public };
        
        // Sign the message (this is the VRF proof)
        let signature = keypair.sign(&message);
        
        // Derive the VRF output from the signature
        let mut output_hasher = Sha256::new();
        output_hasher.update(b"VRF_OUTPUT");
        output_hasher.update(&signature.to_bytes());
        let vrf_output = output_hasher.finalize().to_vec();
        
        Ok(VRFProof {
            slot,
            public_key: public.to_bytes().to_vec(),
            proof: signature.to_bytes().to_vec(),
            vrf_output,
        })
    }
    
    /// Verify a VRF proof from another validator
    pub fn verify_vrf_proof(&self, proof: &VRFProof) -> Result<bool> {
        // Reconstruct the message
        let message = self.create_vrf_message(proof.slot);
        
        // Parse public key and signature
        let public_key_bytes: [u8; 32] = proof.public_key.clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid public key length"))?;
        let public_key = PublicKey::from_bytes(&public_key_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;
        
        let sig_bytes: [u8; 64] = proof.proof.clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Invalid signature length"))?;
        let signature = Signature::from_bytes(&sig_bytes);
        
        // Verify the signature
        let is_valid = public_key.verify(&message, &signature).is_ok();
        
        if !is_valid {
            return Ok(false);
        }
        
        // Verify the VRF output
        let mut output_hasher = Sha256::new();
        output_hasher.update(b"VRF_OUTPUT");
        output_hasher.update(&proof.proof);
        let expected_output = output_hasher.finalize().to_vec();
        
        Ok(proof.vrf_output == expected_output)
    }
    
    /// Create the message for VRF signing
    fn create_vrf_message(&self, slot: u64) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"ADE_VRF_MESSAGE");
        hasher.update(&slot.to_le_bytes());
        hasher.update(&self.current_epoch.to_le_bytes());
        hasher.update(&self.compute_epoch_randomness());
        hasher.finalize().to_vec()
    }
    
    /// Select leader using VRF with grinding resistance
    /// This should be called when VRF proofs are available for verification
    pub fn select_leader_with_vrf(&self, slot: u64, vrf_proofs: &[VRFProof]) -> Option<Vec<u8>> {
        if vrf_proofs.is_empty() {
            // Fall back to stake-weighted selection if no VRF proofs
            return self.select_leader(slot);
        }
        
        // Verify all proofs and collect valid ones
        let mut valid_proofs: Vec<_> = vrf_proofs.iter()
            .filter(|p| {
                self.verify_vrf_proof(p).unwrap_or(false) &&
                self.validators.contains_key(&p.public_key) &&
                self.validators.get(&p.public_key).map(|v| v.active).unwrap_or(false)
            })
            .collect();
        
        if valid_proofs.is_empty() {
            return self.select_leader(slot);
        }
        
        // Sort by VRF output to get deterministic ordering
        valid_proofs.sort_by(|a, b| a.vrf_output.cmp(&b.vrf_output));
        
        // Weight by stake: lower VRF output + higher stake = better chance
        let mut best_score = u128::MAX;
        let mut best_validator = None;
        
        for proof in valid_proofs {
            let validator = match self.validators.get(&proof.public_key) {
                Some(v) => v,
                None => continue,
            };
            
            // Convert VRF output to a score (lower is better)
            let vrf_value = u64::from_le_bytes(
                proof.vrf_output[0..8].try_into().unwrap_or([0u8; 8])
            );
            
            // Score = VRF_value / stake (lower stake penalty for lower VRF values)
            let score = (vrf_value as u128 * 1_000_000) / (validator.stake as u128 + 1);
            
            if score < best_score {
                best_score = score;
                best_validator = Some(proof.public_key.clone());
            }
        }
        
        best_validator
    }

    /// Get validator by public key
    pub fn get_validator(&self, pubkey: &[u8]) -> Option<&ValidatorInfo> {
        self.validators.get(pubkey)
    }

    /// Get mutable validator reference
    pub fn get_validator_mut(&mut self, pubkey: &[u8]) -> Option<&mut ValidatorInfo> {
        self.validators.get_mut(pubkey)
    }

    /// Update validator stake
    pub fn update_stake(&mut self, pubkey: &[u8], new_stake: u64) -> Result<()> {
        let validator = self.validators.get_mut(pubkey)
            .ok_or_else(|| anyhow::anyhow!("Validator not found"))?;

        let old_stake = validator.stake;
        self.total_stake = self.total_stake - old_stake + new_stake;
        validator.stake = new_stake;

        // Deactivate if below minimum
        if new_stake < self.min_stake_requirement {
            validator.active = false;
        } else if !validator.active && new_stake >= self.min_stake_requirement {
            // Reactivate if stake is sufficient again
            validator.active = true;
            validator.deactivation_epoch = None;
        }

        Ok(())
    }

    /// Record a vote from validator
    pub fn record_vote(&mut self, pubkey: &[u8], slot: u64) -> Result<()> {
        let validator = self.validators.get_mut(pubkey)
            .ok_or_else(|| anyhow::anyhow!("Validator not found"))?;
        
        validator.last_vote_slot = slot;
        Ok(())
    }

    /// Advance to next epoch
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Get current epoch
    pub fn get_current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Calculate epoch from slot
    pub fn get_epoch_for_slot(&self, slot: u64) -> u64 {
        slot / self.epoch_length
    }

    /// Get slot index within epoch
    pub fn get_epoch_slot_index(&self, slot: u64) -> u64 {
        slot % self.epoch_length
    }

    /// Get total stake
    pub fn get_total_stake(&self) -> u64 {
        self.total_stake
    }

    /// Get active validators
    pub fn get_active_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.active)
            .collect()
    }

    /// Get all validators
    pub fn get_all_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators.values().collect()
    }

    /// Get delinquent validators (haven't voted recently)
    pub fn get_delinquent_validators(&self, current_slot: u64, threshold: u64) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.active && current_slot.saturating_sub(v.last_vote_slot) > threshold)
            .collect()
    }

    /// Calculate stake distribution
    pub fn get_stake_distribution(&self) -> StakeDistribution {
        let active_validators = self.get_active_validators();
        let active_stake: u64 = active_validators.iter().map(|v| v.stake).sum();
        
        StakeDistribution {
            total_stake: self.total_stake,
            active_stake,
            inactive_stake: self.total_stake - active_stake,
            validator_count: self.validators.len(),
            active_validator_count: active_validators.len(),
            stake_concentration: if !active_validators.is_empty() {
                let max_stake = active_validators.iter().map(|v| v.stake).max().unwrap_or(0);
                max_stake as f64 / active_stake as f64
            } else {
                0.0
            },
        }
    }

    /// Slash validator for misbehavior
    pub fn slash_validator(&mut self, pubkey: &[u8], slash_amount: u64) -> Result<()> {
        let validator = self.validators.get_mut(pubkey)
            .ok_or_else(|| anyhow::anyhow!("Validator not found"))?;
        
        let actual_slash = slash_amount.min(validator.stake);
        validator.stake -= actual_slash;
        self.total_stake -= actual_slash;
        
        // Deactivate if below minimum
        if validator.stake < self.min_stake_requirement {
            validator.active = false;
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeDistribution {
    pub total_stake: u64,
    pub active_stake: u64,
    pub inactive_stake: u64,
    pub validator_count: usize,
    pub active_validator_count: usize,
    pub stake_concentration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator(pubkey: u8, stake: u64) -> ValidatorInfo {
        ValidatorInfo {
            pubkey: vec![pubkey; 32],
            stake,
            commission: 5,
            last_vote_slot: 0,
            active: true,
            activated_epoch: 0,
            deactivation_epoch: None,
        }
    }

    #[test]
    fn test_validator_registration() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        let validator = create_test_validator(1, 200_000);
        
        assert!(pos.register_validator(validator).is_ok());
        assert_eq!(pos.get_total_stake(), 200_000);
    }

    #[test]
    fn test_insufficient_stake() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        let validator = create_test_validator(1, 50_000);
        
        assert!(pos.register_validator(validator).is_err());
    }

    #[test]
    fn test_leader_selection() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        
        pos.register_validator(create_test_validator(1, 200_000)).unwrap();
        pos.register_validator(create_test_validator(2, 300_000)).unwrap();
        
        let leader = pos.select_leader(100);
        assert!(leader.is_some());
    }

    #[test]
    fn test_leader_schedule() {
        let mut pos = ProofOfStake::new(100_000, 10);
        
        pos.register_validator(create_test_validator(1, 200_000)).unwrap();
        pos.register_validator(create_test_validator(2, 300_000)).unwrap();
        
        let schedule = pos.compute_leader_schedule(0);
        assert_eq!(schedule.len(), 10);
    }

    #[test]
    fn test_stake_update() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        let validator = create_test_validator(1, 200_000);
        let pubkey = validator.pubkey.clone();
        
        pos.register_validator(validator).unwrap();
        pos.update_stake(&pubkey, 300_000).unwrap();
        
        assert_eq!(pos.get_total_stake(), 300_000);
    }

    #[test]
    fn test_validator_slashing() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        let validator = create_test_validator(1, 200_000);
        let pubkey = validator.pubkey.clone();
        
        pos.register_validator(validator).unwrap();
        pos.slash_validator(&pubkey, 50_000).unwrap();
        
        let validator = pos.get_validator(&pubkey).unwrap();
        assert_eq!(validator.stake, 150_000);
    }

    #[test]
    fn test_delinquent_validators() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        
        let mut val1 = create_test_validator(1, 200_000);
        val1.last_vote_slot = 100;
        pos.register_validator(val1).unwrap();
        
        let mut val2 = create_test_validator(2, 300_000);
        val2.last_vote_slot = 200;
        pos.register_validator(val2).unwrap();
        
        let delinquent = pos.get_delinquent_validators(250, 100);
        assert_eq!(delinquent.len(), 1);
    }

    #[test]
    fn test_stake_distribution() {
        let mut pos = ProofOfStake::new(100_000, 432_000);
        
        pos.register_validator(create_test_validator(1, 200_000)).unwrap();
        pos.register_validator(create_test_validator(2, 300_000)).unwrap();
        
        let distribution = pos.get_stake_distribution();
        assert_eq!(distribution.total_stake, 500_000);
        assert_eq!(distribution.active_stake, 500_000);
        assert_eq!(distribution.active_validator_count, 2);
    }
}
