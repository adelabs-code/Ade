use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub pubkey: Vec<u8>,
    pub stake: u64,
    pub commission: u8,
    pub last_vote_slot: u64,
    pub active: bool,
}

pub struct ProofOfStake {
    validators: HashMap<Vec<u8>, ValidatorInfo>,
    total_stake: u64,
    min_stake_requirement: u64,
}

impl ProofOfStake {
    pub fn new(min_stake_requirement: u64) -> Self {
        Self {
            validators: HashMap::new(),
            total_stake: 0,
            min_stake_requirement,
        }
    }

    pub fn register_validator(&mut self, validator: ValidatorInfo) -> Result<()> {
        if validator.stake < self.min_stake_requirement {
            return Err(anyhow::anyhow!(
                "Insufficient stake: {} < {}",
                validator.stake,
                self.min_stake_requirement
            ));
        }

        self.total_stake += validator.stake;
        self.validators.insert(validator.pubkey.clone(), validator);
        Ok(())
    }

    pub fn select_leader(&self, slot: u64) -> Option<Vec<u8>> {
        if self.validators.is_empty() {
            return None;
        }

        let active_validators: Vec<_> = self.validators
            .values()
            .filter(|v| v.active)
            .collect();

        if active_validators.is_empty() {
            return None;
        }

        let seed = slot as usize % active_validators.len();
        Some(active_validators[seed].pubkey.clone())
    }

    pub fn get_validator(&self, pubkey: &[u8]) -> Option<&ValidatorInfo> {
        self.validators.get(pubkey)
    }

    pub fn update_stake(&mut self, pubkey: &[u8], new_stake: u64) -> Result<()> {
        let validator = self.validators.get_mut(pubkey)
            .ok_or_else(|| anyhow::anyhow!("Validator not found"))?;

        self.total_stake = self.total_stake - validator.stake + new_stake;
        validator.stake = new_stake;

        if new_stake < self.min_stake_requirement {
            validator.active = false;
        }

        Ok(())
    }

    pub fn get_total_stake(&self) -> u64 {
        self.total_stake
    }

    pub fn get_active_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.active)
            .collect()
    }
}

