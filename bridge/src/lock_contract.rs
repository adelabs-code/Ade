use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositEvent {
    pub depositor: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub target_chain: String,
    pub recipient: Vec<u8>,
    pub nonce: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalEvent {
    pub withdrawer: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub source_chain: String,
    pub nonce: u64,
}

pub struct LockContract {
    locked_funds: HashMap<Vec<u8>, u64>,
    nonce: u64,
}

impl LockContract {
    pub fn new() -> Self {
        Self {
            locked_funds: HashMap::new(),
            nonce: 0,
        }
    }

    pub fn lock_funds(&mut self, token: &[u8], amount: u64) -> Result<u64> {
        let current_locked = self.locked_funds.get(token).unwrap_or(&0);
        self.locked_funds.insert(token.to_vec(), current_locked + amount);
        
        self.nonce += 1;
        Ok(self.nonce)
    }

    pub fn unlock_funds(&mut self, token: &[u8], amount: u64) -> Result<()> {
        let current_locked = self.locked_funds.get(token)
            .ok_or_else(|| anyhow::anyhow!("No locked funds for this token"))?;

        if *current_locked < amount {
            return Err(anyhow::anyhow!("Insufficient locked funds"));
        }

        self.locked_funds.insert(token.to_vec(), current_locked - amount);
        Ok(())
    }

    pub fn get_locked_amount(&self, token: &[u8]) -> u64 {
        *self.locked_funds.get(token).unwrap_or(&0)
    }

    pub fn get_nonce(&self) -> u64 {
        self.nonce
    }
}

impl Default for LockContract {
    fn default() -> Self {
        Self::new()
    }
}



