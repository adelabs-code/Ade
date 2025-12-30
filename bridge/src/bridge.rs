use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use anyhow::Result;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub solana_rpc_url: String,
    pub sidechain_rpc_url: String,
    pub bridge_program_id: Vec<u8>,
    pub supported_tokens: Vec<Vec<u8>>,
}

pub struct Bridge {
    config: BridgeConfig,
    deposits: HashMap<Vec<u8>, DepositInfo>,
    withdrawals: HashMap<Vec<u8>, WithdrawalInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositInfo {
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub token: Vec<u8>,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub status: BridgeStatus,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalInfo {
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub token: Vec<u8>,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub status: BridgeStatus,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeStatus {
    Pending,
    Locked,
    Relayed,
    Completed,
    Failed(String),
}

impl Bridge {
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            deposits: HashMap::new(),
            withdrawals: HashMap::new(),
        }
    }

    pub fn initiate_deposit(
        &mut self,
        from_chain: String,
        to_chain: String,
        amount: u64,
        token: Vec<u8>,
        sender: Vec<u8>,
        recipient: Vec<u8>,
    ) -> Result<Vec<u8>> {
        if !self.config.supported_tokens.contains(&token) {
            return Err(anyhow::anyhow!("Token not supported"));
        }

        let deposit_id = self.generate_deposit_id(&sender, &recipient, amount);
        
        let deposit = DepositInfo {
            from_chain,
            to_chain,
            amount,
            token,
            sender,
            recipient,
            status: BridgeStatus::Pending,
            timestamp: current_timestamp(),
        };

        self.deposits.insert(deposit_id.clone(), deposit);
        info!("Deposit initiated: {:?}", bs58::encode(&deposit_id).into_string());

        Ok(deposit_id)
    }

    pub fn confirm_deposit(&mut self, deposit_id: &[u8]) -> Result<()> {
        let deposit = self.deposits.get_mut(deposit_id)
            .ok_or_else(|| anyhow::anyhow!("Deposit not found"))?;

        deposit.status = BridgeStatus::Locked;
        info!("Deposit confirmed and locked");

        Ok(())
    }

    pub fn initiate_withdrawal(
        &mut self,
        from_chain: String,
        to_chain: String,
        amount: u64,
        token: Vec<u8>,
        sender: Vec<u8>,
        recipient: Vec<u8>,
    ) -> Result<Vec<u8>> {
        let withdrawal_id = self.generate_withdrawal_id(&sender, &recipient, amount);
        
        let withdrawal = WithdrawalInfo {
            from_chain,
            to_chain,
            amount,
            token,
            sender,
            recipient,
            status: BridgeStatus::Pending,
            timestamp: current_timestamp(),
        };

        self.withdrawals.insert(withdrawal_id.clone(), withdrawal);
        info!("Withdrawal initiated: {:?}", bs58::encode(&withdrawal_id).into_string());

        Ok(withdrawal_id)
    }

    pub fn complete_withdrawal(&mut self, withdrawal_id: &[u8]) -> Result<()> {
        let withdrawal = self.withdrawals.get_mut(withdrawal_id)
            .ok_or_else(|| anyhow::anyhow!("Withdrawal not found"))?;

        withdrawal.status = BridgeStatus::Completed;
        info!("Withdrawal completed");

        Ok(())
    }

    fn generate_deposit_id(&self, sender: &[u8], recipient: &[u8], amount: u64) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(sender);
        hasher.update(recipient);
        hasher.update(&amount.to_le_bytes());
        hasher.update(&current_timestamp().to_le_bytes());
        hasher.finalize().to_vec()
    }

    fn generate_withdrawal_id(&self, sender: &[u8], recipient: &[u8], amount: u64) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"withdrawal");
        hasher.update(sender);
        hasher.update(recipient);
        hasher.update(&amount.to_le_bytes());
        hasher.update(&current_timestamp().to_le_bytes());
        hasher.finalize().to_vec()
    }

    pub fn get_deposit_status(&self, deposit_id: &[u8]) -> Option<&BridgeStatus> {
        self.deposits.get(deposit_id).map(|d| &d.status)
    }

    pub fn get_withdrawal_status(&self, withdrawal_id: &[u8]) -> Option<&BridgeStatus> {
        self.withdrawals.get(withdrawal_id).map(|w| &w.status)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

