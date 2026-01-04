use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use ade_transaction::{Transaction, Account};
use ade_consensus::Block;

/// Enhanced RPC state with storage backend
#[derive(Clone)]
pub struct RpcStateBackend {
    pub chain_state: Arc<RwLock<ChainState>>,
    pub accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>,
    pub transactions: Arc<RwLock<HashMap<Vec<u8>, TransactionInfo>>>,
    pub blocks: Arc<RwLock<HashMap<u64, Block>>>,
    pub ai_agents: Arc<RwLock<HashMap<Vec<u8>, AIAgentData>>>,
}

#[derive(Debug, Clone)]
pub struct ChainState {
    pub current_slot: u64,
    pub latest_blockhash: Vec<u8>,
    pub transaction_count: u64,
    pub finalized_slot: u64,
    pub epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionInfo {
    pub signature: Vec<u8>,
    pub slot: u64,
    pub transaction: Vec<u8>, // Serialized transaction
    pub status: TransactionStatus,
    pub fee: u64,
    pub compute_units: u64,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Processed,
    Confirmed,
    Finalized,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIAgentData {
    pub agent_id: Vec<u8>,
    pub model_hash: String,
    pub owner: Vec<u8>,
    pub execution_count: u64,
    pub total_compute_used: u64,
    pub status: String,
    pub config: serde_json::Value,
    pub created_at: u64,
    /// Actual execution history records (limited to recent entries)
    pub execution_history: Vec<ExecutionRecord>,
}

/// Record of a single AI agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub execution_id: Vec<u8>,
    pub timestamp: u64,
    pub compute_used: u64,
    pub success: bool,
    pub input_hash: Vec<u8>,
}

impl RpcStateBackend {
    pub fn new() -> Self {
        Self {
            chain_state: Arc::new(RwLock::new(ChainState {
                current_slot: 0,
                latest_blockhash: vec![0u8; 32],
                transaction_count: 0,
                finalized_slot: 0,
                epoch: 0,
            })),
            accounts: Arc::new(RwLock::new(HashMap::new())),
            transactions: Arc::new(RwLock::new(HashMap::new())),
            blocks: Arc::new(RwLock::new(HashMap::new())),
            ai_agents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a transaction
    pub async fn add_transaction(&self, tx_info: TransactionInfo) {
        let mut txs = self.transactions.write().await;
        txs.insert(tx_info.signature.clone(), tx_info);
        
        // Update count
        let mut chain = self.chain_state.write().await;
        chain.transaction_count += 1;
    }

    /// Get transaction
    pub async fn get_transaction(&self, signature: &[u8]) -> Option<TransactionInfo> {
        let txs = self.transactions.read().await;
        txs.get(signature).cloned()
    }

    /// Add account
    pub async fn add_account(&self, address: Vec<u8>, account: Account) {
        let mut accounts = self.accounts.write().await;
        accounts.insert(address, account);
    }

    /// Get account
    pub async fn get_account(&self, address: &[u8]) -> Option<Account> {
        let accounts = self.accounts.read().await;
        accounts.get(address).cloned()
    }

    /// Add block
    pub async fn add_block(&self, block: Block) {
        let slot = block.header.slot;
        let mut blocks = self.blocks.write().await;
        blocks.insert(slot, block);
        
        // Update chain state
        let mut chain = self.chain_state.write().await;
        chain.current_slot = slot;
    }

    /// Get block
    pub async fn get_block(&self, slot: u64) -> Option<Block> {
        let blocks = self.blocks.read().await;
        blocks.get(&slot).cloned()
    }

    /// Add AI agent
    pub async fn add_ai_agent(&self, agent: AIAgentData) {
        let mut agents = self.ai_agents.write().await;
        agents.insert(agent.agent_id.clone(), agent);
    }

    /// Get AI agent
    pub async fn get_ai_agent(&self, agent_id: &[u8]) -> Option<AIAgentData> {
        let agents = self.ai_agents.read().await;
        agents.get(agent_id).cloned()
    }

    /// Update AI agent execution stats
    pub async fn update_ai_agent_stats(&self, agent_id: &[u8], compute_used: u64) {
        let mut agents = self.ai_agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            agent.execution_count += 1;
            agent.total_compute_used += compute_used;
        }
    }

    /// List AI agents by owner
    pub async fn list_ai_agents_by_owner(&self, owner: &[u8]) -> Vec<AIAgentData> {
        let agents = self.ai_agents.read().await;
        agents.values()
            .filter(|a| a.owner == owner)
            .cloned()
            .collect()
    }

    /// Get multiple accounts
    pub async fn get_multiple_accounts(&self, addresses: &[Vec<u8>]) -> Vec<Option<Account>> {
        let accounts = self.accounts.read().await;
        addresses.iter()
            .map(|addr| accounts.get(addr).cloned())
            .collect()
    }

    /// Get program accounts
    pub async fn get_program_accounts(&self, program_id: &[u8]) -> Vec<(Vec<u8>, Account)> {
        let accounts = self.accounts.read().await;
        accounts.iter()
            .filter(|(_, account)| account.owner == program_id)
            .map(|(addr, account)| (addr.clone(), account.clone()))
            .collect()
    }
}

impl Default for RpcStateBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_backend() {
        let state = RpcStateBackend::new();
        
        let tx_info = TransactionInfo {
            signature: vec![1u8; 64],
            slot: 100,
            transaction: vec![],
            status: TransactionStatus::Confirmed,
            fee: 5000,
            compute_units: 150,
            logs: vec![],
        };
        
        state.add_transaction(tx_info.clone()).await;
        
        let retrieved = state.get_transaction(&vec![1u8; 64]).await;
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_account_storage() {
        let state = RpcStateBackend::new();
        
        let account = Account::new(1000000, vec![0u8; 32]);
        let address = vec![1u8; 32];
        
        state.add_account(address.clone(), account).await;
        
        let retrieved = state.get_account(&address).await;
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_ai_agent_storage() {
        let state = RpcStateBackend::new();
        
        let agent = AIAgentData {
            agent_id: vec![1u8; 32],
            model_hash: "QmTest".to_string(),
            owner: vec![2u8; 32],
            execution_count: 0,
            total_compute_used: 0,
            status: "active".to_string(),
            config: serde_json::json!({}),
            created_at: 12345,
            execution_history: Vec::new(),
        };
        
        state.add_ai_agent(agent.clone()).await;
        
        let retrieved = state.get_ai_agent(&vec![1u8; 32]).await;
        assert!(retrieved.is_some());
    }
}





