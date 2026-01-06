use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;

use ade_transaction::{Transaction, Account};
use ade_consensus::Block;

/// AI Runtime interface for executing AI agents
/// This trait is implemented by the actual node's AI runtime
#[async_trait::async_trait]
pub trait AIRuntimeInterface: Send + Sync {
    /// Ensure a model is available for execution (triggers P2P sync if needed)
    async fn ensure_model_for_execution(&self, model_hash: &[u8]) -> Result<()>;
    
    /// Execute an AI agent with the given request
    async fn execute_agent(&self, request: ExecutionRequest) -> Result<ExecutionResult>;
}

/// Request to execute an AI agent
#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub agent_id: Vec<u8>,
    pub input_data: Vec<u8>,
    pub max_compute: u64,
    pub caller: Vec<u8>,
}

/// Result of AI agent execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub output_data: Vec<u8>,
    pub compute_units_used: u64,
    pub execution_time_us: u64,
    pub success: bool,
}

/// Storage backend trait for RPC state persistence
/// 
/// This allows the RPC server to use different storage backends:
/// - In-memory (for testing)
/// - RocksDB (for production)
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    async fn get_account(&self, address: &[u8]) -> Option<Account>;
    async fn get_transaction(&self, signature: &[u8]) -> Option<TransactionInfo>;
    async fn get_block(&self, slot: u64) -> Option<Block>;
    async fn store_transaction(&self, info: TransactionInfo) -> Result<()>;
}

/// Enhanced RPC state with storage backend - PRODUCTION implementation
/// 
/// Contains all state needed for the RPC server to serve requests,
/// including connections to the actual node services (AI runtime, bridge, etc.)
/// 
/// PRODUCTION FEATURES:
/// - Optional RocksDB connection for persistence
/// - In-memory cache for hot data
/// - Automatic cache invalidation
#[derive(Clone)]
pub struct RpcStateBackend {
    pub chain_state: Arc<RwLock<ChainState>>,
    /// In-memory cache for frequently accessed accounts
    pub accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>,
    /// In-memory cache for recent transactions
    pub transactions: Arc<RwLock<HashMap<Vec<u8>, TransactionInfo>>>,
    /// In-memory cache for recent blocks
    pub blocks: Arc<RwLock<HashMap<u64, Block>>>,
    pub ai_agents: Arc<RwLock<HashMap<Vec<u8>, AIAgentData>>>,
    /// Bridge operations storage for tracking deposits/withdrawals
    pub bridge_operations: Arc<RwLock<HashMap<String, BridgeOperationState>>>,
    /// Connection to the actual AI runtime (None if AI execution is disabled)
    pub ai_runtime: Option<Arc<dyn AIRuntimeInterface>>,
    /// Optional persistent storage backend (RocksDB in production)
    pub persistent_storage: Option<Arc<dyn StorageBackend>>,
    /// Maximum entries to cache in memory
    max_cached_transactions: usize,
    max_cached_blocks: usize,
}

/// Bridge operation state for production tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeOperationState {
    pub id: String,
    pub operation_type: String, // "deposit" or "withdrawal"
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub token: Vec<u8>,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub status: BridgeStatus,
    pub source_tx_hash: Vec<u8>,
    pub target_tx_hash: Option<Vec<u8>>,
    pub created_at: u64,
    pub confirmed_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub confirmations: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BridgeStatus {
    Pending,
    Confirming,
    Confirmed,
    Executing,
    Completed,
    Failed,
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
    /// Create new in-memory only state (for testing)
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
            bridge_operations: Arc::new(RwLock::new(HashMap::new())),
            ai_runtime: None,
            persistent_storage: None,
            max_cached_transactions: 10_000,
            max_cached_blocks: 1_000,
        }
    }
    
    /// Create with persistent storage backend (PRODUCTION)
    /// 
    /// This connects the RPC state to RocksDB for persistence across restarts.
    pub fn with_storage(storage: Arc<dyn StorageBackend>) -> Self {
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
            bridge_operations: Arc::new(RwLock::new(HashMap::new())),
            ai_runtime: None,
            persistent_storage: Some(storage),
            max_cached_transactions: 10_000,
            max_cached_blocks: 1_000,
        }
    }
    
    /// Create with AI runtime connection
    pub fn with_ai_runtime(ai_runtime: Arc<dyn AIRuntimeInterface>) -> Self {
        let mut state = Self::new();
        state.ai_runtime = Some(ai_runtime);
        state
    }
    
    /// Connect AI runtime after construction
    pub fn connect_ai_runtime(&mut self, ai_runtime: Arc<dyn AIRuntimeInterface>) {
        self.ai_runtime = Some(ai_runtime);
    }
    
    /// Connect persistent storage after construction
    pub fn connect_storage(&mut self, storage: Arc<dyn StorageBackend>) {
        self.persistent_storage = Some(storage);
    }
    
    /// Get account - checks cache first, then persistent storage
    pub async fn get_account_with_fallback(&self, address: &[u8]) -> Option<Account> {
        // Check cache first
        {
            let cache = self.accounts.read().await;
            if let Some(account) = cache.get(address) {
                return Some(account.clone());
            }
        }
        
        // Fall back to persistent storage
        if let Some(ref storage) = self.persistent_storage {
            if let Some(account) = storage.get_account(address).await {
                // Cache the result
                let mut cache = self.accounts.write().await;
                cache.insert(address.to_vec(), account.clone());
                return Some(account);
            }
        }
        
        None
    }
    
    /// Get transaction - checks cache first, then persistent storage
    pub async fn get_transaction_with_fallback(&self, signature: &[u8]) -> Option<TransactionInfo> {
        // Check cache first
        {
            let cache = self.transactions.read().await;
            if let Some(tx) = cache.get(signature) {
                return Some(tx.clone());
            }
        }
        
        // Fall back to persistent storage
        if let Some(ref storage) = self.persistent_storage {
            if let Some(tx) = storage.get_transaction(signature).await {
                // Cache the result (with eviction if needed)
                let mut cache = self.transactions.write().await;
                if cache.len() >= self.max_cached_transactions {
                    // Remove a random entry (simple eviction)
                    if let Some(key) = cache.keys().next().cloned() {
                        cache.remove(&key);
                    }
                }
                cache.insert(signature.to_vec(), tx.clone());
                return Some(tx);
            }
        }
        
        None
    }
    
    /// Get bridge operation by ID
    pub async fn get_bridge_operation(&self, id: &str) -> Option<BridgeOperationState> {
        let ops = self.bridge_operations.read().await;
        ops.get(id).cloned()
    }
    
    /// Update bridge operation
    pub async fn update_bridge_operation(&self, op: BridgeOperationState) {
        let mut ops = self.bridge_operations.write().await;
        ops.insert(op.id.clone(), op);
    }
    
    /// Add new bridge operation
    pub async fn add_bridge_operation(&self, op: BridgeOperationState) {
        let mut ops = self.bridge_operations.write().await;
        ops.insert(op.id.clone(), op);
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





