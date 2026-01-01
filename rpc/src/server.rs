use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use tower_http::cors::{CorsLayer, Any};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::info;

use crate::handlers::*;
use crate::types::{RpcRequest, RpcResponse};

#[derive(Clone)]
pub struct RpcState {
    pub chain_state: Arc<RwLock<ChainState>>,
}

#[derive(Debug, Clone)]
pub struct ChainState {
    pub current_slot: u64,
    pub latest_blockhash: Vec<u8>,
    pub transaction_count: u64,
}

pub struct RpcServer {
    port: u16,
    state: RpcState,
}

impl RpcServer {
    pub fn new(port: u16) -> Self {
        let state = RpcState {
            chain_state: Arc::new(RwLock::new(ChainState {
                current_slot: 0,
                latest_blockhash: vec![0u8; 32],
                transaction_count: 0,
            })),
        };

        Self { port, state }
    }

    pub async fn start(self) -> Result<()> {
        let app = Router::new()
            .route("/", post(handle_rpc))
            .route("/health", get(health_check))
            .route("/metrics", get(metrics))
            .layer(CorsLayer::new().allow_origin(Any))
            .with_state(self.state);

        let addr = format!("0.0.0.0:{}", self.port);
        info!("RPC server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn handle_rpc(
    State(state): State<RpcState>,
    Json(request): Json<RpcRequest>,
) -> Json<RpcResponse> {
    match request.method.as_str() {
        // Block Methods
        "getSlot" => get_slot(state).await,
        "getBlockHeight" => get_block_height(state).await,
        "getBlock" => get_block(state, request.params).await,
        "getBlocks" => get_blocks(state, request.params).await,
        "getBlockTime" => get_block_time(state, request.params).await,
        "getFirstAvailableBlock" => get_first_available_block(state).await,
        "getLatestBlockhash" => get_latest_blockhash(state).await,
        "getBlockProduction" => get_block_production(state).await,
        "getBlockCommitment" => get_block_commitment(state, request.params).await,
        
        // Transaction Methods
        "sendTransaction" => send_transaction(state, request.params).await,
        "simulateTransaction" => simulate_transaction(state, request.params).await,
        "getTransaction" => get_transaction(state, request.params).await,
        "getTransactionCount" => get_transaction_count(state).await,
        "getRecentPerformanceSamples" => get_recent_performance_samples(state, request.params).await,
        "getSignatureStatuses" => get_signature_statuses(state, request.params).await,
        "getSignaturesForAddress" => get_signatures_for_address(state, request.params).await,
        
        // Account Methods
        "getBalance" => get_balance(state, request.params).await,
        "getAccountInfo" => get_account_info(state, request.params).await,
        "getMultipleAccounts" => get_multiple_accounts(state, request.params).await,
        "getProgramAccounts" => get_program_accounts(state, request.params).await,
        "getLargestAccounts" => get_largest_accounts(state, request.params).await,
        "getTokenAccountsByOwner" => get_token_accounts_by_owner(state, request.params).await,
        "getTokenSupply" => get_token_supply(state, request.params).await,
        
        // Validator & Staking Methods
        "getVoteAccounts" => get_vote_accounts(state).await,
        "getValidators" => get_validators(state).await,
        "getStakeActivation" => get_stake_activation(state, request.params).await,
        "getStakeMinimumDelegation" => get_stake_minimum_delegation(state).await,
        "getLeaderSchedule" => get_leader_schedule(state, request.params).await,
        "getEpochInfo" => get_epoch_info(state).await,
        "getEpochSchedule" => get_epoch_schedule(state).await,
        
        // Network & Cluster Methods
        "getClusterNodes" => get_cluster_nodes(state).await,
        "getVersion" => get_version(state).await,
        "getGenesisHash" => get_genesis_hash(state).await,
        "getIdentity" => get_identity(state).await,
        "getInflationGovernor" => get_inflation_governor(state).await,
        "getInflationRate" => get_inflation_rate(state).await,
        "getInflationReward" => get_inflation_reward(state, request.params).await,
        "getSupply" => get_supply(state).await,
        
        // AI Agent Methods
        "deployAIAgent" => deploy_ai_agent(state, request.params).await,
        "executeAIAgent" => execute_ai_agent(state, request.params).await,
        "getAIAgentInfo" => get_ai_agent_info(state, request.params).await,
        "updateAIAgent" => update_ai_agent(state, request.params).await,
        "listAIAgents" => list_ai_agents(state, request.params).await,
        "getAIAgentExecutions" => get_ai_agent_executions(state, request.params).await,
        
        // Bridge Methods
        "bridgeDeposit" => bridge_deposit(state, request.params).await,
        "bridgeWithdraw" => bridge_withdraw(state, request.params).await,
        "getBridgeStatus" => get_bridge_status(state, request.params).await,
        "getBridgeHistory" => get_bridge_history(state, request.params).await,
        "estimateBridgeFee" => estimate_bridge_fee(state, request.params).await,
        
        // Utility Methods
        "requestAirdrop" => request_airdrop(state, request.params).await,
        "minimumLedgerSlot" => minimum_ledger_slot(state).await,
        "getSlotLeaders" => get_slot_leaders(state, request.params).await,
        "getFeeForMessage" => get_fee_for_message(state, request.params).await,
        "getRecentPrioritizationFees" => get_recent_prioritization_fees(state, request.params).await,
        "getMaxRetransmitSlot" => get_max_retransmit_slot(state).await,
        "getMaxShredInsertSlot" => get_max_shred_insert_slot(state).await,
        
        _ => Json(RpcResponse::error("Method not found")),
    }
}

async fn health_check() -> &'static str {
    "OK"
}

async fn metrics(State(state): State<RpcState>) -> Json<serde_json::Value> {
    let chain = state.chain_state.read().await;
    Json(serde_json::json!({
        "slot": chain.current_slot,
        "transaction_count": chain.transaction_count,
        "tps": 8500,
        "validator_count": 5,
        "uptime_seconds": 86400
    }))
}
