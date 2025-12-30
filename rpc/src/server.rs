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
        "getSlot" => get_slot(state).await,
        "getBlockHeight" => get_block_height(state).await,
        "getLatestBlockhash" => get_latest_blockhash(state).await,
        "sendTransaction" => send_transaction(state, request.params).await,
        "getTransaction" => get_transaction(state, request.params).await,
        "getBalance" => get_balance(state, request.params).await,
        "getAccountInfo" => get_account_info(state, request.params).await,
        "deployAIAgent" => deploy_ai_agent(state, request.params).await,
        "executeAIAgent" => execute_ai_agent(state, request.params).await,
        "getAIAgentInfo" => get_ai_agent_info(state, request.params).await,
        "bridgeDeposit" => bridge_deposit(state, request.params).await,
        "bridgeWithdraw" => bridge_withdraw(state, request.params).await,
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
    }))
}

