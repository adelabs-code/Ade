use axum::Json;
use crate::server::RpcState;
use crate::types::RpcResponse;

pub async fn get_slot(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    Json(RpcResponse::success(serde_json::json!(chain.current_slot)))
}

pub async fn get_block_height(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    Json(RpcResponse::success(serde_json::json!(chain.current_slot)))
}

pub async fn get_latest_blockhash(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    let blockhash = bs58::encode(&chain.latest_blockhash).into_string();
    Json(RpcResponse::success(serde_json::json!({
        "blockhash": blockhash,
        "lastValidBlockHeight": chain.current_slot + 150
    })))
}

pub async fn send_transaction(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing transaction parameter"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    let signature = format!("sig_{}", chain.transaction_count);
    Json(RpcResponse::success(serde_json::json!(signature)))
}

pub async fn get_transaction(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing signature parameter"));
    }

    Json(RpcResponse::success(serde_json::json!({
        "slot": 100,
        "transaction": {},
        "meta": {
            "err": null,
            "status": { "Ok": null }
        }
    })))
}

pub async fn get_balance(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing account parameter"));
    }

    Json(RpcResponse::success(serde_json::json!({
        "value": 1000000000
    })))
}

pub async fn get_account_info(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing account parameter"));
    }

    Json(RpcResponse::success(serde_json::json!({
        "lamports": 1000000000,
        "owner": "11111111111111111111111111111111",
        "executable": false,
        "rentEpoch": 0
    })))
}

pub async fn deploy_ai_agent(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing deployment parameters"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    Json(RpcResponse::success(serde_json::json!({
        "agentId": format!("agent_{}", chain.transaction_count),
        "signature": format!("sig_{}", chain.transaction_count)
    })))
}

pub async fn execute_ai_agent(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing execution parameters"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    Json(RpcResponse::success(serde_json::json!({
        "executionId": format!("exec_{}", chain.transaction_count),
        "signature": format!("sig_{}", chain.transaction_count),
        "computeUnits": 50000
    })))
}

pub async fn get_ai_agent_info(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing agent ID"));
    }

    Json(RpcResponse::success(serde_json::json!({
        "agentId": "agent_1",
        "modelHash": "QmXx...",
        "owner": "owner_pubkey",
        "executionCount": 42,
        "totalComputeUsed": 2100000
    })))
}

pub async fn bridge_deposit(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing deposit parameters"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    Json(RpcResponse::success(serde_json::json!({
        "depositId": format!("deposit_{}", chain.transaction_count),
        "signature": format!("sig_{}", chain.transaction_count)
    })))
}

pub async fn bridge_withdraw(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing withdrawal parameters"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    Json(RpcResponse::success(serde_json::json!({
        "withdrawalId": format!("withdraw_{}", chain.transaction_count),
        "signature": format!("sig_{}", chain.transaction_count)
    })))
}

